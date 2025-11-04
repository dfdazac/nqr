import os
import sys

import lightgbm as lgb
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm


sys.path.append('rp')
from .kbc.src.models import ComplEx


def inverse_softplus(y):
    return torch.log(torch.exp(torch.tensor(y)) - 1.0)

def load_kbc(model_path, device, nentity, nrelation):
    model = ComplEx(sizes=[nentity, nrelation, nentity], rank=1000, init_size=1e-3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

@torch.no_grad()
def kge_forward(model, h, r, device, nentity):
    bsz = h.size(0)
    r = r.unsqueeze(-1).repeat(bsz, 1)
    h = h.unsqueeze(-1)
    positive_sample = torch.cat((h, r, h), dim=1)
    score = model(positive_sample, score_rhs=True, score_rel=False, score_lhs=False)
    return score[0]

@torch.no_grad()
def neural_adj_matrix(model, rel, nentity, device, thrshd, adj_list):
    bsz = 100
    softmax = nn.Softmax(dim=1)
    relation_embedding = torch.zeros(nentity, nentity).to(torch.float)
    r = torch.LongTensor([rel]).to(device)
    num = torch.zeros(nentity, 1).to(torch.float).to(device)
    for (h, t) in adj_list:
        num[h, 0] += 1
    num = torch.maximum(num, torch.ones(nentity, 1).to(torch.float).to(device))
    for s in range(0, nentity, bsz):
        t = min(nentity, s+bsz)
        h = torch.arange(s, t).to(device)
        score = kge_forward(model, h, r, device, nentity)
        normalized_score = softmax(score) * num[s:t, :]
        mask = (normalized_score >= thrshd).to(torch.float)
        normalized_score = mask * normalized_score
        relation_embedding[s:t, :] = normalized_score.to('cpu')
    return relation_embedding


class KGReasoning(nn.Module):
    def __init__(self, args, device, adj_list, query_name_dict, name_answer_dict, use_reranker=False, hidden_dim=256, activation="relu"):
        super(KGReasoning, self).__init__()
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.device = device
        self.relation_embeddings = list()
        self.fraction = args.fraction
        self.query_name_dict = query_name_dict
        self.name_answer_dict = name_answer_dict
        self.neg_scale = args.neg_scale
        self.kbc_model = load_kbc(args.kbc_path, device, args.nentity, args.nrelation)
        dataset_name = args.data_path.split('/')[-1]
        forced_str = '_forced' if args.force_training_edges else '_nonforced'
        filename = 'neural_adj/'+dataset_name+'_'+str(args.fraction)+'_'+str(args.thrshd)+forced_str+'.pt'
        print(f"Looking for neural adjacency {filename}...")
        if not os.path.exists('neural_adj'):
            os.makedirs('neural_adj')
        if os.path.exists(filename):
            print(f"Loading neural adjacency matrix from {filename}")
            self.relation_embeddings = torch.load(filename, map_location=device)
        else:
            print("Neural adjacency matrix not found.")
            for i in tqdm(range(args.nrelation), desc="Building neural adjacency matrix"):
                relation_embedding = neural_adj_matrix(self.kbc_model, i, args.nentity, device, args.thrshd, adj_list[i])

                relation_embedding[torch.nonzero(relation_embedding >= 1, as_tuple=True)] = 0.9999

                if args.force_training_edges:
                    for (h, t) in adj_list[i]:
                        relation_embedding[h, t] = 1.

                # add fractional
                fractional_relation_embedding = []
                dim = args.nentity // args.fraction
                rest = args.nentity - args.fraction * dim
                for i in range(args.fraction):
                    s = i * dim
                    t = (i+1) * dim
                    if i == args.fraction - 1:
                        t += rest
                    fractional_relation_embedding.append(relation_embedding[s:t, :].to_sparse().to(self.device))

                self.relation_embeddings.append(fractional_relation_embedding)
            torch.save(self.relation_embeddings, filename)

        embedding_dim = self.kbc_model.rank * 2
        if use_reranker:
            if activation == "relu":
                activation_class = nn.ReLU
            elif activation == "elu":
                activation_class = nn.ELU
            else:
                raise ValueError(f"Invalid activation function {activation}")
            
            # Neural network to compute adjustment weights (alpha_p, alpha_n)
            # Input: [score, avg_pos_sim, avg_neg_sim, entity_embedding]
            input_dim = 1 + 1 + 1 + embedding_dim
            self.adjust_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_class(),
                nn.Linear(hidden_dim, 2),
                nn.Sigmoid()
            )

            # initialize bias so that alpha_p ≈ 0.75 and alpha_n ≈ 0.25
            with torch.no_grad():
                last_linear = self.adjust_net[2]  # nn.Linear(hidden_dim, 2)
                last_linear.bias[0] = inverse_softplus(0.75)
                last_linear.bias[1] = inverse_softplus(0.25)
                last_linear.weight.zero_()

    def relation_projection(self, embedding, r_embedding, is_neg=False):
        dim = self.nentity // self.fraction
        rest = self.nentity - self.fraction * dim
        new_embedding = torch.zeros_like(embedding).to(self.device)
        r_argmax = torch.zeros(self.nentity).to(self.device)
        for i in range(self.fraction):
            s = i * dim
            t = (i+1) * dim
            if i == self.fraction - 1:
                t += rest
            fraction_embedding = embedding[:, s:t]
            if fraction_embedding.sum().item() == 0:
                continue
            nonzero = torch.nonzero(fraction_embedding, as_tuple=True)[1]
            fraction_embedding = fraction_embedding[:, nonzero]
            fraction_r_embedding = r_embedding[i].to_dense()[nonzero, :].unsqueeze(0)
            if is_neg:
                fraction_r_embedding = torch.minimum(torch.ones_like(fraction_r_embedding).to(torch.float), self.neg_scale*fraction_r_embedding)
                fraction_r_embedding = 1. - fraction_r_embedding
            fraction_embedding_premax = fraction_r_embedding * fraction_embedding.unsqueeze(-1)
            fraction_embedding, tmp_argmax = torch.max(fraction_embedding_premax, dim=1)
            tmp_argmax = nonzero[tmp_argmax.squeeze()] + s
            new_argmax = (fraction_embedding > new_embedding).to(torch.long).squeeze()
            r_argmax = new_argmax * tmp_argmax + (1-new_argmax) * r_argmax
            new_embedding = torch.maximum(new_embedding, fraction_embedding)
        return new_embedding, r_argmax.cpu().numpy()

    def intersection(self, embeddings):
        return torch.prod(embeddings, dim=0)

    def union(self, embeddings):
        return (1. - torch.prod(1.-embeddings, dim=0))

    def embed_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        exec_query = []
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                bsz = queries.size(0)
                embedding = torch.zeros(bsz, self.nentity).to(torch.float).to(self.device)
                embedding.scatter_(-1, queries[:, idx].unsqueeze(-1), 1)
                exec_query.append(queries[:, idx].item())
                idx += 1
            else:
                embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[0], idx)
                exec_query.append(pre_exec_query)
            r_exec_query = []
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    r_exec_query.append('n')
                else:
                    r_embedding = self.relation_embeddings[queries[0, idx]]
                    if (i < len(query_structure[-1]) - 1) and query_structure[-1][i+1] == 'n':
                        embedding, r_argmax = self.relation_projection(embedding, r_embedding, True)
                    else:
                        embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                    r_exec_query.append((queries[0, idx].item(), r_argmax))
                    r_exec_query.append('e')
                idx += 1
            r_exec_query.pop()
            exec_query.append(r_exec_query)
            exec_query.append('e')
        else:
            embedding_list = []
            union_flag = False
            for ele in query_structure[-1]:
                if ele == 'u':
                    union_flag = True
                    query_structure = query_structure[:-1]
                    break
            for i in range(len(query_structure)):
                embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                exec_query.append(pre_exec_query)
            if union_flag:
                embedding = self.union(torch.stack(embedding_list))
                idx += 1
                exec_query.append(['u'])
            else:
                embedding = self.intersection(torch.stack(embedding_list))
            exec_query.append('e')

        return embedding, idx, exec_query

    def rerank_cosine(self, scores, preferences, labels, alpha=0.3, beta=0.0):
        similarities = self.kbc_model.compute_similarities(preferences)

        positive_similarities = torch.zeros_like(scores)
        negative_similarities = torch.zeros_like(scores)
        pos_labels_mask = labels == 1
        neg_labels_mask = labels == 0

        if pos_labels_mask.sum().item() > 0:
            positive_similarities = similarities[pos_labels_mask].mean(dim=0)
        if neg_labels_mask.sum().item() > 0:
            negative_similarities = similarities[neg_labels_mask].mean(dim=0)

        w_p = (1 + beta) / 2.0
        w_n = (1 - beta) / 2.0
        preference_scores = w_p * positive_similarities - w_n * negative_similarities
        scores = alpha * scores + (1.0 - alpha) * preference_scores

        return scores

    def adjust_scores(self, scores, avg_pos_sim, avg_neg_sim, entity_embeddings, return_deltas=False):
        """
        Neural reranking using individual similarities.

        Computes s'[e] = s[e] + alpha_p * avg_pos_sim - alpha_n * avg_neg_sim
        where (alpha_p, alpha_n) are predicted by a neural network.
        """
        # Prepare input for neural network: [score, avg_pos_sim, avg_neg_sim, entity_embedding]
        network_input = torch.cat([
            scores.unsqueeze(-1),
            avg_pos_sim.unsqueeze(-1),
            avg_neg_sim.unsqueeze(-1),
            entity_embeddings
        ], dim=-1)
        # (num_entities, 1 + 1 + 1 + embedding_dim)

        # Predict adjustment weights
        alphas = self.adjust_net(network_input)
        # (num_entities, 2)
        alpha_p = alphas[:, 0]
        alpha_n = alphas[:, 1]

        # Compute adjusted scores
        score_deltas = alpha_p * avg_pos_sim - alpha_n * avg_neg_sim
        # score_deltas = 0.75 * avg_pos_sim - 0.25 * avg_neg_sim
        adjusted_scores = scores + score_deltas

        if return_deltas:
            return adjusted_scores, score_deltas
        else:
            return adjusted_scores

    def rerank_nqr(self, scores, preferences, labels):
        similarities = self.kbc_model.compute_similarities(preferences)

        positive_similarities = torch.zeros_like(scores)
        negative_similarities = torch.zeros_like(scores)
        pos_labels_mask = labels == 1
        neg_labels_mask = labels == 0

        if pos_labels_mask.sum().item() > 0:
            positive_similarities = similarities[pos_labels_mask].mean(dim=0)
        if neg_labels_mask.sum().item() > 0:
            negative_similarities = similarities[neg_labels_mask].mean(dim=0)

        embeddings = self.kbc_model.embeddings[0].weight
        scores = self.adjust_scores(scores, positive_similarities, negative_similarities, embeddings)
        return scores

    def _ranknet_loss(self, hi_scores, lo_scores, pos_batch_id, neg_batch_id, batch_size, margin_loss=False):
        pairwise_diff = lo_scores.unsqueeze(0) - hi_scores.unsqueeze(1)
        # (p * batch_size, n * batch_size)
        batch_mask = pos_batch_id.unsqueeze(1) == neg_batch_id.unsqueeze(0)
        # (p * batch_size, n * batch_size)

        # RankNet loss: BCE applied to sigmoid of pairwise differences (pos - neg)
        loss = -F.logsigmoid(-pairwise_diff) * batch_mask.float()

        # Sum over negatives
        loss = torch.sum(loss, dim=1)
        # (p * batch_size,)

        # Sum over positives for each element in batch
        batch_loss = torch.scatter_add(input=torch.zeros(batch_size, device=hi_scores.device),
                                       dim=0,
                                       index=pos_batch_id,
                                       src=loss)

        # Average per element in batch
        pairs_per_batch = torch.scatter_add(input=torch.zeros(batch_size, device=hi_scores.device),
                                            dim=0,
                                            index=pos_batch_id,
                                            src=batch_mask.float().sum(dim=1))
        batch_loss = batch_loss / pairs_per_batch

        loss = batch_loss.mean()
        return loss

    def _get_negative_samples(self, pos_data, neg_data, batch_size, num_negatives=100):
        positives, pos_batch_id = pos_data
        negatives, neg_batch_id = neg_data
        device = pos_batch_id.device

        random_batch_id = torch.repeat_interleave(torch.arange(batch_size, device=device), num_negatives)
        sampling_probabilities = torch.ones(batch_size, self.nentity, device=device)
        # Make sure that the sampled entities are not in the answer set
        sampling_probabilities[pos_batch_id, positives] = 0
        sampling_probabilities[neg_batch_id, negatives] = 0
        # Normalize probabilities for each row to sum to 1
        sampling_probabilities /= sampling_probabilities.sum(dim=1, keepdim=True)
        random_indices = torch.multinomial(sampling_probabilities, num_negatives, replacement=True)
        random_indices = random_indices.view(-1)

        return random_indices, random_batch_id

    def _get_mean_similarities(self, candidates, candidate_batch_id, preferences, prefs_batch_id):
        answer_preference_batch_mask = candidate_batch_id.unsqueeze(1) == prefs_batch_id.unsqueeze(0)
        answer_preference_pairs = torch.stack([
            candidates.unsqueeze(1).expand(-1, preferences.shape[0])[answer_preference_batch_mask],
            preferences.unsqueeze(0).expand(candidates.shape[0], -1)[answer_preference_batch_mask]
        ], dim=1)

        similarities = self.kbc_model.compute_similarities_from_pairs(answer_preference_pairs)

        rows, _ = answer_preference_batch_mask.nonzero(as_tuple=True)
        num_rows = candidates.shape[0]

        sums = torch.zeros(num_rows, device=similarities.device)
        counts = torch.zeros(num_rows, device=similarities.device)

        sums.index_add_(0, rows, similarities)
        counts.index_add_(0, rows, torch.ones_like(similarities))

        mean_similarities = sums / counts.clamp_min(1.0)

        return mean_similarities

    def _get_new_scores(self, scores, in_cluster_entities, in_cluster_batch_id, preferences, labels, prefs_batch_id):
        pos_labels_mask = labels == 1
        neg_labels_mask = labels == 0

        in_cluster_scores = scores[in_cluster_batch_id, in_cluster_entities]
        in_cluster_pos_similarities = torch.zeros_like(in_cluster_scores)
        in_cluster_neg_similarities = torch.zeros_like(in_cluster_scores)

        if pos_labels_mask.any() > 0:
            in_cluster_pos_similarities = self._get_mean_similarities(in_cluster_entities,
                                                                      in_cluster_batch_id,
                                                                      preferences[pos_labels_mask],
                                                                      prefs_batch_id[pos_labels_mask])
        if neg_labels_mask.any() > 0:
            in_cluster_neg_similarities = self._get_mean_similarities(in_cluster_entities,
                                                                      in_cluster_batch_id,
                                                                      preferences[neg_labels_mask],
                                                                      prefs_batch_id[neg_labels_mask])

        new_in_cluster_scores, in_cluster_deltas = self.adjust_scores(in_cluster_scores,
                                                                      in_cluster_pos_similarities,
                                                                      in_cluster_neg_similarities,
                                                                      self.kbc_model.embeddings[0](in_cluster_entities),
                                                                      return_deltas=True)

        return new_in_cluster_scores, in_cluster_deltas

    @staticmethod
    def _collect_batched_scores(scores, batch_ids, batch_size):
        device = scores.device
        batch_ids_range = torch.arange(len(batch_ids), device=device)
        batch_one_hot = torch.zeros((len(batch_ids), batch_size), dtype=torch.int64, device=device)
        batch_one_hot[batch_ids_range, batch_ids] = 1
        sequence_nums = batch_one_hot.cumsum(0) * batch_one_hot
        sequence_nums = (sequence_nums - 1).clamp(min=0)
        sequence_nums = sequence_nums[batch_ids_range, batch_ids]

        # Get maximum number of entities per batch
        max_entities = sequence_nums.max() + 1

        # Create output matrix filled with -1
        batched_scores = torch.full((batch_size, max_entities), -1e-9, device=device)

        # Fill the matrix with scores
        batched_scores[batch_ids, sequence_nums] = scores

        return batched_scores

    def reranking_loss(self, pref_data, scores, in_cluster_data, out_cluster_data):
        """
        Computes the margin-based loss for reranking.

        Args:
            pref_data: tuple (preferences, labels, prefs_batch_id)
            scores: (batch_size, num_entities) base model scores.
            in_cluster_data: tuple (in_cluster_entities, in_cluster_batch_id)
            out_cluster_data: tuple (out_cluster_entities, out_cluster_batch_id)

        Returns:
            Tensor: Computed loss.
        """
        preferences, labels, prefs_batch_id = pref_data
        in_cluster_entities, in_cluster_batch_id = in_cluster_data
        out_cluster_entities, out_cluster_batch_id = out_cluster_data
        batch_size = scores.shape[0]

        new_in_cluster_scores, in_cluster_deltas = self._get_new_scores(
            scores,
            in_cluster_entities,
            in_cluster_batch_id,
            preferences,
            labels,
            prefs_batch_id
        )

        new_out_cluster_scores, out_cluster_deltas = self._get_new_scores(
            scores,
            out_cluster_entities,
            out_cluster_batch_id,
            preferences,
            labels,
            prefs_batch_id
        )

        random_entities, random_batch_id = self._get_negative_samples(in_cluster_data, out_cluster_data, batch_size)
        new_random_scores, random_deltas = self._get_new_scores(
            scores,
            random_entities,
            random_batch_id,
            preferences,
            labels,
            prefs_batch_id
        )

        preference_loss = self._ranknet_loss(new_in_cluster_scores, new_out_cluster_scores, in_cluster_batch_id, out_cluster_batch_id, batch_size)

        answer_scores = torch.cat([new_in_cluster_scores, new_out_cluster_scores])
        answer_batch_id = torch.cat([in_cluster_batch_id, out_cluster_batch_id])
        answer_loss = self._ranknet_loss(answer_scores, new_random_scores, answer_batch_id, random_batch_id, batch_size)

        all_deltas = in_cluster_deltas, out_cluster_deltas, random_deltas

        return preference_loss, answer_loss, all_deltas

    def extract_features(self, scores, preferences, labels):
        """
        Extract features for LightGBM: [base_score, mean_pos_sim, mean_neg_sim].

        Args:
            scores: (num_entities,) base scores for all entities
            preferences: entity IDs of preference feedback
            labels: labels for preferences (1=positive, 0=negative)

        Returns:
            features: (num_entities, 3) feature matrix
        """
        similarities = self.kbc_model.compute_similarities(preferences)

        pos_labels_mask = labels == 1
        neg_labels_mask = labels == 0

        mean_pos_sim = torch.zeros_like(scores)
        mean_neg_sim = torch.zeros_like(scores)

        if pos_labels_mask.sum().item() > 0:
            mean_pos_sim = similarities[pos_labels_mask].mean(dim=0)
        if neg_labels_mask.sum().item() > 0:
            mean_neg_sim = similarities[neg_labels_mask].mean(dim=0)

        features = torch.stack([scores, mean_pos_sim, mean_neg_sim], dim=1)
        return features

    def train_lightgbm(self, training_data):
        """
        Train a LightGBM ranker on training data.

        Args:
            training_data: list of (features, relevance, group_size) tuples

        Returns:
            Trained LightGBM booster
        """

        # Aggregate all training data
        all_features = []
        all_relevance = []
        all_groups = []

        for features, relevance, group_size in training_data:
            all_features.append(features)
            all_relevance.append(relevance)
            all_groups.append(group_size)

        X = np.vstack(all_features)
        y = np.hstack(all_relevance)
        groups = all_groups

        n_groups = len(groups)
        group_indices = np.arange(n_groups)
        train_groups, val_groups = train_test_split(group_indices, test_size=0.2, random_state=42)

        group_boundaries = np.cumsum([0] + list(groups))

        def get_split_data(selected_groups):
            idx = np.concatenate([
                np.arange(group_boundaries[g], group_boundaries[g + 1])
                for g in selected_groups
            ])
            subset_groups = [groups[g] for g in selected_groups]
            return X[idx], y[idx], subset_groups

        X_train, y_train, groups_train = get_split_data(train_groups)
        X_val, y_val, groups_val = get_split_data(val_groups)

        train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
        valid_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)

        def objective(trial):
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [10],
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "feature_fraction": 1.0,  # small feature space, use all
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 3, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 0.5),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 0.5),
                "label_gain": [0, 1, 3],
                "feature_pre_filter": False,
                "verbose": -1,
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[valid_data]
            )

            return model.best_score["valid_0"]["ndcg@10"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        print("Best trial parameters:")
        print(study.best_trial.params)
        print(f"Best NDCG@10: {study.best_value:.4f}")

        best_params = {
            **study.best_trial.params,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "label_gain": [0, 1, 3],
            "verbose": -1,
        }

        full_train_data = lgb.Dataset(X, label=y, group=groups)
        booster = lgb.train(
            best_params,
            full_train_data,
            num_boost_round=100
        )

        return booster

    def rerank_lightgbm(self, scores, preferences, labels, lgb_model):
        """
        Rerank scores using a trained LightGBM model.

        Args:
            scores: (num_entities,) base scores
            preferences: entity IDs of preference feedback
            labels: labels for preferences (1=positive, 0=negative)
            lgb_model: trained LightGBM booster

        Returns:
            reranked_scores: (num_entities,) adjusted scores
        """
        features = self.extract_features(scores, preferences, labels)
        X = features.cpu().numpy()
        predictions = lgb_model.predict(X)
        return torch.from_numpy(predictions).to(scores.device)
