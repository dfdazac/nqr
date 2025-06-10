import logging
from itertools import pairwise

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
import sys
import json
sys.path.append('rp')
from .kbc.src.models import ComplEx

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
    def __init__(self, args, device, adj_list, query_name_dict, name_answer_dict, preference_embedding="mean", num_layers=2, activation="relu", margin=0.1):
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
            # p = torch.full((args.nentity, args.nentity), 0.001)
            # relation_embedding = torch.bernoulli(p)
            # relation_embedding = (relation_embedding>=1).to(torch.float) * 0.9999 + (relation_embedding<1).to(torch.float) * relation_embedding
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

        self.preference_embedding = preference_embedding
        self.num_layers = num_layers
        self.margin = margin
        if preference_embedding != "none":
            if activation == "relu":
                activation_class = nn.ReLU
            elif activation == "elu":
                activation_class = nn.ELU
            else:
                raise ValueError(f"Invalid activation function {activation}")
            if preference_embedding == "selfattn":
                self.self_attention_1 = nn.MultiheadAttention(embed_dim=embedding_dim + 1, num_heads=1,
                                                              batch_first=True)
                self.layer_norm_1 = nn.LayerNorm(embedding_dim + 1)
                self.fc1 = nn.Sequential(nn.Linear(embedding_dim + 1, embedding_dim), activation_class())
                if self.num_layers == 2:
                    self.self_attention_2 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
                    self.layer_norm_2 = nn.LayerNorm(embedding_dim)

                fc2 = nn.Linear(embedding_dim * 2 + 1, embedding_dim)  # Input: [preference_embedding, entity_embedding, score]
            elif preference_embedding == "mean":
                fc2 = nn.Linear(embedding_dim + 1 + embedding_dim + 1, embedding_dim)  # Input: [preference_embedding, preference_label, entity_embedding, score]
            else:
                raise ValueError(f"Invalid preference embedding type {preference_embedding}")
            self.adjust_net = nn.Sequential(fc2,
                                            activation_class(),
                                            nn.Linear(embedding_dim, 1),
                                            nn.Tanh())

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

    def find_ans(self, exec_query, query_structure, anchor):
        ans_structure = self.name_answer_dict[self.query_name_dict[query_structure]]
        return self.backward_ans(ans_structure, exec_query, anchor)

    def backward_ans(self, ans_structure, exec_query, anchor):
        if ans_structure == 'e': # 'e'
            return exec_query, exec_query

        elif ans_structure[0] == 'u': # 'u'
            return ['u'], 'u'

        elif ans_structure[0] == 'r': # ['r', 'e', 'r']
            cur_ent = anchor
            ans = []
            for ele, query_ele in zip(ans_structure[::-1], exec_query[::-1]):
                if ele == 'r':
                    r_id, r_argmax = query_ele
                    ans.append(r_id)
                    cur_ent = int(r_argmax[cur_ent])
                elif ele == 'n':
                    ans.append('n')
                else:
                    ans.append(cur_ent)
            return ans[::-1], cur_ent

        elif ans_structure[1][0] == 'r': # [[...], ['r', ...], 'e']
            r_ans, r_ent = self.backward_ans(ans_structure[1], exec_query[1], anchor)
            e_ans, e_ent = self.backward_ans(ans_structure[0], exec_query[0], r_ent)
            ans = [e_ans, r_ans, anchor]
            return ans, e_ent

        else: # [[...], [...], 'e']
            ans = []
            for ele, query_ele in zip(ans_structure[:-1], exec_query[:-1]):
                ele_ans, ele_ent = self.backward_ans(ele, query_ele, anchor)
                ans.append(ele_ans)
            ans.append(anchor)
            return ans, ele_ent

    def rerank_random(self, scores, preferences, labels):
        # Randomly permute the scores
        idx = torch.randperm(scores.size(0))
        scores = scores[idx]
        return scores

    def rerank_greedy(self, scores, preferences, labels):
        max_score = scores.max()
        min_score = scores.min()
        scores[preferences[labels == 1]] = max_score + 1
        scores[preferences[labels == 0]] = min_score - 1
        return scores

    def rerank_cosine(self, scores, preferences, labels, alpha_p, alpha_n):
        similarities = self.kbc_model.compute_similarities(preferences)

        positive_similarities = similarities[labels == 1].sum(dim=0)
        negative_similarities = -similarities[labels == 0].sum(dim=0)

        scores = scores * alpha_p + positive_similarities * (1 - alpha_p) + scores * alpha_n + negative_similarities * (1 - negative_similarities)

        return scores

    def rerank_fuzzi(self, scores, preferences, labels):
        similarities = self.kbc_model.compute_similarities(preferences, kind="real")

        # similarities are in theory in (-1.0, 1.0), but let's prevent numerical issues
        similarities = torch.clamp(similarities, -1.0 + 1e-9, 1.0 - 1e-9)
        # arctan maps to (-inf, inf), and sigmoid to (0, 1)
        similarities = torch.sigmoid(torch.arctanh(similarities))

        positive_similarities = similarities[labels == 1].prod(dim=0)
        negative_similarities = (1.0 - similarities[labels == 0]).prod(dim=0)

        scores = scores * positive_similarities * negative_similarities

        return scores

    def rerank_nqr(self, scores, preferences, labels):
        # == Part 1: Embed preferences ==
        m = self.embed_preferences(preferences.unsqueeze(0), labels.unsqueeze(0))
        embeddings = self.kbc_model.embeddings[0].weight
        m = m.expand(embeddings.shape[0], -1)
        scores = self.adjust_scores(m, embeddings, scores)
        return scores

    def embed_preferences(self, preferences, labels):
        """Retrieve embeddings of entities specified as preferences"""
        preferences_mask = preferences < 0
        preferences[preferences_mask] = 0

        # == Part 1: Compute preference embeddings m ==
        m = self.kbc_model.embeddings[0](preferences)
        # (batch_size, num_preferences, embedding_dim)
        m = torch.cat([m, labels.unsqueeze(-1)], dim=-1)
        # (batch_size, num_preferences, embedding_dim + 1)

        # # Alternative 1: simple mean
        if self.preference_embedding == "mean":
            m = torch.mean(m, dim=1)
            # (batch_size, embedding_dim + 1)
        elif self.preference_embedding == "selfattn":
            # Alternative 2: self attention and mean pooling
            m = self.self_attention_1(m, m, m, key_padding_mask=preferences_mask)[0]
            m = self.layer_norm_1(m)
            m = self.fc1(m)
            if self.num_layers == 2:
                m = self.self_attention_2(m, m, m, key_padding_mask=preferences_mask)[0]
                m = self.layer_norm_2(m)
            # (batch_size, num_preferences, embedding_dim)
            m = torch.mean(m, dim=1)
            # (batch_size, embedding_dim)
        else:
            raise ValueError(f"Invalid preference embedding type {self.preference_embedding}")

        return m

    def adjust_scores(self, pref_embeddings, candidates, scores, return_deltas=False):
        inputs = torch.cat([pref_embeddings, candidates, scores.unsqueeze(-1)], dim=1)
        # (p * batch_size, 2 * embedding_dim)
        score_deltas = self.adjust_net(inputs).squeeze(-1)
        # (p * batch_size, 1)
        new_scores = scores + score_deltas
        # (p * batch_size)
        if return_deltas:
            return new_scores, score_deltas
        else:
            return new_scores

    def _ranknet_loss(self, pos_scores, neg_scores, pos_batch_id, neg_batch_id, batch_size, margin_loss=True):
        pairwise_diff = neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
        # (p * batch_size, n * batch_size)
        batch_mask = pos_batch_id.unsqueeze(1) == neg_batch_id.unsqueeze(0)
        # (p * batch_size, n * batch_size)

        if margin_loss:
            loss = torch.clamp(self.margin + pairwise_diff, min=0) * batch_mask
            # (p * batch_size, n * batch_size)
        else:
            # RankNet loss: BCE applied to sigmoid of pairwise differences (pos - neg)
            loss = -F.logsigmoid(-pairwise_diff) * batch_mask

        # Sum over negatives
        loss = torch.sum(loss, dim=1)
        # (p * batch_size,)

        # Sum over positives for each element in batch
        batch_loss = torch.scatter_add(input=torch.zeros(batch_size, device=pos_scores.device),
                                       dim=0,
                                       index=pos_batch_id,
                                       src=loss)

        # Average per element in batch
        pairs_per_batch = torch.scatter_add(input=torch.zeros(batch_size, device=pos_scores.device),
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

    def reranking_loss(self, preferences, labels, scores, pos_data, neg_data, use_nqr=True):
        """
        Computes the margin-based loss for reranking.

        Args:
            preferences (Tensor): Padded tensor of preference set inputs. (batch_size, num_preferences)
            labels (Tensor): Corresponding labels for the preferences. (batch_size, num_preferences)
            scores (Tensor): Initial scores from the base QA model, shape (batch_size, num_entities).
            positives (Tensor): Padded tensor of positively preferred entities.
            negatives (Tensor): Padded tensor of negatively preferred entities.

        Returns:
            Tensor: Computed loss.
        """
        batch_size = preferences.shape[0]
        device = preferences.device

        # == Part 1: Embed preferences ==
        m = self.embed_preferences(preferences, labels)

        # == Part 2: Compute score adjustments for positive and negatives ==
        positives, pos_batch_id = pos_data
        negatives, neg_batch_id = neg_data

        pos_scores = scores[pos_batch_id, positives]
        new_pos_scores, pos_deltas = self.adjust_scores(pref_embeddings=m[pos_batch_id],
                                                    candidates=self.kbc_model.embeddings[0](positives),
                                                    scores=pos_scores,
                                                    return_deltas=True)
        # (p * batch_size)

        neg_scores = scores[neg_batch_id, negatives]
        new_neg_scores, neg_deltas = self.adjust_scores(pref_embeddings=m[neg_batch_id],
                                                    candidates=self.kbc_model.embeddings[0](negatives),
                                                    scores=neg_scores,
                                                    return_deltas=True)
        # (n * batch_size)

        # Preference loss: rank positives higher than negatives
        preference_loss = self._ranknet_loss(new_pos_scores, new_neg_scores, pos_batch_id, neg_batch_id, batch_size,
                                             margin_loss=use_nqr)

        # == Part 3: Compute score adjustments for random examples ==
        # Get negative examples
        random_indices, random_batch_id = self._get_negative_samples(pos_data, neg_data, batch_size)
        random_scores = scores[random_batch_id, random_indices]
        new_random_scores, random_deltas = self.adjust_scores(pref_embeddings=m[random_batch_id],
                                                              candidates=self.kbc_model.embeddings[0](random_indices),
                                                              scores=random_scores,
                                                              return_deltas=True)
        # (batch_size * num_random_examples)

        if use_nqr:
            # Answer loss: preserve global rankings by minimizing KL divergence
            prev_scores = self._collect_batched_scores(scores=torch.cat([pos_scores, neg_scores, random_scores]),
                                                       batch_ids=torch.cat([pos_batch_id, neg_batch_id, random_batch_id]),
                                                       batch_size=batch_size)
            new_scores = self._collect_batched_scores(scores=torch.cat([new_pos_scores, new_neg_scores, new_random_scores]),
                                                      batch_ids=torch.cat([pos_batch_id, neg_batch_id, random_batch_id]),
                                                      batch_size=batch_size)
            # (batch_size, max_entities), padded with -inf
            answer_loss = F.kl_div(input=torch.log_softmax(new_scores, dim=-1),
                                   target=torch.log_softmax(prev_scores, dim=-1),
                                   reduction='batchmean',
                                   log_target=True)
        else:
            # Normal Ranknet objective
            answer_scores = torch.cat([new_pos_scores, new_neg_scores])
            # (p * batch_size + n * batch_size)
            answer_batch_id = torch.cat([pos_batch_id, neg_batch_id])
            # (p * batch_size + n * batch_size)
            answer_loss = self._ranknet_loss(answer_scores, new_random_scores, answer_batch_id, random_batch_id,
                                             batch_size, margin_loss=False)

        all_deltas = pos_deltas, neg_deltas, random_deltas

        return preference_loss, answer_loss, all_deltas
