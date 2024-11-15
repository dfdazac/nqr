from collections import Counter

import torch
from gqs.dataset import Dataset as GQSDataset
from gqs.loader import QueryGraphBatch, get_query_data_loaders
from gqs.sample import resolve_sample
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from tap import Tap


class Arguments(Tap):
    dataset: str = "fb15k237"
    train_data: list[str] = ["/1hop/0qual:1000"]
    valid_data: list[str] = ["/1hop/0qual:100"]
    test_data: list[str] = ["/1hop/0qual:100"]

    num_answers_threshold: int = 10

    batch_size: int = 4
    num_workers: int = 0


def main(args: Arguments):
    # Load KG and query data
    dataset = GQSDataset(args.dataset)
    data_loaders, information = get_query_data_loaders(
        dataset=dataset,
        train=map(resolve_sample, args.train_data),
        validation=map(resolve_sample, args.valid_data),
        test=map(resolve_sample, args.test_data),
        batch_size=32,
        num_workers=args.num_workers,
    )

    # Load entity descriptions (long ones have priority)
    entity_to_text = dict()
    text_files = ["entity2textlong.txt", "entity2text.txt"]
    for file in text_files:
        with open(f"datasets/{args.dataset}/mapping/{file}") as f:
            for line in f:
                entity, text = line.strip().split("\t")
                if entity not in entity_to_text:
                    entity_to_text[entity] = text

    # Embed and cluster answers to queries
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for batch in data_loaders["train"]:
        batch: QueryGraphBatch

        # Proceed only with queries that have a number of targets
        # greater than the threshold
        target_graph_ids = batch.easy_targets[0, :]
        graph_id_to_num_targets = Counter(target_graph_ids.tolist())
        graph_ids_above_threshold = []
        for graph_id, num_targets in graph_id_to_num_targets.items():
            if num_targets > args.num_answers_threshold:
                graph_ids_above_threshold.append(graph_id)

        if len(graph_ids_above_threshold) == 0:
            # No query in this batch with num_targets above threshold, move on
            continue

        graph_ids_above_threshold = torch.tensor(graph_ids_above_threshold)
        above_threshold_mask = graph_ids_above_threshold.unsqueeze(1) == target_graph_ids.unsqueeze(0)
        above_threshold_mask = above_threshold_mask.any(dim=0)
        targets_above_threshold = batch.easy_targets[:, above_threshold_mask]

        # Embed targets
        graph_ids = targets_above_threshold[0]
        target_ids = targets_above_threshold[1]
        batch_texts = []
        for t in target_ids:
            entity = dataset.entity_mapper.inverse_lookup(t.item())
            text = entity_to_text[entity]
            batch_texts.append(text)

        batch_embeddings = embedder.encode(batch_texts,
                                           batch_size=args.batch_size)

        # Find clusters within answers to each query in batch
        for graph_id in graph_ids.unique():
            targets_mask = (graph_id == graph_ids).numpy()
            target_embeddings = batch_embeddings[targets_mask]
            
            best_k = None
            best_bic = float('inf')
            best_gmm = None

            k_min = 2
            k_max = target_embeddings.shape[0]

            # Find the optimal number of clusters (according to BIC)
            for k in range(k_min, min(k_max, 20) + 1):
                gmm = GaussianMixture(n_components=k, random_state=0)
                gmm.fit(target_embeddings)
                bic = gmm.bic(target_embeddings)

                if bic < best_bic:
                    best_bic = bic
                    best_k = k
                    best_gmm = gmm

            print(f'Optimal number of clusters (k): {best_k}')
            print(f'BIC score: {best_bic}')
            clusters = best_gmm.predict(target_embeddings)


main(Arguments().parse_args())
