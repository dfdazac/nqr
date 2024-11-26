from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from gqs.dataset import Dataset as GQSDataset
from gqs.loader import QueryGraphBatch, get_query_data_loaders
from gqs.sample import resolve_sample
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from shc import shc_linkage


COMMAND_EMBED = "embed"
COMMAND_GENERATE = "generate"
COMMANDS = Literal[COMMAND_EMBED, COMMAND_GENERATE]


class Arguments(Tap):
    command: COMMANDS

    dataset: str = "fb15k237"
    train_data: list[str] = ["/1hop/0qual:1000"]
    valid_data: list[str] = ["/1hop/0qual:100"]
    test_data: list[str] = ["/1hop/0qual:100"]
    embedding_model: str = "dunzhang/stella_en_400M_v5"

    num_answers_threshold: int = 10

    batch_size: int = 64
    num_workers: int = 0

    def configure(self):
        self.add_argument("command")


def get_embeddings_filename(embedding_model: str) -> str:
    return f"{embedding_model.replace('/', '--')}-embeddings.pt"


def embed(args: Arguments):
    """Load descriptions of entities from a text file and embed them using a
    SentenceTransformer model. File is expected to have one row per entity,
    with the first column being the entity ID and the second its description,
    separated by a tab character."""
    def collate_fn(items: list[dict[str, str]]):
        all_entities = []
        all_descriptions = []
        for it in items:
            entity, description = it["text"].strip().split("\t")
            all_entities.append(entity)
            all_descriptions.append(description)

        return {"entities": all_entities, "descriptions": all_descriptions}

    entity_set = set()
    all_embeddings = []
    all_descriptions = []
    entity_to_row = {}
    num_rows = 0

    embedder = SentenceTransformer(args.embedding_model,
                                   trust_remote_code=True)
    text_files = ["entity2textlong.txt", "entity2text.txt"]

    for file in text_files:
        file_path = f"datasets/{args.dataset}/mapping/{file}"
        dataset = load_dataset("text",
                               data_files=file_path,
                               split="train")
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn)

        for i, batch in enumerate(tqdm(loader, desc=f"Embedding {file}")):
            entities_to_embed = []
            descriptions_to_embed = []
            for entity, description in zip(batch["entities"], batch["descriptions"]):
                if entity not in entity_set:
                    entity_set.add(entity)
                    entities_to_embed.append(entity)
                    descriptions_to_embed.append(description)

            if len(entities_to_embed) == 0:
                continue

            embeddings = embedder.encode(descriptions_to_embed)
            all_embeddings.append(embeddings)
            all_descriptions.extend(descriptions_to_embed)
            entity_to_row.update({e: num_rows + i for i, e in enumerate(entities_to_embed)})
            num_rows += len(entities_to_embed)

    all_embeddings = np.concatenate(all_embeddings)
    embeddings_filename = get_embeddings_filename(args.embedding_model)
    torch.save({"embeddings": all_embeddings,
                "descriptions": all_descriptions,
                "entity_to_row": entity_to_row},
               f"datasets/{args.dataset}/mapping/{embeddings_filename}")


def generate(args: Arguments):
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

    embeddings_filename = get_embeddings_filename(args.embedding_model)
    emb_data = torch.load(f"datasets/{args.dataset}/mapping/{embeddings_filename}")
    embeddings = emb_data["embeddings"]
    descriptions = emb_data["descriptions"]
    entity_to_row = emb_data["entity_to_row"]

    neighbors_search = NearestNeighbors(n_neighbors=5, metric="cosine")
    neighbors_search.fit(embeddings)

    # Cluster answers to queries
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

        entities = [dataset.entity_mapper.inverse_lookup(t.item()) for t in target_ids]
        entity_rows = [entity_to_row[e] for e in entities]
        batch_embeddings = embeddings[entity_rows]
        batch_texts = [descriptions[r] for r in entity_rows]

        # Find clusters within answers to each query in batch
        for graph_id in graph_ids.unique():
            targets_mask = (graph_id == graph_ids).numpy()
            target_embeddings = batch_embeddings[targets_mask]

            labels = []
            for emb_id, batch_id in enumerate(targets_mask.nonzero()[0]):
                print("\t", emb_id, batch_texts[batch_id][:100])
                labels.append(batch_texts[batch_id][:150])

            z, p_values = shc_linkage(target_embeddings, method="average", metric="cosine")
            print(np.hstack([z, np.expand_dims(p_values, 1)]))

            # Plot the dendrogram to visually inspect possible clusters
            plt.figure(figsize=(20, 7))
            dendrogram(z, labels=labels, orientation="left")
            plt.xlabel("Query targets")
            plt.ylabel("Euclidean Distance")
            plt.subplots_adjust(left=0.0, right=0.3)
            plt.show()

            input("Press Enter to continue...")


if __name__ == "__main__":
    args = Arguments().parse_args()
    if args.command == COMMAND_EMBED:
        embed(args)
    elif args.command == COMMAND_GENERATE:
        generate(args)
