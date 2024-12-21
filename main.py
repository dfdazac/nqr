from collections import Counter
from typing import Literal

import numpy as np
import torch
from gqs.dataset import Dataset as GQSDataset
from gqs.loader import TorchQuery, get_query_datasets
from gqs.sample import resolve_sample
from scipy.cluster.hierarchy import linkage
from sentence_transformers import SentenceTransformer
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset


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
    metadata = GQSDataset(args.dataset)
    # Load KG and query data
    dataset, information = get_query_datasets(
        metadata,
        train=map(resolve_sample, args.train_data),
        validation=map(resolve_sample, args.valid_data),
        test=map(resolve_sample, args.test_data)
    )

    embeddings_filename = get_embeddings_filename(args.embedding_model)
    emb_data = torch.load(f"datasets/{args.dataset}/mapping/{embeddings_filename}")
    embeddings = emb_data["embeddings"]
    descriptions = emb_data["descriptions"]
    entity_to_row = emb_data["entity_to_row"]

    # Cluster answers to queries
    for split in ["train", "validation", "test"]:
        for query in tqdm(dataset[split], desc=f"Generating {split}"):
            print(query)
            query: TorchQuery
            if split == "train":
                target_ids = query.easy_targets
            else:
                target_ids = query.hard_targets

            subsets = []
            if len(target_ids) >= args.num_answers_threshold:
                # Embed targets
                entities = [metadata.entity_mapper.inverse_lookup(t.item()) for t in target_ids]
                entity_rows = [entity_to_row[e] for e in entities]
                target_embeddings = embeddings[entity_rows]

                # Find clusters within answers to each query in batch
                z = linkage(target_embeddings, method="average", metric="cosine")

                # Compute a mapping from cluster number to data points in it
                n = target_embeddings.shape[0]
                cluster_map = {i: [i] for i in range(n)}
                for i, row in enumerate(z):
                    cluster_1, cluster_2 = int(row[0]), int(row[1])
                    new_cluster = cluster_map[cluster_1] + cluster_map[cluster_2]
                    cluster_map[n + i] = new_cluster  # Update cluster map

                # Traverse the hierarchy from top to bottom, collecting clusters
                cluster_threshold = int(0.2 * n)

                all_indices_set = {i for i in range(n)}
                for i in range(len(z) - 1, -1, -1):
                    u, v, *_ = z[i].astype(int)

                    for cluster_id in (u, v):
                        positives = cluster_map[cluster_id]
                        if len(positives) >= cluster_threshold:
                            negatives = all_indices_set.difference(set(positives))

                            subsets.append([positives, list(negatives)])


if __name__ == "__main__":
    args = Arguments().parse_args()
    if args.command == COMMAND_EMBED:
        embed(args)
    elif args.command == COMMAND_GENERATE:
        generate(args)
