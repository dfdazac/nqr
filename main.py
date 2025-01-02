from typing import Literal
import os.path as osp

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
from sentence_transformers import SentenceTransformer
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt

from datasets import load_dataset


COMMAND_EMBED = "embed"
COMMAND_CLUSTER = "cluster"
COMMANDS = Literal[COMMAND_EMBED, COMMAND_CLUSTER]


class Arguments(Tap):
    command: COMMANDS

    data_path: str
    embedding_model: str = "dunzhang/stella_en_400M_v5"

    num_answers_threshold: int = 10
    plot: bool = False

    batch_size: int = 64
    num_workers: int = 0

    def configure(self):
        # Positional arguments
        self.add_argument("command")
        self.add_argument("data_path")


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
        file_path = osp.join(args.data_path, file)
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
    out_path = osp.join(args.data_path, embeddings_filename)
    torch.save({"embeddings": all_embeddings,
                "descriptions": all_descriptions,
                "entity_to_row": entity_to_row},
                out_path)


def cluster(args: Arguments):
    # Load entity embeddings
    embeddings_filename = get_embeddings_filename(args.embedding_model)
    emb_data = torch.load(osp.join(args.data_path, embeddings_filename))
    embeddings = emb_data["embeddings"]
    descriptions = emb_data["descriptions"]
    entity_to_row = emb_data["entity_to_row"]

    # Load queries
    splits = ["train", "valid", "test"]
    queries = dict()
    answers = dict()
    for split in splits:
        with open(osp.join(args.data_path, f"{split}-queries.pkl"), "rb") as f:
            queries[split] = pkl.load(f)
        kind = f"-hard" if split != "train" else ""
        with open(osp.join(args.data_path, f"{split}{kind}-answers.pkl"), "rb") as f:
            answers[split] = pkl.load(f)
    with open(osp.join(args.data_path, "id2ent.pkl"), "rb") as f:
        id2ent = pkl.load(f)
    with open(osp.join(args.data_path, "id2rel.pkl"), "rb") as f:
        id2rel = pkl.load(f)

    # Cluster answers to queries
    for split in splits:
        subsets = dict()
        all_queries = queries[split][('e', ('r',))]

        for query in tqdm(all_queries, desc=f"Generating {split}"):
            target_ids = list(answers[split][query])

            query_subsets = []
            if len(target_ids) >= args.num_answers_threshold:
                subject, predicate = query[0], query[1][0]
                if args.plot:
                    print(query)
                    print(f"Subject: [{id2ent[subject]}] {descriptions[entity_to_row[id2ent[subject]]][:150]}")
                    print(f"Predicate: {id2rel[predicate]}")

                # Embed targets
                entities = [id2ent[t] for t in target_ids]
                entity_rows = [entity_to_row[e] for e in entities]
                target_embeddings = embeddings[entity_rows]
                labels = [descriptions[r][:150] for r in entity_rows]
                if args.plot:
                    for ent_id, l in zip(entities, labels):
                        print(f"\t [{ent_id}] {l}")

                # Find clusters within answers to each query in batch
                z = linkage(target_embeddings, method="average", metric="cosine")

                # Plot the dendrogram to visually inspect possible clusters
                if args.plot:
                    plt.figure(figsize=(20, 7))
                    dendrogram(z, labels=labels, orientation="left")
                    plt.xlabel("Query targets")
                    plt.ylabel("Euclidean Distance")
                    plt.subplots_adjust(left=0.0, right=0.3)
                    plt.show()

                # Compute a mapping from cluster number to data points in it
                n = target_embeddings.shape[0]
                cluster_map = {i: [i] for i in range(n)}
                for i, row in enumerate(z):
                    cluster_1, cluster_2 = int(row[0]), int(row[1])
                    new_cluster = cluster_map[cluster_1] + cluster_map[cluster_2]
                    cluster_map[n + i] = new_cluster  # Update cluster map

                # Traverse the hierarchy from top to bottom collecting clusters
                cluster_threshold = int(0.2 * n)

                all_indices_set = {i for i in range(n)}
                for i in range(len(z) - 1, -1, -1):
                    u, v, *_ = z[i].astype(int)

                    for cluster_id in (u, v):
                        pos_idx = cluster_map[cluster_id]
                        if len(pos_idx) >= cluster_threshold:
                            neg_idx = all_indices_set.difference(set(pos_idx))

                            positives = [target_ids[i] for i in pos_idx]
                            negatives = [target_ids[i] for i in neg_idx]

                            query_subsets.append((positives, negatives))

            subsets[query] = query_subsets
            if args.plot and len(query_subsets) > 0:
                for subset in query_subsets:
                    for kind, ids in zip(("Positives", "Negatives"), subset):
                        print(kind)
                        for id in ids:
                            print(f"\t[{id2ent[id]}] {descriptions[entity_to_row[id2ent[id]]][:150]}")

                a = input("Press enter to continue")

        with open(osp.join(args.data_path, f"{split}-subsets.pkl"), "wb") as f:
            pkl.dump(subsets, f)

        queries_with_subsets = sum(map(lambda s: len(s) > 0, subsets.values()))
        print(f"Generated subsets for {queries_with_subsets:,} out of {len(all_queries):,} queries")
        print(f"Total number of subsets: {sum(map(len, subsets.values())):,}")
        total_examples = 0
        for subset_list in subsets.values():
            for subset in subset_list:
                total_examples += len(subset[0])  # subset[0] contains the positives
        print(f"Total number of examples: {total_examples:,}")


if __name__ == "__main__":
    args = Arguments().parse_args()
    if args.command == COMMAND_EMBED:
        embed(args)
    elif args.command == COMMAND_CLUSTER:
        cluster(args)
