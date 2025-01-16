import os.path as osp
import pickle as pkl
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm

from quack.feedback import FeedbackGenerator


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

    # Initialize FeedbackGenerator
    feedback_generator = FeedbackGenerator(embeddings,entity_to_row, id2ent,
                                           id2rel, args.num_answers_threshold,
                                           args.plot)

    # Cluster answers to queries
    for split in splits:
        subsets = dict()
        all_queries = queries[split][('e', ('r',))]

        for query in tqdm(all_queries, desc=f"Generating {split}"):
            target_ids = list(answers[split][query])
            query_subsets = feedback_generator.generate(query, target_ids,
                                                        descriptions)
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


def main():
    args = Arguments().parse_args()
    if args.command == COMMAND_EMBED:
        embed(args)
    elif args.command == COMMAND_CLUSTER:
        cluster(args)


if __name__ == "__main__":
    main()
