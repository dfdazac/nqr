import os.path as osp
import pickle as pkl
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
import random
import requests
from sentence_transformers import SentenceTransformer
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm

from quack.feedback import PreferenceGenerator
from quack.qto.query import name_query_dict, query_name_dict
from quack.qto.util import flatten


COMMAND_EMBED = "embed"
COMMAND_GENERATE = "generate"
COMMAND_DESCRIBE = "describe"
COMMAND_LOAD = "load"
COMMANDS = Literal[COMMAND_EMBED, COMMAND_LOAD, COMMAND_GENERATE, COMMAND_DESCRIBE]


class Arguments(Tap):
    command: COMMANDS

    data_path: str
    graphdb_endpoint: str = "http://localhost:7200/repositories/fb15k237"
    embedding_model: str = "NovaSearch/stella_en_400M_v5"

    tasks = "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u-DNF.up-DNF"

    min_answer_threshold: int = 10
    """Minimum number of answers to consider a query"""
    max_answer_threshold: int = 100
    """Maximum number of answers to consider a query"""
    max_num_sessions: int = 5
    """Maximum number of sessions (partitions of the answer set) to generate
    for each query"""

    subsample_train: bool = False
    subsample_valid: bool = False
    subsample_test: bool = False
    subsampling_ratio: float = None
    """The fraction of generated queries to keep. If not set, all queries are kept."""
    seed: int = 0
    """Random seed used when subsampling queries"""

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


def generate(args: Arguments):
    # Load entity embeddings
    embeddings_filename = get_embeddings_filename(args.embedding_model)
    emb_data = torch.load(osp.join(args.data_path, embeddings_filename), weights_only=False)
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

    # Initialize PreferenceGenerator
    preference_generator = PreferenceGenerator(embeddings, entity_to_row, id2ent,
                                               id2rel, args.max_num_sessions,
                                               args.plot)
    query_structures = [name_query_dict[query_type] for query_type in args.tasks.split(".")]
    subsample_map = {
        "train": args.subsample_train,
        "valid": args.subsample_valid,
        "test": args.subsample_test
    }
    # Cluster answers to queries
    for split in splits:
        split_sessions = dict()
        num_queries = 0

        for structure in query_structures:
            flat_structure = flatten(structure)
            if structure not in queries[split]:
                continue
            all_queries = queries[split][structure]
            structure_name = query_name_dict[structure]
            num_queries += len(all_queries)
            structure_query_sessions = dict()

            print(f"Generating {split} split for {structure_name} queries...")
            for query in tqdm(all_queries, mininterval=1, disable=args.plot):
                answer_ids = list(answers[split][query])
                num_answers = len(answer_ids)

                if not args.min_answer_threshold <= num_answers <= args.max_answer_threshold:
                    continue

                if args.plot:
                    print(f"Query type: {structure_name}")
                    print(f"Query structure: {structure}")
                    flat_query = flatten(query)
                    print("Flat query:", " ".join([f"{t}_{k}" for k, t in enumerate(flat_structure)]))
                    for i, (kind, identifier) in enumerate(zip(flat_structure, flat_query)):
                        if kind == "e":
                            print(f"{i}: [{id2ent[identifier]}] {descriptions[entity_to_row[id2ent[identifier]]][:150]}")
                        elif kind == "r":
                            print(f"{i} Predicate: {id2rel[identifier]}")

                query_sessions = preference_generator.generate(answer_ids, descriptions)
                structure_query_sessions[query] = query_sessions

                if args.plot and len(query_sessions) > 0:
                    for session in query_sessions:
                        for kind, ids in zip(("Positives", "Negatives"), session):
                            print(kind)
                            for id in ids:
                                print(f"\t[{id2ent[id]}] {descriptions[entity_to_row[id2ent[id]]][:150]}")

                    a = input("Press enter to continue")

            if args.subsampling_ratio is not None and subsample_map.get(split, False):
                    random.seed(args.seed)
                    num_samples = int(args.subsampling_ratio * len(structure_query_sessions))
                    subsampled_queries = random.sample(list(structure_query_sessions.keys()), num_samples)
                    structure_query_sessions = {q: structure_query_sessions[q] for q in subsampled_queries}

            split_sessions.update(structure_query_sessions)

        with open(osp.join(args.data_path, f"{split}-sessions.pkl"), "wb") as f:
            pkl.dump(split_sessions, f)

    print("Done!")


def describe(args: Arguments):
    """Prints the number of queries and sessions for each query structure in
    each split.
    """
    def print_row(label, data):
        print(f"{label:>10}", end="")
        for d in data:
            if type(d) == str:
                print(f"{d:>7} & ", end="")
            else:
                print(f"{d:>7,} & ", end="")
        print()

    print(f"Loading data from {args.data_path}")

    splits = ["train", "valid", "test"]
    for split in splits:
        with open(osp.join(args.data_path, f"{split}-queries.pkl"), "rb") as f:
            structure_to_queries = pkl.load(f)
        with open(osp.join(args.data_path, f"{split}-sessions.pkl"), "rb") as f:
            query_to_sessions = pkl.load(f)

        print(f"Split: {split}")
        query_count = dict()
        session_count = dict()
        structures = []
        for structure in query_name_dict:
            if structure not in structure_to_queries:
                continue
            structures.append(structure)
            queries = structure_to_queries[structure]

            num_queries = 0
            num_sessions = 0
            for query in queries:
                if query in query_to_sessions:
                    num_queries += 1
                    num_sessions += len(query_to_sessions[query])
            query_count[structure] = num_queries
            session_count[structure] = num_sessions

        query_count["Total"] = sum(query_count.values())
        session_count["Total"] = sum(session_count.values())

        # structures = list(structure_to_queries.keys())
        structure_names = [query_name_dict[s] for s in structures] + ["Total"]
        structures += ["Total"]

        cell_divider = "-" * 7
        print_row(cell_divider, [cell_divider for _ in structures])
        print_row("Structure", structure_names)
        print_row("Queries", [query_count[s] for s in structures])
        print_row("Sessions", [session_count[s] for s in structures])


def load_graphdb(args: Arguments):
    # Write .nt files from txt
    all_splits = ["train", "valid", "test"]
    graph_uri_map = {
        "train": "http://example.org/train",
        "valid": "http://example.org/valid",
        "test": "http://example.org/test"
    }

    for split in all_splits:
        txt_path = osp.join(args.data_path, f"{split}.txt")
        nt_path = osp.join(args.data_path, f"{split}.nt")
        with open(txt_path) as txt_file, open(nt_path, "w") as nt_file:
            for line in txt_file:
                s, p, o = line.strip().split()
                nt_file.write(f"<http://example.org/Q{s}> "
                              f"<http://example.org/P{p}> "
                              f"<http://example.org/Q{o}> .\n")
        print(f"nt file saved at {nt_path}")

        # Post files to GraphDB
        graphdb_post_url = f"{args.graphdb_endpoint}/statements"
        graph_uri = graph_uri_map[split]
        params = {"context": f"<{graph_uri}>"}
        headers = {"Content-Type": "application/n-triples"}

        with open(nt_path, "rb") as data:
            response = requests.post(graphdb_post_url, params=params, headers=headers, data=data)

        if response.status_code == 204:
            print(f"Successfully uploaded {split}.nt to graph <{graph_uri}>")
        else:
            print(f"Failed to upload {split}.nt: {response.status_code} {response.text}")


def main():
    args = Arguments().parse_args()
    if args.command == COMMAND_EMBED:
        embed(args)
    elif args.command == COMMAND_LOAD:
        load_graphdb(args)
    elif args.command == COMMAND_GENERATE:
        generate(args)
        describe(args)
    elif args.command == COMMAND_DESCRIBE:
        describe(args)
    else:
        raise ValueError(f"Invalid command: {args.command}")


if __name__ == "__main__":
    main()
