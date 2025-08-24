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
from quack.graph import GraphDatabase


COMMAND_EMBED = "embed"
COMMAND_GENERATE = "generate"
COMMAND_DESCRIBE = "describe"
COMMAND_LOAD = "load"
COMMANDS = Literal[COMMAND_EMBED, COMMAND_LOAD, COMMAND_GENERATE, COMMAND_DESCRIBE]

TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"


class Arguments(Tap):
    command: COMMANDS

    data_path: str
    graphdb_endpoint: str = "http://localhost:7200/repositories/fb15k237"
    embedding_model: str = "NovaSearch/stella_en_400M_v5"

    tasks = "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u-DNF.up-DNF"
    splits = "train.valid.test"

    max_bindings: int = 10_000_000
    """Maximum number of bindings to consider for a query"""
    min_items_to_cluster: int = 10
    """Minimum number of bindings to consider for clustering"""
    max_items_to_cluster: int = 100
    """Maximum number of bindings to consider for clustering"""
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
    debug: bool = False

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
    embeddings_filename = get_embeddings_filename(args.embedding_model)
    emb_data = torch.load(osp.join(args.data_path, embeddings_filename), weights_only=False)
    embeddings = emb_data["embeddings"]
    descriptions = emb_data["descriptions"]
    entity_to_row = emb_data["entity_to_row"]

    with open(osp.join(args.data_path, "id2ent.pkl"), "rb") as f:
        id2ent = pkl.load(f)
    with open(osp.join(args.data_path, "id2rel.pkl"), "rb") as f:
        id2rel = pkl.load(f)

    preference_generator = PreferenceGenerator(embeddings, entity_to_row, id2ent,
                                               id2rel, args.max_num_sessions,
                                               args.plot)
    tasks_list = args.tasks.split(".")
    query_structures = [name_query_dict[query_type] for query_type in tasks_list]
    subsample_map = {
        "train": args.subsample_train,
        "valid": args.subsample_valid,
        "test": args.subsample_test
    }

    graph_database = GraphDatabase(args.graphdb_endpoint, tasks_list, args.max_bindings)

    splits = [TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT]
    requested_splits = args.splits.split(".")
    if not all([s in splits for s in requested_splits]):
        raise ValueError(f"splits must be a subset of {splits}, got {requested_splits}")

    for i, split in enumerate(splits):
        if split not in requested_splits:
            continue

        split_sessions = dict()
        num_queries = 0

        with open(osp.join(args.data_path, f"{split}-queries.pkl"), "rb") as f:
            split_queries = pkl.load(f)
        kind = f"-hard" if split != TRAIN_SPLIT else ""
        with open(osp.join(args.data_path, f"{split}{kind}-answers.pkl"), "rb") as f:
            split_answers = pkl.load(f)

        for structure in query_structures:
            flat_structure = flatten(structure)
            if structure not in split_queries:
                continue
            all_queries = split_queries[structure]
            structure_name = query_name_dict[structure]
            num_queries += len(all_queries)
            structure_query_sessions = dict()

            queries_to_sample = int(args.subsampling_ratio * len(all_queries)) if args.subsampling_ratio else -1

            task = query_name_dict[structure]
            query_splits = splits[:i + 1]
            bar_description = f"{structure_name} {split}"

            total = queries_to_sample if queries_to_sample > 0 else len(all_queries)
            with tqdm(total=total, mininterval=1, disable=args.plot, desc=bar_description, ncols=80) as pbar:
                for query in all_queries:
                    query_hard_answers = split_answers[query]

                    try:
                        full_bindings = graph_database.run_query(task, query_splits, flatten(query))
                    except OverflowError:
                        continue

                    if split == TRAIN_SPLIT:
                        easy_bindings = full_bindings
                    else:
                        # Easy bindings consider all edges except those in the current split
                        easy_bindings = graph_database.run_query(task, query_splits[:-1], flatten(query))

                    # Some queries can only be answered when the validation/test edges are added
                    if len(easy_bindings) == 0:
                        continue

                    num_variables = full_bindings.shape[1]
                    unique_full_answers = set(full_bindings.iloc[:, -1])
                    assert query_hard_answers.issubset(unique_full_answers)

                    if args.plot:
                        print(f"Query type: {structure_name}")
                        print(f"Query structure: {structure}")
                        flat_query = flatten(query)
                        print("Flat query:", " ".join([f"{t}_{k}" for k, t in enumerate(flat_structure)]))
                        for j, (kind, identifier) in enumerate(zip(flat_structure, flat_query)):
                            if kind == "e":
                                print(f"{j}: [{id2ent[identifier]}] {descriptions[entity_to_row[id2ent[identifier]]][:150]}")
                            elif kind == "r":
                                print(f"{j} Predicate: {id2rel[identifier]}")

                    # Find clusters for values of intermediate variables
                    session_data = [[] for _ in range(num_variables)]
                    session_data_stored = False
                    for j in range(num_variables):
                        full_answers = list(set(full_bindings.iloc[:, j]))
                        # Find clusters of answers. This results in different partitions of the answer set, of the form
                        # (preferred, non-preferred).
                        if args.min_items_to_cluster <= len(full_answers) <= args.max_items_to_cluster:
                            query_sessions = preference_generator.generate(full_answers, descriptions)
                        else:
                            continue

                        # Check if clusters of intermediate variables (explicit feedback) lead to clusters of target
                        # variable assignments (implicit feedback). If so, they make it into the dataset.
                        easy_var_target_bindings = easy_bindings.iloc[:, [j, -1]]
                        full_var_target_bindings = full_bindings.iloc[:, [j, -1]]

                        easy_var_bindings = set(easy_var_target_bindings.iloc[:, 0])
                        for session in query_sessions:
                            full_positives, full_negatives = session

                            # Are there enough entities if we drop test (hard) bindings?
                            easy_positives = set(full_positives) & easy_var_bindings
                            easy_negatives = set(full_negatives) & easy_var_bindings
                            if len(easy_positives) < 5 or len(easy_negatives) < 5:
                                continue

                            # Select rows corresponding to entities in positives and negatives
                            pos_implicit_answers_mask = full_var_target_bindings.iloc[:, 0].isin(full_positives)
                            neg_implicit_answers_mask = full_var_target_bindings.iloc[:, 0].isin(full_negatives)
                            # Select the answers (at position -1) induced by the positives and negatives
                            pos_implicit_answers = set(full_var_target_bindings[pos_implicit_answers_mask].iloc[:, -1])
                            neg_implicit_answers = set(full_var_target_bindings[neg_implicit_answers_mask].iloc[:, -1])

                            # Induced answers might overlap. Get answers only those reachable from the positive set
                            strict_pos_implicit_answers = pos_implicit_answers.difference(neg_implicit_answers)
                            # Compute intersection to limit to hard answers
                            strict_pos_implicit_answers = strict_pos_implicit_answers & query_hard_answers

                            # We finally add an instance to the dataset if the clustering of the intermediate
                            # variable leads to non-empty and non-overlapping sets of induced answers.
                            if len(strict_pos_implicit_answers) > 0:
                                strict_neg_implicit_answers = query_hard_answers - strict_pos_implicit_answers
                                if len(strict_neg_implicit_answers) > 0:
                                    session_data[j].append((
                                        list(easy_positives),
                                        list(easy_negatives),
                                        list(strict_pos_implicit_answers),
                                        list(strict_neg_implicit_answers)
                                    ))
                                    session_data_stored = True

                    if session_data_stored:
                        structure_query_sessions[query] = session_data
                        pbar.update()
                        if 0 < queries_to_sample == len(structure_query_sessions) and subsample_map.get(split, False):
                            break
                        if len(structure_query_sessions) == 10 and args.debug:
                            break

                pbar.close()

            split_sessions.update(structure_query_sessions)

        with open(osp.join(args.data_path, f"{split}-sessions-v2.pkl"), "wb") as f:
            pkl.dump(split_sessions, f)

    print("Done!")


def describe(args: Arguments):
    """Prints the number of queries and sessions for each query structure in
    each split.
    """
    def print_row(label, data):
        print(f"{label:>12}", end="")
        for d in data:
            if type(d) == str:
                print(f"{d:>8} & ", end="")
            else:
                print(f"{d:>8,} & ", end="")
        print()

    print(f"Loading data from {args.data_path}")

    splits = ["train", "valid", "test"]
    for split in splits:
        with open(osp.join(args.data_path, f"{split}-queries.pkl"), "rb") as f:
            structure_to_queries = pkl.load(f)
        with open(osp.join(args.data_path, f"{split}-sessions-v2.pkl"), "rb") as f:
            query_to_sessions = pkl.load(f)
        kind = "-hard" if split != "train" else ""
        with open(osp.join(args.data_path, f"{split}{kind}-answers.pkl"), "rb") as f:
            answers = pkl.load(f)

        print(f"Split: {split}")
        query_count = dict()
        pref_query_count = dict()
        session_count = dict()
        max_answers = dict()
        avg_answers = dict()
        structures = []
        for structure in query_name_dict:
            if structure not in structure_to_queries:
                continue
            structures.append(structure)
            queries = structure_to_queries[structure]

            num_queries = len(queries)  # total queries for this structure
            num_pref_queries = 0        # queries in query_to_sessions
            num_sessions = 0
            max_num_answers = 0
            total_num_answers = 0
            num_counted = 0
            for query in queries:
                if query in query_to_sessions:
                    num_pref_queries += 1
                    num_sessions += sum(map(len, query_to_sessions[query]))
                # Compute max and avg number of answers for this structure
                if query in answers:
                    num_ans = len(answers[query])
                    total_num_answers += num_ans
                    num_counted += 1
                    if num_ans > max_num_answers:
                        max_num_answers = num_ans
            query_count[structure] = num_queries
            pref_query_count[structure] = num_pref_queries
            session_count[structure] = num_sessions
            max_answers[structure] = max_num_answers
            avg_answers[structure] = (total_num_answers / num_counted) if num_counted > 0 else 0

        query_count["Total"] = sum(query_count.values())
        pref_query_count["Total"] = sum(pref_query_count.values())
        session_count["Total"] = sum(session_count.values())
        max_answers["Total"] = max(max_answers.values()) if max_answers else 0
        avg_answers["Total"] = (sum(avg_answers[s] * (len(structure_to_queries[s]) if s in structure_to_queries else 0) for s in structures[:-1]) / sum(len(structure_to_queries[s]) for s in structures[:-1]) ) if structures[:-1] else 0

        # structures = list(structure_to_queries.keys())
        structure_names = [query_name_dict[s] for s in structures] + ["Total"]
        structures += ["Total"]

        # cell_divider = "-" * 10
        # print_row(cell_divider, [cell_divider for _ in structures])
        print_row("Structure", structure_names)
        print_row("Queries", [query_count[s] for s in structures])
        print_row("PrefQueries", [pref_query_count[s] for s in structures])
        print_row("Sessions", [session_count[s] for s in structures])
        print_row("MaxAns", [max_answers[s] for s in structures])
        print_row("AvgAns", [f"{avg_answers[s]:.2f}" for s in structures])
        print()


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
