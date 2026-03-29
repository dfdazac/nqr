import argparse
import collections
import logging
import os
import os.path as osp
import pickle
from collections import defaultdict
import random
import time
from pprint import pprint

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ndcg_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats as stats
import wandb
import yaml

from nqr.qto.dataset import TestDataset
from nqr.qto.model import KGReasoning
from nqr.qto.util import flatten, flatten_query, set_global_seed

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   ('e', ('r', 'r', 'r', 'r')): '4p',
                   ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '4i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
                   }
name_answer_dict = {'1p': ['e', ['r', ], 'e'],
                    '2p': ['e', ['r', 'e', 'r'], 'e'],
                    '3p': ['e', ['r', 'e', 'r', 'e', 'r'], 'e'],
                    '2i': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], 'e'],
                    '3i': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['e', ['r', ], 'e'], 'e'],
                    'ip': [[['e', ['r', ], 'e'], ['e', ['r', ], 'e'], 'e'], ['r', ], 'e'],
                    'pi': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r', ], 'e'], 'e'],
                    '2in': [['e', ['r', ], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    '3in': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'inp': [[['e', ['r', ], 'e'], ['e', ['r', 'n'], 'e'], 'e'], ['r', ], 'e'],
                    'pin': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'pni': [['e', ['r', 'e', 'r', 'n'], 'e'], ['e', ['r', ], 'e'], 'e'],
                    '2u-DNF': [['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['u', ], 'e'],
                    'up-DNF': [[['e', ['r', ], 'e'], ['e', ['r', ], 'e'], ['u', ], 'e'], ['r', ], 'e'],
                    }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())
espace = 9
rspace = 11


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(args, config, parser_defaults):
    """Merge YAML config with command line arguments. Command line args take precedence."""
    if config is None:
        return args

    # Convert args namespace to dict for easier manipulation
    args_dict = vars(args)

    for key, value in config.items():
        # Only update if the argument wasn't explicitly set (i.e., it's still the default value)
        if key in args_dict and args_dict[key] == parser_defaults.get(key):
            # Convert string values to appropriate types based on parser defaults
            if key in parser_defaults:
                default_type = type(parser_defaults[key])
                if default_type == bool:
                    # Handle boolean values specially
                    if isinstance(value, str):
                        args_dict[key] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        args_dict[key] = bool(value)
                elif default_type in (int, float, str):
                    args_dict[key] = default_type(value)
                else:
                    args_dict[key] = value
            else:
                args_dict[key] = value

    return argparse.Namespace(**args_dict)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--test_annotated', action='store_true', help="test annotated sessions")
    parser.add_argument('--do_cp', action='store_true', help="do cardinality prediction")
    parser.add_argument('--path', action='store_true', help="do interpretation study")
    parser.add_argument('--wandb', action='store_true', help="log to wandb")
    parser.add_argument('--notes', default=None, type=str, help="notes for wandb")
    parser.add_argument('--test_run', action='store_true', help="run on a small dataset")
    parser.add_argument('--save_scores', action='store_true', help="save query scores for all entities")
    parser.add_argument('--verbose', action='store_true', help="enable verbose debug output during evaluation")

    parser.add_argument('--num_epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('--valid_frequency', default=5000, type=int, help='validation frequency in training steps')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--kbc_path', type=str, default=None, help="kbc model path")
    parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint path")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=0, type=int, help="used to speed up torch.dataloader")

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--fraction', type=int, default=1, help='fraction the entity to save gpu memory usage')
    parser.add_argument('--thrshd', type=float, default=0.001, help='thrshd for neural adjacency matrix')
    parser.add_argument('--neg_scale', type=int, default=1, help='scaling neural adjacency matrix for negation')
    parser.add_argument('--force_training_edges', action='store_true',
                        help='set scores for edges in training graph to 1')

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u-DNF.up-DNF', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=12345, type=int, help="random seed")
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument("--preference", default="none", choices=["positive", "negative", "mixed", "none"],
                        help="preference type")
    parser.add_argument('--reranker',
                        default='cosine',
                        type=str,
                        choices=['default', 'cosine', 'ranknet', 'lightgbm_lambdamart'],
                        help='reranker method')

    # Cosine hyperparameters
    parser.add_argument('--alpha', default=0.5, type=float, help="Convex combination parameter for the cosine similarity reranker")
    parser.add_argument('--beta', default=0.0, type=float, help="Positive-negative combination parameter for the cosine similarity reranker")

    parser.add_argument("--hidden_dim", default=256, type=int, help="Hidden dimension for the neural reranking network")
    parser.add_argument("--activation", default="relu", choices=["relu", "elu"],
                        help="Activation function for the reranking network")
    parser.add_argument('--config', type=str, default=None, help='path to YAML config file')
    parser.add_argument('--profile_time', action='store_true', help="profile execution time of rerankers")

    # Parse arguments first to get config path
    parsed_args = parser.parse_args(args)

    # If no config file specified, return parsed args as-is
    if parsed_args.config is None:
        return parsed_args

    # Load YAML config
    try:
        config = load_config_from_yaml(parsed_args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{parsed_args.config}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}")
        exit(1)

    # Get default values by creating a parser with no arguments
    default_args = parser.parse_args([])
    parser_defaults = vars(default_args)

    # Merge config with parsed arguments
    merged_args = merge_config_with_args(parsed_args, config, parser_defaults)

    return merged_args


def log_metrics(mode, metrics, file_pointer):
    for metric in metrics:
        logging.info('%s %s: %f' % (mode, metric, metrics[metric]))
        print('%s %s: %f' % (mode, metric, metrics[metric]))
        file_pointer.write('%s %s: %f\n' % (mode, metric, metrics[metric]))


def read_triples(filenames, nrelation, datapath):
    adj_list = [[] for i in range(nrelation)]
    edges_all = set()
    edges_vt = set()
    for filename in filenames:
        with open(filename) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                adj_list[int(r)].append((int(h), int(t)))
    for filename in ['valid.txt', 'test.txt']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                edges_all.add((int(h), int(r), int(t)))
                edges_vt.add((int(h), int(r), int(t)))
    with open(os.path.join(datapath, "train.txt")) as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            edges_all.add((int(h), int(r), int(t)))

    return adj_list, edges_all, edges_vt


def compute_metrics(embedding, hard_answers, easy_answers, queries_unflatten):
    device = embedding.device
    order = torch.argsort(embedding, dim=-1, descending=True)
    ranking = torch.argsort(order)
    # eval
    hard_answers = hard_answers[queries_unflatten[0]]
    easy_answers = easy_answers[queries_unflatten[0]]
    num_hard = len(hard_answers)
    num_easy = len(easy_answers)
    cur_ranking = ranking[list(easy_answers) + list(hard_answers)]

    cur_ranking, indices = torch.sort(cur_ranking)
    masks_hard = indices >= num_easy
    masks_easy = indices < num_easy
    answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
    cur_ranking_hard = cur_ranking[masks_hard]  # take indices that belong to the hard answers
    cur_ranking_easy = cur_ranking[masks_easy]  # take indices that belong to the easy answers

    mrr_hard = torch.mean(1. / cur_ranking_hard).item()
    h1_hard = torch.mean((cur_ranking_hard <= 1).to(torch.float)).item()
    h3_hard = torch.mean((cur_ranking_hard <= 3).to(torch.float)).item()
    h10_hard = torch.mean((cur_ranking_hard <= 10).to(torch.float)).item()
    mrr_easy = torch.mean(1. / cur_ranking_easy).item()
    h1_easy = torch.mean((cur_ranking_easy <= 1).to(torch.float)).item()
    h3_easy = torch.mean((cur_ranking_easy <= 3).to(torch.float)).item()
    h10_easy = torch.mean((cur_ranking_easy <= 10).to(torch.float)).item()
    if num_easy == 0:
        mrr_easy, h1_easy, h3_easy, h10_easy = 1, 1, 1, 1

    return {
        'mrr_hard': mrr_hard,
        'hits@1_hard': h1_hard,
        'hits@3_hard': h3_hard,
        'hits@10_hard': h10_hard,
        'num_hard_answer': num_hard,
        'mrr_easy': mrr_easy,
        'hits@1_easy': h1_easy,
        'hits@3_easy': h3_easy,
        'hits@10_easy': h10_easy,
        'num_easy_answer': num_easy,
    }


def find_rerank_regressions(base_scores, reranked_scores, easy_answers, hard_answers, top_k=5):
    correct_answers = set(easy_answers) | set(hard_answers)
    if not correct_answers:
        return []

    device = base_scores.device
    nentity = base_scores.shape[0]
    correct_indices = torch.tensor(sorted(correct_answers), device=device, dtype=torch.long)
    correct_mask = torch.zeros(nentity, dtype=torch.bool, device=device)
    correct_mask[correct_indices] = True

    def get_positions_and_incorrect_above(scores):
        order = torch.argsort(scores, dim=-1, descending=True)
        positions = torch.empty_like(order)
        positions[order] = torch.arange(order.numel(), device=device)

        incorrect_prefix = torch.cumsum((~correct_mask[order]).to(torch.long), dim=0)
        incorrect_above_sorted = incorrect_prefix - (~correct_mask[order]).to(torch.long)
        incorrect_above = torch.empty_like(incorrect_above_sorted)
        incorrect_above[order] = incorrect_above_sorted
        return positions, incorrect_above

    base_positions, base_incorrect_above = get_positions_and_incorrect_above(base_scores)
    reranked_positions, reranked_incorrect_above = get_positions_and_incorrect_above(reranked_scores)

    incorrect_above_delta = reranked_incorrect_above[correct_indices] - base_incorrect_above[correct_indices]
    position_drop = reranked_positions[correct_indices] - base_positions[correct_indices]
    harmed_mask = incorrect_above_delta > 0

    if not harmed_mask.any():
        return []

    harmed_entities = correct_indices[harmed_mask]
    harmed_scores = reranked_scores[harmed_entities]
    harmed_order = torch.argsort(harmed_scores, descending=True)[:top_k]

    regressions = []
    for idx in harmed_order.tolist():
        entity_id = harmed_entities[idx].item()
        regressions.append({
            "entity_id": entity_id,
            "base_rank": base_positions[entity_id].item() + 1,
            "reranked_rank": reranked_positions[entity_id].item() + 1,
            "position_drop": position_drop[harmed_mask][idx].item(),
            "base_incorrect_above": base_incorrect_above[entity_id].item(),
            "reranked_incorrect_above": reranked_incorrect_above[entity_id].item(),
            "incorrect_above_delta": incorrect_above_delta[harmed_mask][idx].item(),
            "reranked_score": reranked_scores[entity_id].item(),
        })

    return regressions


def train(model, args, tasks, device, output_path):
    '''
    Train model on dataloader
    '''
    queries, answers, _, sessions = load_data(args, tasks, "train")
    queries = flatten_query(queries)
    train_dataset = TestDataset(queries, sessions, args.nentity, args.nrelation)

    valid_queries, valid_hard_answers, valid_easy_answers, valid_sessions = load_data(args, tasks, "valid")
    valid_queries = flatten_query(valid_queries)
    valid_dataset = TestDataset(valid_queries, valid_sessions, args.nentity, args.nrelation)

    test_queries, test_hard_answers, test_easy_answers, test_sessions = load_data(args, tasks, "test")
    test_queries = flatten_query(test_queries)
    test_dataset = TestDataset(test_queries, test_sessions, args.nentity, args.nrelation)

    if args.test_run:
        train_dataset = torch.utils.data.Subset(valid_dataset, range(10))

        valid_dataset = train_dataset

        test_dataset = valid_dataset
        test_hard_answers, test_easy_answers = valid_hard_answers, valid_easy_answers

    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Disable gradients for KG embeddings
    for p in model.kbc_model.parameters():
        p.requires_grad = False

    batch_preference_losses = []
    batch_answer_losses = []
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(range(len(dataloader)), desc=f"Epoch {epoch}/{args.num_epochs}", mininterval=1)
        iterator = iter(dataloader)
        finished = False
        num_batches = 0
        while not finished:
            batch_scores = []
            batch_positives = []
            batch_positive_ids = []
            batch_negatives = []
            batch_negative_ids = []
            batch_preferences = []
            batch_labels = []
            batch_pref_ids = []
            num_preferences = random.randint(1, 10)
            for i in range(args.batch_size):
                try:
                    flat_queries, queries, query_structures, sessions = next(iterator)
                except StopIteration:
                    finished = True
                    break

                flat_queries = torch.tensor(flat_queries, device=device)
                scores, *_ = model.embed_query(flat_queries, query_structures[0], 0)
                batch_scores.append(scores)

                # Pick a random session
                positives, negatives = random.choice(sessions[0])
                batch_positives.extend(positives)
                batch_positive_ids.extend([i] * len(positives))
                batch_negatives.extend(negatives)
                batch_negative_ids.extend([i] * len(negatives))

                # Pick a random mix of positive and negative feedback and cap it at num_preferences
                combined = [(item, 1) for item in positives] + [(item, 0) for item in negatives]
                sampled = random.sample(combined, min(num_preferences, len(combined)))
                preferences, labels = zip(*sampled)

                batch_preferences.extend(preferences)
                batch_labels.extend(labels)
                batch_pref_ids.extend([i] * len(preferences))

            if len(batch_scores) == 0:
                break

            train_bar.update(len(batch_scores))

            batch_scores = torch.cat(batch_scores)
            batch_preferences = torch.tensor(batch_preferences, device=device)
            batch_labels = torch.tensor(batch_labels, device=device)
            batch_pref_ids = torch.tensor(batch_pref_ids, device=device)
            batch_positives = torch.tensor(batch_positives, device=device)
            batch_negatives = torch.tensor(batch_negatives, device=device)
            batch_positive_ids = torch.tensor(batch_positive_ids, device=device)
            batch_negative_ids = torch.tensor(batch_negative_ids, device=device)

            preference_loss, answer_loss, deltas = model.reranking_loss(
                (batch_preferences, batch_labels, batch_pref_ids),
                batch_scores,
                (batch_positives, batch_positive_ids),
                (batch_negatives, batch_negative_ids)
            )

            loss = preference_loss + answer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_preference_losses.append(preference_loss.item())
            batch_answer_losses.append(answer_loss.item())
            num_batches += 1

            if num_batches % 100 == 0:
                delta_dict = {}
                for delta_type, delta in zip(("pos", "neg", "random"), deltas):
                    delta_dict[f"{delta_type}_delta_min"] = delta.min().item()
                    delta_dict[f"{delta_type}_delta_max"] = delta.max().item()
                    delta_dict[f"{delta_type}_delta_mean"] = delta.mean().item()

                wandb.log({"preference_loss": torch.tensor(batch_preference_losses).mean().item(),
                           "answer_loss": torch.tensor(batch_answer_losses).mean().item(),
                           "loss": loss.item(),
                           **delta_dict})
                batch_preference_losses = []
                batch_answer_losses = []

        train_bar.close()

    all_metrics = evaluate(model, valid_hard_answers, valid_easy_answers, args, valid_dataloader,
                           query_name_dict, device, output_path, "valid", preference="mixed")
    wandb.log({f"valid_{k}": v for k, v in all_metrics.items() if "cumulative" in k})

    all_metrics = evaluate(model, test_hard_answers, test_easy_answers, args, test_dataloader, query_name_dict, device,
                           output_path, "test", preference="mixed")
    wandb.log({f"test_{k}": v for k, v in all_metrics.items() if "cumulative" in k})
    torch.save(model.state_dict(), osp.join(output_path, f'{wandb.run.id}-model.pt'))


@torch.inference_mode()
def train_lightgbm(model, args, tasks, device, output_path):
    '''
    Train LightGBM ranker on training data
    '''
    tasks_str = "_".join(sorted(tasks))
    cache_filename = f"lgb_training_data_{tasks_str}.pkl"
    cache_path = osp.join(args.data_path, cache_filename)

    if osp.exists(cache_path):
        print(f"Loading cached training data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} training examples from cache")
    else:
        queries, answers, _, sessions = load_data(args, tasks, "train")
        queries = flatten_query(queries)
        train_dataset = TestDataset(queries, sessions, args.nentity, args.nrelation)

        if args.test_run:
            train_dataset = torch.utils.data.Subset(train_dataset, range(10))
        else:
            # Limit training dataset to 10,000 samples
            train_size = min(len(train_dataset), 10000)
            if train_size < len(train_dataset):
                train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
                print(f"Limited training dataset to {train_size} samples")

        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn,
            shuffle=False
        )

        model.to(device)
        training_data = []
        
        print("Collecting training data for LightGBM...")
        for flat_queries, queries, query_structures, sessions in tqdm(dataloader, desc="Extracting features"):
            sessions = sessions[0]
            if len(sessions) == 0:
                continue
                
            flat_queries = torch.tensor(flat_queries, device=device)
            scores, *_ = model.embed_query(flat_queries, query_structures[0], 0)
            scores = scores.squeeze()
            
            # For each session, create training example
            for positives, negatives in sessions:
                num_random = min(100, args.nentity - len(positives) - len(negatives))
                all_entities = set(range(args.nentity))
                available = all_entities - set(positives) - set(negatives)
                random_entities = random.sample(list(available), num_random)
                
                entities = positives + negatives + random_entities
                relevance = [2] * len(positives) + [1] * len(negatives) + [0] * len(random_entities)
                
                # Sample some preferences for feature extraction
                num_prefs = min(10, len(positives) + len(negatives))
                pref_entities = random.sample(positives + negatives, num_prefs)
                pref_labels = [1 if e in positives else 0 for e in pref_entities]
                
                preferences = torch.tensor(pref_entities, device=device)
                labels = torch.tensor(pref_labels, device=device)
                
                # Extract features for all entities
                features = model.extract_features(scores, preferences, labels)
                features_np = features[entities].cpu().numpy()
                relevance_np = np.array(relevance)
                
                training_data.append((features_np, relevance_np, len(entities)))
        
        print(f"Collected {len(training_data)} training examples")
        
        print(f"Saving training data to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump(training_data, f)
        print("Training data cached successfully")
    
    print("Training LightGBM ranker...")
    lgb_model = model.train_lightgbm(training_data)
    
    model_path = osp.join(output_path, f'{wandb.run.id}-lightgbm.txt')
    lgb_model.save_model(model_path)
    print(f"Saved LightGBM model to {model_path}")

    test_queries, test_hard_answers, test_easy_answers, test_sessions = load_data(args, tasks, "test")
    test_queries = flatten_query(test_queries)
    test_dataset = TestDataset(test_queries, test_sessions, args.nentity, args.nrelation)
    if args.test_run:
        test_dataset = torch.utils.data.Subset(test_dataset, range(10))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )
    
    all_metrics = evaluate(model, test_hard_answers, test_easy_answers, args,
                           test_dataloader, query_name_dict, device, output_path, "test",
                           preference="mixed", lgb_model=lgb_model)
    wandb.log({f"test_{k}": v for k, v in all_metrics.items() if "cumulative" in k})
    
    return lgb_model


def compute_ndcg(relevance_scores, predicted_scores, k_values=(10, 100)):
    results = {}
    for k in k_values:
        results[f"ndcg@{k}"] = ndcg_score(relevance_scores, predicted_scores, k=k, ignore_ties=False)
    return results


@torch.inference_mode()
def evaluate(model: KGReasoning, hard_answers, easy_answers, args, dataloader, query_name_dict, device, output_path,
             mode, preference, lgb_model=None):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)
    results = defaultdict(list)
    reranked_delta = defaultdict(lambda: defaultdict(list))
    session_count = 0

    use_mean_cosine = args.reranker == "cosine_mean"

    evaluate_preferences = args.preference != "none"

    total_metrics_over_10_steps = defaultdict(list)
    query_to_ranking = defaultdict(dict)
    alpha_plot_saved = False

    with open(osp.join(args.data_path, "id2ent.pkl"), "rb") as f:
        id2ent = pickle.load(f)
    with open(osp.join(args.data_path, "id2rel.pkl"), "rb") as f:
        id2rel = pickle.load(f)
    with open(osp.join(args.data_path, "entity2text.txt")) as f:
        ent2text = dict()
        for line in f:
            key, value = line.strip().split("\t")
            ent2text[key] = value

    execution_times = []

    for flat_queries, queries, query_structures, sessions in tqdm(dataloader,
                                                                  desc=f"Evaluating on {mode} - preference: {preference}",
                                                                  mininterval=1,
                                                                  disable=args.verbose):
        sessions = sessions[0]

        if evaluate_preferences and len(sessions) == 0:
            raise ValueError("No sessions found for query")
        flat_structure = flatten(query_structures[0])
        entity_ids = [value for value, kind in zip(flat_queries[0], flat_structure) if kind == "e"]
        relation_ids = [value for value, kind in zip(flat_queries[0], flat_structure) if kind == "r"]
        anchor = ent2text[id2ent[entity_ids[0]]]
        relations = [id2rel[i] for i in relation_ids]
        flat_queries = torch.as_tensor(flat_queries, device=device, dtype=torch.long)

        query_start_time = time.time()
        scores, _, exec_query = model.embed_query(flat_queries, query_structures[0], 0)
        query_time = time.time() - query_start_time

        scores = scores.squeeze()
        initial_metrics = compute_metrics(scores, hard_answers, easy_answers, queries)

        if args.save_scores:
            query_to_ranking[query_name_dict[query_structures[0]]][queries[0]] = scores.tolist()

        query_cumulative_metrics = defaultdict(float)
        base_metrics = dict(initial_metrics)

        if evaluate_preferences:
            query_easy_answers = list(easy_answers[queries[0]])
            query_hard_answers = list(hard_answers[queries[0]])

            if args.verbose:
                last_relation = relations[-1]
                if last_relation != '+/award/award_nominee/award_nominations./award/award_nomination/award':
                    continue
                print("======== ANCHOR ==========")
                print(anchor)
                print("=============================\n")

                print("======== RELATIONS ==========")
                print(relations)
                print("=============================\n")

                print("======== TOP-10 answers ==========")
                print([ent2text[id2ent[i]] for i in scores.topk(10).indices.tolist()])
                print("=============================\n")

                cont = input("Enter y to continue, other to skip")

                if not cont == 'y':
                    continue

            for session in sessions:
                session_count += 1
                # session_scores = scores.clone()

                positives, negatives = session

                if args.verbose:
                    print("======== POSITIVES ==========")
                    print([ent2text[id2ent[i]] for i in positives])
                    print("=============================\n")

                    print("======== NEGATIVES ==========")
                    print([ent2text[id2ent[i]] for i in negatives])
                    print("=============================\n")

                relevance_scores = torch.zeros_like(scores, dtype=torch.long)
                relevance_scores[negatives] = 1
                relevance_scores[positives] = 2

                # Remove easy answers from NDCG evaluation (these are known in the graph)
                keep_mask = torch.ones_like(scores, dtype=torch.bool)
                keep_mask[query_easy_answers] = False

                assert torch.all(relevance_scores[query_hard_answers] >= 1)
                assert torch.all(keep_mask[query_hard_answers])

                # Slice to filtered candidates: hard answers + all remaining unknowns
                rel_eval = relevance_scores[keep_mask].unsqueeze(0).cpu()
                scores_eval = scores[keep_mask].unsqueeze(0).cpu()

                if args.verbose:
                    print("======== BASE METRICS ==========")
                base_metrics.update(compute_ndcg(rel_eval, scores_eval))
                if args.verbose:
                    print(base_metrics)
                    print("=============================\n")

                if preference == "positive":
                    session_feedback = positives[:10]
                    session_labels = [1] * len(session_feedback)
                elif preference == "negative":
                    session_feedback = negatives[:10]
                    session_labels = [0] * len(session_feedback)
                elif preference == "mixed":
                    pos_feedback = positives[:5]
                    neg_feedback = negatives[:5]
                    pos_labels = [1] * len(pos_feedback)
                    neg_labels = [0] * len(neg_feedback)

                    session_feedback = [i for pair in zip(positives[:5], negatives[:5]) for i in pair]
                    session_labels = [i for pair in zip(pos_labels, neg_labels) for i in pair]

                cumulative_metrics = defaultdict(float)
                metrics_over_10_steps = defaultdict(list)

                # Determine range based on profiling mode
                if args.profile_time:
                    # Only execute the last iteration for profiling
                    range_start = len(session_feedback) - 1
                    range_end = len(session_feedback)
                else:
                    range_start = 0
                    range_end = len(session_feedback)

                for t in range(range_start, range_end):
                    # Rerank embedding scores based on preferences
                    preferences = torch.tensor(session_feedback[:t + 1], device=device)
                    labels = torch.tensor(session_labels[:t + 1], device=device)

                    # Start timing measurement for profiling
                    if args.profile_time:
                        start_time = time.time()

                    if args.reranker == "default":
                        session_scores = scores
                    elif args.reranker == "cosine":
                        session_scores = model.rerank_cosine(scores, preferences, labels, args.alpha, args.beta)
                    elif args.reranker == "lightgbm_lambdamart":
                        session_scores = model.rerank_lightgbm(scores, preferences, labels, lgb_model)
                    elif args.reranker == "ranknet":
                        session_scores = model.rerank_nqr(scores, preferences, labels)

                    # End timing measurement for profiling
                    if args.profile_time:
                        execution_times.append(time.time() - start_time + query_time)

                    if t == 2 and args.verbose:
                        print("======== NEW TOP 10 ==========")
                        print([ent2text[id2ent[i]] for i in session_scores.topk(10).indices.tolist()])
                        print("=============================\n")

                        regressions = find_rerank_regressions(
                            scores,
                            session_scores,
                            query_easy_answers,
                            query_hard_answers,
                            top_k=5,
                        )
                        print("======== CORRECT ANSWERS HARMED BY RERANKING ==========")
                        if regressions:
                            for regression in regressions:
                                entity_name = ent2text[id2ent[regression["entity_id"]]]
                                print(
                                    f'{entity_name} | reranked score={regression["reranked_score"]:.4f} | '
                                    f'rank {regression["base_rank"]}->{regression["reranked_rank"]} '
                                    f'({regression["position_drop"]:+d}) | incorrect above '
                                    f'{regression["base_incorrect_above"]}->{regression["reranked_incorrect_above"]} '
                                    f'({regression["incorrect_above_delta"]:+d})'
                                )
                        else:
                            print("None")
                        print("=============================\n")

                    # Compute pairwise accuracy after reranking
                    pos_scores = session_scores[positives].unsqueeze(1)
                    neg_scores = session_scores[negatives].unsqueeze(0)
                    pairwise_accuracy = (pos_scores > neg_scores).float().mean().item()
                    cumulative_metrics["pairwise_accuracy"] += pairwise_accuracy

                    instant_metrics = compute_metrics(session_scores, hard_answers, easy_answers, queries)
                    scores_eval = session_scores[keep_mask].unsqueeze(0).cpu()
                    instant_metrics.update(compute_ndcg(rel_eval, scores_eval))

                    if t == 2 and args.verbose:
                        print("======== METRICS ==========")
                        print(instant_metrics)
                        print("=============================\n")

                    for metric in instant_metrics:
                        if metric.startswith('num'):
                            continue
                        absolute_delta = instant_metrics[metric] - base_metrics[metric]
                        relative_delta = absolute_delta / (1.0 if base_metrics[metric] == 0 else base_metrics[metric])
                        cumulative_metrics[f"{metric}"] += instant_metrics[metric]
                        cumulative_metrics[f"{metric}_delta"] += relative_delta
                        if t < 10 <= len(session_feedback):
                            if t == 0:
                                metrics_over_10_steps[metric].append(base_metrics[metric])
                            metrics_over_10_steps[metric].append(instant_metrics[metric])
                            metrics_over_10_steps[f"{metric}_delta"].append(relative_delta)

                        if t == len(session_feedback) - 1:
                            reranked_delta[query_structures[0]][metric].append(absolute_delta)

                    if t < 10 <= len(session_feedback):
                        metrics_over_10_steps['pairwise_accuracy'].append(pairwise_accuracy)

                    if args.reranker == "default":
                        break

                for metric in cumulative_metrics:
                    if args.reranker == "default":
                        session_length = 1
                    else:
                        session_length = len(session_feedback)
                    query_cumulative_metrics[metric] += cumulative_metrics[metric] / session_length

                for metric in metrics_over_10_steps:
                    total_metrics_over_10_steps[metric].append(metrics_over_10_steps[metric])

            for metric, value in query_cumulative_metrics.items():
                base_metrics[f"cumulative_{metric}"] = value / len(sessions)
            results[query_structures[0]].append(base_metrics)
        else:
            results[query_structures[0]].append(initial_metrics)

    metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    for query_structure in results:
        for metric in results[query_structure][0].keys():
            if metric in ['num_hard_answer', 'num_easy_answer']:
                continue
            metrics[query_structure][metric] = sum([result[metric] for result in results[query_structure]]) / len(results[query_structure])
        metrics[query_structure]['num_queries'] = len(results[query_structure])

    if args.save_scores:
        for structure, rankings_dict in query_to_ranking.items():
            scores_output_path = osp.join(output_path, "query-scores", structure)
            os.makedirs(scores_output_path)
            with open(osp.join(scores_output_path, "qto_scores.pkl"), "wb") as f:
                pickle.dump(rankings_dict, f)

    num_query_structures = 0
    num_queries = 0
    with open(osp.join(output_path, f'all_metrics_{mode}_{preference}.txt'), "w") as f:
        for query_structure in metrics:
            log_metrics(mode + " " + query_name_dict[query_structure],
                        metrics[query_structure],
                        file_pointer=f)
            for metric in metrics[query_structure]:
                all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
                if metric != 'num_queries':
                    average_metrics[metric] += metrics[query_structure][metric]
            num_queries += metrics[query_structure]['num_queries']
            num_query_structures += 1

    with open(osp.join(output_path, f'average_metrics_{mode}_{preference}.txt'), "w") as f:
        for metric in average_metrics:
            average_metrics[metric] /= num_query_structures
            all_metrics["_".join(["average", metric])] = average_metrics[metric]
        log_metrics('%s average' % mode, average_metrics, file_pointer=f)

    if evaluate_preferences:
        with open(osp.join(output_path, f"p_values_{mode}_{preference}.txt"), "w") as f:
            for query_structure in reranked_delta:
                p_values_dict = dict()
                for metric, deltas in reranked_delta[query_structure].items():
                    _, p_value = stats.wilcoxon(deltas, alternative="two-sided")
                    p_values_dict[metric] = p_value
                log_metrics(f"{mode} delta p_value {query_name_dict[query_structure]}", p_values_dict, file_pointer=f)

        with open(osp.join(output_path, f'metrics_over_time_{mode}_{preference}.pkl'), 'wb') as f:
            pickle.dump(total_metrics_over_10_steps, f)

    # Save timing results if profiling was enabled
    if args.profile_time and execution_times:
        with open(osp.join(output_path, f'execution_times_{mode}_{preference}_{args.reranker}.txt'), 'w') as f:
            f.write(
                f"Execution times for reranker '{args.reranker}' in {mode} mode with {preference} preference:\n")
            f.write(f"Total queries: {len(execution_times)}\n")
            f.write(f"Average time: {sum(execution_times) / len(execution_times):.6f} seconds\n")
            f.write(f"Min time: {min(execution_times):.6f} seconds\n")
            f.write(f"Max time: {max(execution_times):.6f} seconds\n")
            f.write("\nIndividual execution times (seconds):\n")
            for i, exec_time in enumerate(execution_times):
                f.write(f"{i + 1}: {exec_time:.6f}\n")

    return all_metrics


def load_data(args, tasks, split):
    """Load queries and remove queries not in tasks for the given split ("valid" or "test")."""
    logging.info(f"Loading {split} data")

    queries = pickle.load(open(os.path.join(args.data_path, f"{split}-queries.pkl"), 'rb'))
    if split == "train":
        easy_answers = pickle.load(open(os.path.join(args.data_path, f"train-answers.pkl"), 'rb'))
        hard_answers = None
    else:
        hard_answers = pickle.load(open(os.path.join(args.data_path, f"{split}-hard-answers.pkl"), 'rb'))
        easy_answers = pickle.load(open(os.path.join(args.data_path, f"{split}-easy-answers.pkl"), 'rb'))

    annotated_test_sessions_str = "-annotated" if args.test_annotated else ""
    sessions_path = osp.join(args.data_path, f"{split}-sessions{annotated_test_sessions_str}.pkl")
    if osp.exists(sessions_path):
        sessions = pickle.load(open(sessions_path, "rb"))
    else:
        sessions = dict()

    print(f"Loaded {len(sessions)} sessions from {sessions_path}")

    query_structures = list(queries.keys())
    for structure in query_structures:
        if query_name_dict[structure] not in tasks:
            queries.pop(structure)
        elif args.preference != "none":
            queries[structure] = {q for q in queries[structure] if q in sessions}

    return queries, hard_answers, easy_answers, sessions


def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open('%s/stats.txt' % args.data_path) as f:
        stats = f.readlines()
        num_entities = int(stats[0].split(' ')[-1])
        num_relations = int(stats[1].split(' ')[-1])

    global id2ent, id2rel
    with open('%s/id2ent.pkl' % args.data_path, 'rb') as f:
        id2ent = pickle.load(f)
    with open('%s/ent2id.pkl' % args.data_path, 'rb') as f:
        ent2id = pickle.load(f)
    with open('%s/id2rel.pkl' % args.data_path, 'rb') as f:
        id2rel = pickle.load(f)

    args.nentity = num_entities
    args.nrelation = num_relations

    adj_list, edges_y, edges_p = read_triples([os.path.join(args.data_path, "train.txt")], args.nrelation,
                                              args.data_path)
    model = KGReasoning(args, device, adj_list, query_name_dict, name_answer_dict,
                        use_reranker=args.reranker in ("ranknet", "lambdamart"),
                        hidden_dim=args.hidden_dim,
                        activation=args.activation)

    lgb_model = None
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}")
        if args.reranker == "lightgbm_lambdamart":
            lgb_model = lgb.Booster(model_file=args.checkpoint)
        else:
            model.load_state_dict(torch.load(args.checkpoint))
            model.to(device)

    pprint(vars(args))

    dataset_name = args.data_path.split('/')[-1]
    folder_name = f"{dataset_name}_{args.fraction}_{args.thrshd}_{args.reranker}"
    if args.reranker == "cosine":
        folder_name += f"_{args.alpha}_{args.beta}"
    elif args.reranker == "ranknet":
        folder_name += f"_{args.lr}"
    if args.do_valid:
        folder_name += "_valid"
    if args.do_test:
        folder_name += "_test"
    folder_name += f"_{args.preference}_{int(time.time())}"

    # Initialize wandb early to get run ID
    wandb.init(project="nqr", mode='online' if args.wandb else 'disabled', notes=args.notes)
    folder_name += f"_{wandb.run.id}"
    output_path = os.path.join('results', folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Now create wandb_config with valid output_path
    wandb_config = {**vars(args), "output_path": output_path}
    wandb.config.update(wandb_config)

    print(f"Running with output path {output_path}")

    if args.do_train:
        if args.reranker == "lightgbm_lambdamart":
            lgb_model = train_lightgbm(model, args, tasks, device, output_path)
        else:
            train(model, args, tasks, device, output_path)

    if args.do_valid:
        queries, hard_answers, easy_answers, sessions = load_data(args, tasks, "valid")
        queries = flatten_query(queries)
        dataloader = DataLoader(
            TestDataset(queries, sessions, args.nentity, args.nrelation),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )
        evaluate(model, hard_answers, easy_answers, args, dataloader, 
                query_name_dict, device, output_path, "valid", args.preference, lgb_model=lgb_model)

    if args.do_test:
        queries, hard_answers, easy_answers, sessions = load_data(args, tasks, "test")
        queries = flatten_query(queries)
        dataloader = DataLoader(
            TestDataset(queries, sessions, args.nentity, args.nrelation),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )
        evaluate(model, hard_answers, easy_answers, args, dataloader, 
                query_name_dict, device, output_path, "test", args.preference, lgb_model=lgb_model)

    print(f"Done, output path is {output_path}")


if __name__ == '__main__':
    main(parse_args())
