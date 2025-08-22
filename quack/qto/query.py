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

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats as stats
import wandb

from quack.qto.dataset import TestDataset, InfiniteDataLoaderIterator
from quack.qto.model import KGReasoning
from quack.qto.util import flatten_query, set_global_seed

print(  )

query_name_dict = {('e', ('r',)): '1p',
                   (('e', ('r',)), ('e', ('s',))): '1ps1',
                   (('e', ('r',)), ('e', ('s', 'n'))): '1pns1',

                   ('e', ('r', 'r')): '2p',
                   ((('e', ('r',)), ('e', ('s',))), ('r',)): '2ps1',
                   ((('e', ('r',)), ('e', ('s', 'n'))), ('r',)): '2pns1',
                   (('e', ('r', 'r')), ('e', ('s',))): '2ps2',
                   (('e', ('r', 'r')), ('e', ('s', 'n'))): '2pns2',
                   (((('e', ('r',)), ('e', ('s',))), ('r',)), ('e', ('s',))): '2ps12',
                   (((('e', ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('s', 'n'))): '2pns12',

                   ('e', ('r', 'r', 'r')): '3p',
                   ((('e', ('r',)), ('e', ('s',))), ('r', 'r')): '3ps1',
                   ((('e', ('r',)), ('e', ('s', 'n'))), ('r', 'r')): '3pns1',
                   ((('e', ('r', 'r')), ('e', ('s',))), ('r',)): '3ps2',
                   ((('e', ('r', 'r')), ('e', ('s', 'n'))), ('r',)): '3pns2',
                   (('e', ('r', 'r', 'r')), ('e', ('s',))): '3ps3',
                   (('e', ('r', 'r', 'r')), ('e', ('s', 'n'))): '3pns3',
                   (((((('e', ('r',)), ('e', ('s',))), ('r',)), ('e', ('s',))), ('r',)), ('e', ('s',))): '3ps123',
                   (((((('e', ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('s', 'n'))): '3pns123',

                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('s',))): '2is1',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('s', 'n'))): '2ins1',

                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('s',))): '3is1',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('s', 'n'))): '3ins1',

                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   ((('e', ('r',)), ('e', ('r',)), ('e', ('s',))), ('r',)): 'ips1',
                   ((('e', ('r',)), ('e', ('r',)), ('e', ('s', 'n'))), ('r',)): 'ipns1',
                   (((('e', ('r',)), ('e', ('r',))), ('r',)), ('e', ('s',))): 'ips2',
                   (((('e', ('r',)), ('e', ('r',))), ('r',)), ('e', ('s', 'n'))): 'ipns2',
                   (((('e', ('r',)), ('e', ('r',)), ('e', ('s',))), ('r',)), ('e', ('s',))): 'ips12',
                   (((('e', ('r',)), ('e', ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('s', 'n'))): 'ipns12',

                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (((('e', ('r',)), ('e', ('s',))), ('r',)), ('e', ('r',))): 'pis1',
                   (((('e', ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('r',))): 'pins1',
                   (('e', ('r', 'r')), ('e', ('r',)), ('e', ('s',))): 'pis2',
                   (('e', ('r', 'r')), ('e', ('r',)), ('e', ('s', 'n'))): 'pins2',
                   (((('e', ('r',)), ('e', ('s',))), ('r',)), ('e', ('r',)), ('e', ('s',))): 'pis12',
                   (((('e', ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('r',)), ('e', ('s', 'n'))): 'pins12',

                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r', 'n')), ('e', ('s',))): '2ins1',
                   (('e', ('r',)), ('e', ('r', 'n')), ('e', ('s', 'n'))): '2inns1',

                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')), ('e', ('s',))): '3ins1',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')), ('e', ('s', 'n'))): '3inns1',

                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   ((('e', ('r',)), ('e', ('r', 'n')), ('e', ('s',))), ('r',)): 'inps1',
                   ((('e', ('r',)), ('e', ('r', 'n')), ('e', ('s', 'n'))), ('r',)): 'inpns1',
                   (((('e', ('r',)), ('e', ('r', 'n'))), ('r',)), ('e', ('s',))): 'inps2',
                   (((('e', ('r',)), ('e', ('r', 'n'))), ('r',)), ('e', ('s', 'n'))): 'inpns2',
                   (((('e', ('r',)), ('e', ('r', 'n')), ('e', ('s',))), ('r',)), ('e', ('s',))): 'inps12',
                   (((('e', ('r',)), ('e', ('r', 'n')), ('e', ('s', 'n'))), ('r',)), ('e', ('s', 'n'))): 'inpns12',

                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (((('e', ('r',)), ('e', ('s',))), ('r',)), ('e', ('r', 'n'))): 'pins1',
                   (((('e', ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('r', 'n'))): 'pinns1',
                   (('e', ('r', 'r')), ('e', ('r', 'n')), ('e', ('s',))): 'pins2',
                   (('e', ('r', 'r')), ('e', ('r', 'n')), ('e', ('s', 'n'))): 'pinns2',
                   (((('e', ('r',)), ('e', ('s',))), ('r',)), ('e', ('r', 'n')), ('e', ('s',))): 'pins12',
                   (((('e', ('r',)), ('e', ('s', 'n'))), ('r',)), ('e', ('r', 'n')), ('e', ('s', 'n'))): 'pinns12',

                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (((('e', ('r',)), ('e', ('s',))), ('r', 'n')), ('e', ('r',))): 'pnis1',
                   (((('e', ('r',)), ('e', ('s', 'n'))), ('r', 'n')), ('e', ('r',))): 'pnins1',
                   (('e', ('r', 'r', 'n')), ('e', ('r',)), ('e', ('s',))): 'pnis2',
                   (('e', ('r', 'r', 'n')), ('e', ('r',)), ('e', ('s', 'n'))): 'pnins2',
                   (((('e', ('r',)), ('e', ('s',))), ('r', 'n')), ('e', ('r',)), ('e', ('s',))): 'pnis12',
                   (((('e', ('r',)), ('e', ('s', 'n'))), ('r', 'n')), ('e', ('r',)), ('e', ('s', 'n'))): 'pnins12',

                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('s',))): '2us1-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('s', 'n'))): '2uns1-DNF',

                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   (((('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('s',))), ('r',)): 'ups1-DNF',
                   (((('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('s', 'n'))), ('r',)): 'upns1-DNF',
                   (((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)), ('e', ('s',))): 'ups2-DNF',
                   (((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)), ('e', ('s', 'n'))): 'upns2-DNF',
                   ((((('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('s',))), ('r',)), ('e', ('s',))): 'ups12-DNF',
                   ((((('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('s', 'n'))), ('r',)), ('e', ('s', 'n'))): 'upns12-DNF',

                   ('e', ('r', 'r', 'r', 'r')): '4p',
                   ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '4i',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
                   }
name_answer_dict = {'1p': ['e', ['r',], 'e'],
                    '2p': ['e', ['r', 'e', 'r'], 'e'],
                    '3p': ['e', ['r', 'e', 'r', 'e', 'r'], 'e'],
                    '2i': [['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'],
                    '3i': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'],
                    'ip': [[['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'], ['r',], 'e'],
                    'pi': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r',], 'e'], 'e'],
                    '2in': [['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    '3in': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'inp': [[['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'], ['r',], 'e'],
                    'pin': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'pni': [['e', ['r', 'e', 'r', 'n'], 'e'], ['e', ['r',], 'e'], 'e'],
                    '2u-DNF': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['u',], 'e'],
                    'up-DNF': [[['e', ['r',], 'e'], ['e', ['r',], 'e'], ['u',], 'e'], ['r',], 'e'],
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())
espace = 9
rspace = 11


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_cp', action='store_true', help="do cardinality prediction")
    parser.add_argument('--path', action='store_true', help="do interpretation study")
    parser.add_argument('--wandb', action='store_true', help="log to wandb")
    parser.add_argument('--notes', default=None, type=str, help="notes for wandb")
    parser.add_argument('--test_run', action='store_true', help="run on a small dataset")
    parser.add_argument('--save_scores', action='store_true', help="save query scores for all entities")

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
    parser.add_argument('--force_training_edges', action='store_true', help='set scores for edges in training graph to 1')
    
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u-DNF.up-DNF', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=12345, type=int, help="random seed")
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument("--preference", default="none", choices=["positive", "negative", "mixed", "none"], help="preference type")
    parser.add_argument('--reranker',
                        default='nqr',
                        type=str,
                        choices=['default', 'random', 'greedy', 'cosine', 'ranknet', 'nqr', 'score', 'cosine_mean', 'logit', 'logitv2', 'logitv3', 'gated', 'beta', 'trustband', 'logitrank', 'tempered', 'residual', 'contrastive', 'relative'],
                        help='reranker method')
    parser.add_argument('--alpha_p', default=0.5, type=float, help="Alpha_p parameter for the cosine similarity reranker")
    parser.add_argument('--alpha_n', default=0.5, type=float, help="Alpha_n parameter for the cosine similarity reranker")
    parser.add_argument('--preference_embedding', default="none", choices=["none", "mean", "selfattn"], help="preference embedding method")
    parser.add_argument("--num_layers", default=2, choices=[1, 2], type=int, help="Number of layers for the preference embedding")
    parser.add_argument("--activation", default="relu", choices=["relu", "elu"], help="Activation function for the reranking network")
    parser.add_argument("--margin", default=0.1, type=float, help="margin for the nqr reranker")
    parser.add_argument("--kl_weight", default=1.0, type=float, help="kl divergence weight")
    return parser.parse_args(args)


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

                batch_preferences.append(torch.tensor(preferences, device=device))
                batch_labels.append(torch.tensor(labels, device=device))

            if len(batch_scores) == 0:
                break

            train_bar.update(len(batch_scores))

            batch_scores = torch.cat(batch_scores)
            batch_preferences = pad_sequence(batch_preferences, batch_first=True, padding_value=-1)
            batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-1)
            batch_positives = torch.tensor(batch_positives, device=device)
            batch_negatives = torch.tensor(batch_negatives, device=device)
            batch_positive_ids = torch.tensor(batch_positive_ids, device=device)
            batch_negative_ids = torch.tensor(batch_negative_ids, device=device)
            # and something similar for batch_inputs and batch_labels, once this is ready, we can do the following
            preference_loss, answer_loss, deltas = model.reranking_loss(
                batch_preferences,
                batch_labels,
                batch_scores,
                (batch_positives, batch_positive_ids),
                (batch_negatives, batch_negative_ids),
                use_nqr=args.reranker == "nqr"
            )
            loss = preference_loss + args.kl_weight * answer_loss
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
        if epoch % args.valid_frequency == 0:
            all_metrics = evaluate(model, valid_hard_answers, valid_easy_answers, args, valid_dataloader, query_name_dict, device, output_path, "valid", preference="mixed")
            wandb.log({f"valid_{k}": v for k, v in all_metrics.items() if "cumulative" in k})

    all_metrics = evaluate(model, test_hard_answers, test_easy_answers, args, test_dataloader, query_name_dict, device, output_path, "test", preference="mixed")
    wandb.log({f"test_{k}": v for k, v in all_metrics.items() if "cumulative" in k})
    torch.save(model.state_dict(), osp.join(output_path, f'{wandb.run.id}-model.pt'))


@torch.inference_mode()
def evaluate(model: KGReasoning, hard_answers, easy_answers, args, dataloader, query_name_dict, device, output_path, mode, preference):
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
    for flat_queries, queries, query_structures, sessions in tqdm(dataloader, desc=f"Evaluating on {mode} - preference: {preference}", mininterval=1):
        sessions = sessions[0]
        if evaluate_preferences and len(sessions) == 0:
            raise ValueError("No sessions found for query")

        flat_queries = torch.LongTensor(flat_queries).to(device)
        scores, _, exec_query = model.embed_query(flat_queries, query_structures[0], 0)
        scores = scores.squeeze()
        initial_metrics = compute_metrics(scores, hard_answers, easy_answers, queries)

        if args.save_scores:
            query_to_ranking[query_name_dict[query_structures[0]]][queries[0]] = scores.tolist()

        query_cumulative_metrics = defaultdict(float)

        if evaluate_preferences:
            for session in sessions:
                session_count += 1
                # session_scores = scores.clone()

                positives, negatives = session

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
                for t in range(len(session_feedback)):
                    # Rerank embedding scores based on preferences
                    preferences = torch.tensor(session_feedback[:t+1], device=device)
                    labels = torch.tensor(session_labels[:t+1], device=device)
                    if args.reranker == "default":
                        session_scores = scores
                    if args.reranker in ("cosine", "cosine_mean"):
                        session_scores = model.rerank_cosine(scores, preferences, labels, args.alpha_p, args.alpha_n, use_mean_cosine)
                    elif args.reranker == "random":
                        session_scores = model.rerank_random(scores, preferences, labels)
                    elif args.reranker == "greedy":
                        session_scores = model.rerank_greedy(scores, preferences, labels)
                    elif args.reranker in ("ranknet", "nqr"):
                        session_scores = model.rerank_nqr(scores, preferences, labels)
                    elif args.reranker == "score":
                        session_scores = model.rerank_score(scores, preferences, labels)
                    elif args.reranker == "logit":
                        session_scores = model.rerank_logit(scores, preferences, labels)
                    elif args.reranker == "logitv2":
                        session_scores = model.rerank_logit_v2(scores, preferences, labels)
                    elif args.reranker == "logitv3":
                        session_scores = model.rerank_logit_v3(scores, preferences, labels)
                    elif args.reranker == "gated":
                        session_scores = model.rerank_logit_gated(scores, preferences, labels)
                    elif args.reranker == "beta":
                        session_scores = model.rerank_beta(scores, preferences, labels)
                    elif args.reranker == "trustband":
                        session_scores = model.rerank_logit_trustband(scores, preferences, labels)
                    elif args.reranker == "logitrank":
                        session_scores = model.rerank_beta(scores, preferences, labels)
                    elif args.reranker == "tempered":
                        session_scores = model.rerank_logit_tempered(scores, preferences, labels)
                    elif args.reranker == "residual":
                        session_scores = model.rerank_logit_residual(scores, preferences, labels)
                    elif args.reranker == "contrastive":
                        session_scores = model.rerank_logit_contrastive(scores, preferences, labels)
                    elif args.reranker == "relative":
                        session_scores = model.rerank_logit_relative(scores, preferences, labels)

                    # Compute pairwise accuracy after reranking
                    pos_scores = session_scores[positives].unsqueeze(1)
                    neg_scores = session_scores[negatives].unsqueeze(0)
                    num_pairs = len(positives) * len(negatives)
                    pairwise_accuracy = (pos_scores > neg_scores).sum().item() / num_pairs
                    cumulative_metrics["pairwise_accuracy"] += pairwise_accuracy

                    instant_metrics = compute_metrics(session_scores, hard_answers, easy_answers, queries)

                    for metric in instant_metrics:
                        if metric.startswith('num'):
                            continue
                        absolute_delta = instant_metrics[metric] - initial_metrics[metric]
                        relative_delta = absolute_delta / (1.0 if initial_metrics[metric] == 0 else initial_metrics[metric])
                        cumulative_metrics[f"{metric}"] += instant_metrics[metric]
                        cumulative_metrics[f"{metric}_delta"] += relative_delta
                        if t < 10 <= len(session_feedback):
                            if t == 0:
                                metrics_over_10_steps[metric].append(initial_metrics[metric])
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
                initial_metrics[f"cumulative_{metric}"] = value / len(sessions)

        results[query_structures[0]].append(initial_metrics)

    metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    for query_structure in results:
        for metric in results[query_structure][0].keys():
            if metric in ['num_hard_answer', 'num_easy_answer']:
                continue
            metrics[query_structure][metric] = sum([result[metric] for result in results[query_structure]])/len(results[query_structure])
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
        log_metrics('%s average'%mode, average_metrics, file_pointer=f)

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
    sessions_path = osp.join(args.data_path, f"{split}-sessions.pkl")
    if osp.exists(sessions_path):
        sessions = pickle.load(open(os.path.join(args.data_path, f"{split}-sessions.pkl"), "rb"))
    else:
        sessions = dict()

    query_structures = list(queries.keys())
    for structure in query_structures:
        if query_name_dict[structure] not in tasks:
            queries.pop(structure)
        elif args.preference != "none":
                queries[structure] = {q for q in queries[structure] if q in sessions}

    return queries, hard_answers, easy_answers, sessions


def evaluate_split(args, tasks, split, model, query_name_dict, device, output_path, preference):
    queries, hard_answers, easy_answers, sessions = load_data(args, tasks, split)
    queries = flatten_query(queries)
    dataloader = DataLoader(
        TestDataset(
            queries,
            sessions,
            args.nentity,
            args.nrelation,
        ),
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )
    evaluate(model, hard_answers, easy_answers, args, dataloader, query_name_dict, device, output_path, split, preference)


def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open('%s/stats.txt'%args.data_path) as f:
        stats = f.readlines()
        num_entities = int(stats[0].split(' ')[-1])
        num_relations = int(stats[1].split(' ')[-1])
    
    global id2ent, id2rel
    with open('%s/id2ent.pkl'%args.data_path, 'rb') as f:
        id2ent = pickle.load(f)
    with open('%s/ent2id.pkl'%args.data_path, 'rb') as f:
        ent2id = pickle.load(f)
    with open('%s/id2rel.pkl'%args.data_path, 'rb') as f:
        id2rel = pickle.load(f)

    args.nentity = num_entities
    args.nrelation = num_relations

    adj_list, edges_y, edges_p = read_triples([os.path.join(args.data_path, "train.txt")], args.nrelation, args.data_path)
    model = KGReasoning(args, device, adj_list, query_name_dict, name_answer_dict, args.preference_embedding, args.num_layers, args.activation, args.margin)

    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
        model.to(device)

    pprint(vars(args))

    dataset_name = args.data_path.split('/')[-1]
    folder_name = f"{dataset_name}_{args.fraction}_{args.thrshd}_{args.reranker}"
    if args.reranker == "cosine":
        folder_name += f"_{args.alpha_p}_{args.alpha_n}"
    elif args.reranker == "nqr":
        folder_name += f"_{args.lr}"
    if args.do_valid:
        folder_name += "_valid"
    if args.do_test:
        folder_name += "_test"
    folder_name += f"_{args.preference}_{int(time.time())}"

    # Initialize wandb early to get run ID
    wandb.init(project="quack", mode='online' if args.wandb else 'disabled', notes=args.notes)
    folder_name += f"_{wandb.run.id}"
    output_path = os.path.join('results', folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Now create wandb_config with valid output_path
    wandb_config = {**vars(args), "output_path": output_path}
    wandb.config.update(wandb_config)

    print(f"Running with output path {output_path}")

    if args.do_train:
        train(model, args, tasks, device, output_path)

    if args.do_valid:
        evaluate_split(args, tasks, "valid", model, query_name_dict, device, output_path, args.preference)

    if args.do_test:
        evaluate_split(args, tasks, "test", model, query_name_dict, device, output_path, args.preference)

    print(f"Done, output path is {output_path}")

if __name__ == '__main__':
    main(parse_args())
