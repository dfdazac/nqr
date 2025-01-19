import argparse
import collections
import logging
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TestDataset
from .model import KGReasoning
from .util import flatten_query, set_global_seed


query_name_dict = {('e', ('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
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
    
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_cp', action='store_true', help="do cardinality prediction")
    parser.add_argument('--path', action='store_true', help="do interpretation study")

    parser.add_argument('--train', action='store_true', help="do test")
    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--kbc_path', type=str, default=None, help="kbc model path")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=0, type=int, help="used to speed up torch.dataloader")
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--fraction', type=int, default=1, help='fraction the entity to save gpu memory usage')
    parser.add_argument('--thrshd', type=float, default=0.001, help='thrshd for neural adjacency matrix')
    parser.add_argument('--neg_scale', type=int, default=1, help='scaling neural adjacency matrix for negation')
    
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=12345, type=int, help="random seed")
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    return parser.parse_args(args)


def log_metrics(mode, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s: %f' % (mode, metric, metrics[metric]))
        print('%s %s: %f' % (mode, metric, metrics[metric]))
        writer.write('%s %s: %f\n' % (mode, metric, metrics[metric]))


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
            'MRR_hard': mrr_hard,
            'HITS1_hard': h1_hard,
            'HITS3_hard': h3_hard,
            'HITS10_hard': h10_hard,
            'num_hard_answer': num_hard,
            'MRR_easy': mrr_easy,
            'HITS1_easy': h1_easy,
            'HITS3_easy': h3_easy,
            'HITS10_easy': h10_easy,
            'num_easy_answer': num_easy,
        }

@torch.inference_mode()
def evaluate(model, hard_answers, easy_answers, args, dataloader, query_name_dict, device, writer, edges_y, edges_p, cp_thrshd):
    '''
    Evaluate queries in dataloader
    '''
    mode = "Test"
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)
    logs = defaultdict(list)
    session_count = 0
    total_cumulative_pairwise_accuracy = 0
    total_metrics_delta = defaultdict(float)

    total_metrics_over_t_10 = defaultdict(list)

    for flat_queries, queries, query_structures, sessions in tqdm(dataloader):
        sessions = sessions[0]
        if len(sessions) == 0:
            continue

        flat_queries = torch.LongTensor(flat_queries).to(device)
        embedding, _, exec_query = model.embed_query(flat_queries, query_structures[0], 0)
        embedding = embedding.squeeze()
        initial_metrics = compute_metrics(embedding, hard_answers, easy_answers, queries)

        done = False

        for session in sessions:
            session_count += 1
            session_embedding = embedding.clone()

            positives, negatives = session
            cumulative_pairwise_accuracy = 0
            metrics_over_t_10 = defaultdict(list)
            for t in range(len(positives)):
                # Rerank embedding scores based on feedback
                preferences = torch.tensor(positives[:t+1], device=device)
                session_embedding = model.rerank(session_embedding, preferences, alpha=0.5)
                # embedding = reranker.rerank(embedding, feedback)

                # Compute pairwise accuracy after reranking
                pos_scores = session_embedding[positives].unsqueeze(1)
                neg_scores = session_embedding[negatives].unsqueeze(0)
                num_pairs = len(positives) * len(negatives)
                pairwise_accuracy = (pos_scores > neg_scores).sum().item()
                cumulative_pairwise_accuracy += pairwise_accuracy / num_pairs

                if len(positives) == 10:
                    instant_metrics = compute_metrics(session_embedding, hard_answers, easy_answers, queries)
                    for metric in instant_metrics:
                        if metric.startswith('num'):
                            continue
                        metrics_over_t_10[metric].append(instant_metrics[metric] - initial_metrics[metric])
                    metrics_over_t_10['pairwise_accuracy'].append(pairwise_accuracy / num_pairs)

            for metric in metrics_over_t_10:
                total_metrics_over_t_10[metric].append(metrics_over_t_10[metric])

                # if len(total_metrics_over_t_10[metric]) == 10:
                #     done = True

            session_metrics = compute_metrics(session_embedding, hard_answers, easy_answers, queries)
            for metric in session_metrics:
                if metric.startswith('num'):
                    continue
                total_metrics_delta[metric] += session_metrics[metric] - initial_metrics[metric]

            cumulative_pairwise_accuracy /= len(positives)
            total_cumulative_pairwise_accuracy += cumulative_pairwise_accuracy

        logs[query_structures[0]].append(initial_metrics)

        if done:
            break

    average_cum_pairwise_accuracy = total_cumulative_pairwise_accuracy / max(1, session_count)
    log_metrics('Preference', {'Cumulative Pairwise Accuracy': average_cum_pairwise_accuracy}, writer)

    # Plot metrics over time for queries with 10 answers
    # Extract data for MRR_hard and pairwise_accuracy
    mrr_hard = total_metrics_over_t_10['MRR_hard']  # List of lists
    pairwise_acc = total_metrics_over_t_10['pairwise_accuracy']  # List of lists

    # Convert lists of lists into numpy arrays for easier manipulation
    mrr_hard = np.array(mrr_hard)  # Shape: (num_queries, num_timesteps)
    pairwise_acc = np.array(pairwise_acc)  # Shape: (num_queries, num_timesteps)

    # Define time steps
    num_timesteps = mrr_hard.shape[1]
    time_steps = np.arange(1, num_timesteps + 1)
    # Create subplots for boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Metrics and titles for looping
    metrics = [pairwise_acc, mrr_hard]
    titles = ['Pairwise Accuracy', '$\Delta$MRR']

    # Loop over the metrics and plot boxplots
    for ax, metric, title in zip(axes, metrics, titles):
        ax.boxplot(metric, positions=time_steps, widths=0.6, medianprops={'color': 'black'})
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Number of interactions')
        ax.set_ylabel('Value')
        if title == 'Pairwise Accuracy':
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.set_ylim(-1.1, 1.1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.axhline(0, color='black', linewidth=1)  # Horizontal line at y=0
        ax.axvline(0, color='black', linewidth=1)  # Vertical line at x=0

    fig.suptitle('Distribution of Metrics Over Time')
    plt.tight_layout()
    plt.show()

    average_metrics_delta = dict()
    for metric in total_metrics_delta:
        average_metrics_delta[metric] = total_metrics_delta[metric] / session_count
    log_metrics('Average ranking delta', average_metrics_delta, writer)

    metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    for query_structure in logs:
        for metric in logs[query_structure][0].keys():
            if metric in ['num_hard_answer', 'num_easy_answer']:
                continue
            metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
        metrics[query_structure]['num_queries'] = len(logs[query_structure])
    
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], metrics[query_structure], writer)
        for metric in metrics[query_structure]:
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, average_metrics, writer)

    writer.write('\n')
    return all_metrics

def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))
    valid_sessions = pickle.load(open(os.path.join(args.data_path, "valid-sessions.pkl"), "rb"))
    test_sessions = pickle.load(open(os.path.join(args.data_path, "test-sessions.pkl"), "rb"))
    
    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in valid_sessions:
                del valid_sessions[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]
            if query_structure in test_sessions:
                del test_sessions[query_structure]

    return valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, valid_sessions, test_sessions

def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_name = args.data_path.split('/')[1].split('-')[0]
    if args.data_path.split('/')[1].split('-')[1] == "237":
        dataset_name += "-237"
    filename = os.path.join('results', dataset_name+'_'+str(args.fraction)+'_'+str(args.thrshd)+'.txt')
    if not os.path.exists('results'):
        os.makedirs('results')
    writer = open(filename, 'a+')

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    
    global id2ent, id2rel
    with open('%s/id2ent.pkl'%args.data_path, 'rb') as f:
        id2ent = pickle.load(f)
    with open('%s/ent2id.pkl'%args.data_path, 'rb') as f:
        ent2id = pickle.load(f)
    with open('%s/id2rel.pkl'%args.data_path, 'rb') as f:
        id2rel = pickle.load(f)
    
    args.nentity = nentity
    args.nrelation = nrelation

    adj_list, edges_y, edges_p = read_triples([os.path.join(args.data_path, "train.txt")], args.nrelation, args.data_path)

    valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, valid_sessions, test_sessions = load_data(args, tasks)
    
    valid_queries = flatten_query(valid_queries)
    valid_dataloader = DataLoader(
        TestDataset(
            valid_queries,
            valid_sessions,
            args.nentity, 
            args.nrelation, 
        ), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn
    )

    test_queries = flatten_query(test_queries)
    test_dataloader = DataLoader(
        TestDataset(
            test_queries,
            test_sessions,
            args.nentity, 
            args.nrelation, 
        ), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn
    )
    
    model = KGReasoning(args, device, adj_list, query_name_dict, name_answer_dict)

    cp_thrshd = None
    
    evaluate(model, test_hard_answers, test_easy_answers, args, test_dataloader, query_name_dict, device, writer, edges_y, edges_p, cp_thrshd)

if __name__ == '__main__':
    main(parse_args())