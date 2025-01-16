import numpy as np
import torch

from torch.utils.data import Dataset
from .util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple, flatten

class TestDataset(Dataset):
    def __init__(self, queries, subsets, nentity, nrelation):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.subsets = subsets
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        subsets = self.subsets[query]
        query_structure = self.queries[idx][1]
        return flatten(query), query, query_structure, subsets
    
    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        query_unflatten = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        subsets = [_[3] for _ in data]
        return query, query_unflatten, query_structure, subsets
