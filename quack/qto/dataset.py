import numpy as np
import torch

from torch.utils.data import Dataset
from .util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple, flatten

class TestDataset(Dataset):
    def __init__(self, queries, sessions, nentity, nrelation):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.sessions = sessions
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        sessions = self.sessions[query] if query in self.sessions else []
        query_structure = self.queries[idx][1]
        return flatten(query), query, query_structure, sessions
    
    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        query_unflatten = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        sessions = [_[3] for _ in data]
        return query, query_unflatten, query_structure, sessions


class InfiniteDataLoaderIterator:
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
