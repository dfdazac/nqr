# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import errno
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser


def prepare_dataset(path):
    """KBC dataset preprocessing. 
    1) Maps each entity and relation to a unique id
    2) Create a corresponding folder of `cwd/data/dataset`, with mapped train/test/valid files.
    3) Create `to_skip_lhs` & `to_skip_rhs` for filtered metrics
    4) Save the mapping `rel_id` & `ent_id` for analysis.

    Args:
        path: a path of a folder containing 3 tab-separated files, `train`, `valid` and `test`.
        name: name of the dataset
    """
    files = ['train.txt', 'valid.txt', 'test.txt']
    data_path = Path(path)

    entities_to_id, relations_to_id = {}, {}
    for f in files:
        file_path = data_path / f
        with open(file_path, 'r') as f:
            for line in f:
                lhs, rel, rhs = line.strip().split('\t')
                if lhs not in entities_to_id:
                    entities_to_id[lhs] = len(entities_to_id)
                if rhs not in entities_to_id:
                    entities_to_id[rhs] = len(entities_to_id)
                if rel not in relations_to_id:
                    relations_to_id[rel] = len(relations_to_id)

    n_relations = len(relations_to_id)
    n_entities = len(entities_to_id)
    print("{} entities and {} relations".format(n_entities, n_relations))

    kbc_data_path = data_path / 'kbc_data'
    os.makedirs(kbc_data_path)
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        with open(kbc_data_path / f, 'w+') as f:
            for (x, i) in dic.items():
                f.write("{}\t{}\n".format(x, i))

    # map train/test/valid with the ids
    for f in files:
        file_path = data_path / f
        with open(file_path, 'r') as triples_file:
            examples = []
            for line in triples_file:
                lhs, rel, rhs = line.strip().split('\t')
                examples.append([entities_to_id[lhs],
                                 relations_to_id[rel],
                                 entities_to_id[rhs]])

        with open(kbc_data_path / (f + '.pickle'), 'wb') as out:
            pickle.dump(np.array(examples).astype('uint64'), out)

    print("Creating filtering lists")

    # create filtering files 
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        with open(kbc_data_path / (f + '.pickle'), 'rb') as f:
            examples = pickle.load(f)

        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for pos, skip in to_skip.items():
        for query, ans in skip.items():
            to_skip_final[pos][query] = sorted(list(ans))

    with open(kbc_data_path / 'to_skip.pickle', 'wb') as out:
        pickle.dump(to_skip_final, out)
    print('Done processing!')


def main():
    parser = ArgumentParser(description="Preprocess datasets for KBC")
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()
    path = args.data_path
    dataset = os.path.basename(path)

    try:
        prepare_dataset(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(e)
            print("File exists. skipping...")
        else:
            raise


if __name__ == "__main__":
    main()
