import random
import sys, os
from collections import Counter
from time import time
from typing import List

import numpy as np
import psutil
from rec.utils import tb
import torch
import tqdm

from rec.dataset.dataset import Dataset
from rec.recommender.dml.dml import DMLSessionRecommender
from rec.recommender.dml.utils import shuffle
from rec.recommender import VMSessionKnnRecommender


class BatchedSampler:
    def __init__(self, rec: DMLSessionRecommender, pad=0):
        self.pad = pad
        self.rec = rec
        self.tensors = []

        # faster access
        self._device = rec._device
        self.train_sessions = rec.train_sessions
        self.items_id_to_idx = rec.items_id_to_idx
        self.items_idx_to_id = rec.items_idx_to_id
        self.max_len = rec.max_len
        self.items_num = rec.items_num
        self.M = rec.M
        self.c = rec.c
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        t_start = time()
        self.tensors = self.sample()
        for i, t in enumerate(self.tensors):
            assert (type(t) is list) or (type(t) is np.ndarray) or (type(t) is tuple)

        self.tensors = [np.array(t, dtype=np.long) for t in self.tensors]

        # all tensors have the same dimension
        assert 1 == len(np.unique([t.shape[0] for t in self.tensors]))

        for i, t in enumerate(self.tensors):
            v = t.flatten()
            if type(self.pad) is int:
                v = v[v != self.pad]
            tb.add_histogram(f'sampler/x{i}-hist', v, self.calls)
            tb.add_scalar(f'sampler/x{i}-unique', len(np.unique(v)), self.calls)

        t_take = time() - t_start

        t_start = time()
        self.tensors = [torch.LongTensor(t).to(self._device) for t in self.tensors]
        m_take = time() - t_start
        print(f'Data prep. time: {t_take:>3.0f} and {m_take:>3.0f} (s). ', end='\n')

    def sample(self):
        pass

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, i):
        return [t[i] for t in self.tensors]

    def session_padding(self, s):
        n = (self.max_len - len(s))
        if n == 0:
            return s
        if self.pad == 'same':
            p = s[0]
        else:
            p = self.pad
        return [p] * (self.max_len - len(s)) + s  # front padding


class SessionPosNegSampler(BatchedSampler):
    def __init__(self, rec: DMLSessionRecommender, batch_size=32, pos_len=5, neg_len=5, pad=0, wtf=False):
        super().__init__(rec, pad)
        self.wtf = wtf
        self.neg_len = neg_len
        self.pos_len = pos_len
        self.batch_size = batch_size

    def sample(self):
        all_x1 = []
        all_x2 = []
        all_x3 = []
        batch_x1 = []
        set_x2 = set()
        batch_x2 = []
        c = Counter()
        shs = set()
        indices = np.random.permutation(np.arange(len(self.train_sessions)))
        for _idx in indices:
            s = self.train_sessions[_idx]
            split_idx = random.randrange(1, len(s)) if len(s) > 1 else 1
            x1 = s[:split_idx]
            x2 = s[split_idx:split_idx + self.pos_len]
            if len(set_x2.intersection(set(x2))) > 0:
                continue
            set_x2 = set_x2.union(set(x2))
            # add padding
            x1 = x1[-self.max_len:]
            c[len(x1)] += 1
            needed_pos = self.pos_len - len(x2)
            if needed_pos > 0:
                x2 = x2 + x1[-needed_pos:]  # back padding
            needed_pos = self.pos_len - len(x2)
            if needed_pos > 0:
                x2 = x2 + [x2[0]] * needed_pos
            # coo
            x1 = self.session_padding(x1)
            batch_x1.append(x1)
            shs.add(hash(tuple(x1)))
            batch_x2.append(x2)
            if len(batch_x1) == self.batch_size:
                _batch_x1 = np.array(batch_x1, dtype=np.int32)
                _batch_x2 = np.array(batch_x2, dtype=np.int32)
                all_x1.append(_batch_x1)
                all_x2.append(_batch_x2)
                # negatives
                _batch_x3 = np.zeros((self.batch_size, self.neg_len), dtype=np.long)
                for _i in range(self.batch_size):
                    p_items = _batch_x2[_i]
                    # negative sampling
                    p = self.c.copy()
                    # pos_items = np.array(list(set(i for s in batch_x1 for i in s).union(set_x2)))
                    if self.wtf:
                        p[p_items] = 0
                        p = p.max() - p  # inverse for neg. sampling
                    else:
                        p = (p.max() + 1) - p  # inverse for neg. sampling
                        p[p_items] = 0
                    p[self.pad] = 0
                    p = p / p.sum()
                    _batch_x3[_i] = np.random.choice(np.arange(len(p)), self.neg_len, True, p)
                all_x3.append(_batch_x3)

                # clean
                batch_x1 = []
                batch_x2 = []
                batch_x1_x2_coo = []
                batch_x2_coo = []
                set_x2 = set()
        print('Sessions len hist.:', [v for k, v in sorted(list(c.items()))], end=' ')
        print('Seq.:', len(shs))
        tb.add_scalar(f'sampler/seq-num', len(all_x1) * self.batch_size, self.calls)
        tb.add_scalar(f'sampler/seq-uniq', len(shs), self.calls)
        return all_x1, all_x2, all_x3


class SlidingWindowSampler(BatchedSampler):
    def __init__(self, rec: DMLSessionRecommender, batch_size=32, pos_len=5, neg_len=5, pop_neg=True, pad=0):
        super().__init__(rec, pad)
        self.neg_len = neg_len
        self.pos_len = pos_len
        self.batch_size = batch_size

        self._generate_sequences()

    @staticmethod
    def _sliding_window(tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, -step_size):
                if i - window_size >= 0:
                    yield (tensor[i - window_size:i], window_size)
                else:
                    break
        else:
            num_paddings = window_size - len(tensor)
            # Pad sequence with 0s if it is shorter than windows size.
            yield (np.pad(tensor, (num_paddings, 0), 'constant'), len(tensor))

    def _generate_sequences(self):
        counts = [len(s) for s in self.train_sessions]
        max_sequence_length = self.pos_len + self.max_len
        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])
        self.sequences = np.zeros((num_subsequences, self.max_len), dtype=np.int64)
        self.sequences_targets = np.zeros((num_subsequences, self.pos_len), dtype=np.int64)
        self.session_indices = np.zeros((num_subsequences, 1), dtype=np.int64)
        for s_idx, s in enumerate(self.train_sessions):
            for i, (seq, seq_elem) in enumerate(self._sliding_window(s, max_sequence_length)):
                self.sequences_targets[i][:] = seq[-self.pos_len:]
                self.sequences[i][:] = seq[:self.max_len]
                self.session_indices[i][0] = s_idx

    def _generate_negative_samples(self, session_indices):
        sessions_ = session_indices.squeeze()
        negative_samples = np.zeros((session_indices.shape[0], self.neg_len), np.int64)
        items_num = self.items_num - 1
        for i, s_idx in enumerate(sessions_):
            s = set(self.train_sessions[s_idx])
            s_items = len(s)
            candidates_set = set(np.random.randint(1, items_num, s_items + self.neg_len + 1))
            candidates_set -= s
            l = list(candidates_set)
            assert max(l) < (items_num + 1)
            negative_samples[i][:] = l[:self.neg_len]
        return negative_samples

    def sample(self):
        sequences_np, targets_np, session_indices = shuffle(
            self.sequences, self.sequences_targets, self.session_indices
        )
        negatives_np = self._generate_negative_samples(session_indices)

        # batch splitting
        bn = sequences_np.shape[0] // self.batch_size
        sequences_np = sequences_np[:bn * self.batch_size]
        targets_np = targets_np[:bn * self.batch_size]
        negatives_np = negatives_np[:bn * self.batch_size]

        return (
            sequences_np.reshape(bn, self.batch_size,
                                 self.max_len), targets_np.reshape(bn, self.batch_size, self.pos_len),
            negatives_np.reshape(bn, self.batch_size, self.neg_len)
        )


class PerItemSampler(BatchedSampler):
    def __init__(self, rec: DMLSessionRecommender, batch_items=8, instances=4, pos_len=4, pad=0):
        super().__init__(rec, pad)
        self.pos_len = pos_len
        self.instances = instances
        self.batch_items = batch_items
        print(f'Building inverted item-session index...')
        self.item_idx_to_session_idx = {idx: [] for idx, id in self.items_idx_to_id.items()}
        for s_idx, s in enumerate(self.train_sessions):
            for i in s[1:]:  # cannot be the first one
                self.item_idx_to_session_idx[i].append(s_idx)
        print('Inverted index built.')

    def sample(self):
        skipped_items = 0
        all_x1 = []
        all_x2 = []
        all_x3 = []
        batch_x1 = []
        batch_x2 = []
        batch_x3 = []
        items = set()
        c = Counter()
        indices = list(np.random.permutation(list(self.item_idx_to_session_idx.keys())))
        batch_items = 0
        for i in indices:
            sessions = []
            for s_idx in self.item_idx_to_session_idx[i]:
                s = self.train_sessions[s_idx]
                if len(items.intersection(s)) == 0:
                    sessions.append(s_idx)
            if len(sessions) < 2:
                skipped_items += 1
                continue
            session_indices = np.random.choice(sessions, self.instances)
            for s_idx in session_indices:
                s = self.train_sessions[s_idx]
                ii = s[1:].index(i)
                x1 = s[:ii][-self.max_len:]
                c[len(x1)] += 1
                x2 = s[ii:(ii + self.pos_len)]
                needed_pos = self.pos_len - len(x2)
                if needed_pos > 0:
                    x2 = x2 + x1[-needed_pos:]  # back padding
                needed_pos = self.pos_len - len(x2)
                if needed_pos > 0:
                    x2 = x2 + [x2[0]] * needed_pos
                x1 = self.session_padding(x1)
                batch_x1.append(x1)
                batch_x2.append(x2)
                items = items.union(x2)
            batch_x3.append(i)
            batch_items += 1

            if batch_items == self.batch_items:
                _batch_x1 = np.array(batch_x1, dtype=np.int32)
                _batch_x2 = np.array(batch_x2, dtype=np.int32)
                _batch_x3 = np.array(batch_x3, dtype=np.int32)
                all_x1.append(_batch_x1)
                all_x2.append(_batch_x2)
                all_x3.append(_batch_x3)
                # clean
                batch_items = 0
                batch_x1 = []
                batch_x2 = []
                batch_x3 = []
                items = set()

        print(f'Skipped items: {skipped_items}')
        tb.add_scalar('sampler/skipped_items', skipped_items, self.calls)
        return all_x1, all_x2, all_x3


class BalancedPerItemSampler(BatchedSampler):
    def __init__(self, rec: DMLSessionRecommender, topk_items=2000, batch_items=8, instances=4, pad=0):
        super().__init__(rec, pad)
        self.instances = instances
        self.batch_items = batch_items
        self.topk_items = topk_items
        if topk_items > 1.0:
            self.selected_num = topk_items
        else:
            self.selected_num = topk_items * self.items_num
        print(f'Taking only of most popular items: {self.selected_num}')
        self.selected_idx = np.array(self.c).argsort()[::-1][:self.selected_num]
        self.selected_idx_to_lbl = {i: n for n, i in enumerate(self.selected_idx)}
        print(f'Building inverted item-session index...')
        self.item_idx_to_session_idx = {i: [] for i in self.selected_idx}
        for s_idx, s in enumerate(self.train_sessions):
            for i in s[1:]:  # cannot be the first one
                if i in self.selected_idx:
                    self.item_idx_to_session_idx[i].append(s_idx)
        print('Index builded.')

    def sample(self):
        all_x1 = []
        all_x2 = []
        all_x3 = []
        batch_x1 = []
        batch_x2 = []
        batch_x3 = []
        set_x2 = set()
        c = Counter()
        indices = list(np.random.permutation(self.selected_idx.copy()))
        batch_items = 0
        for i in indices:
            sessions = self.item_idx_to_session_idx[i]
            if len(sessions) < 3:
                continue
            session_indices = np.random.choice(sessions, self.instances)
            for s_idx in session_indices:
                s = self.train_sessions[s_idx]
                set_x2 = set_x2.union(set(s))
                ii = s[1:].index(i)
                split_idx = random.randrange(1, ii + 2)
                x1 = s[:split_idx][-self.max_len:]
                c[len(x1)] += 1
                x1 = [0] * (self.max_len - len(x1)) + x1  # front padding
                batch_x1.append(x1)
            batch_x2.append([i])
            batch_x3.append(self.selected_idx_to_lbl[i])
            batch_items += 1

            if batch_items == self.batch_items:
                _batch_x1 = np.array(batch_x1, dtype=np.int32)
                _batch_x2 = np.array(batch_x2, dtype=np.int32)
                _batch_x3 = np.array(batch_x3, dtype=np.int32)
                all_x1.append(_batch_x1)
                all_x2.append(_batch_x2)
                all_x3.append(_batch_x3)
                # clean
                batch_items = 0
                batch_x1 = []
                batch_x2 = []
                batch_x3 = []
                set_x2 = set()

        return all_x1, all_x2, all_x3


class Node:
    def __init__(self, parent, item):
        self.count = 0
        self.children = {}
        self.parent = parent
        self.item = item

    def get_and_update(self, child_key):
        if not child_key in self.children.keys():
            self.children[child_key] = Node(self, child_key)
        node = self.children[child_key]
        node.count += 1
        return node

    def find(self, seq: List[int]):
        while len(seq) > 0 and seq[0] == 0:
            seq = seq[1:]
        if len(seq) == 0:
            return self
        elif seq[0] in self.children:
            return self.children[seq[0]].find(seq[1:])
        else:
            return Node(None, None)

    def __getitem__(self, seq: List[int]):
        return self.find(seq).count

    def prefix(self):
        r = []
        n = self
        while n.parent != None:
            r.append(n.item)
            n = n.parent
        return r[::-1]

    def sub_nodes(self, deepth=-1):
        if deepth > 0:
            return dfs(self, deepth)
        else:
            return dfs(self, deepth)[1:]

    def next_items(self, items_num, excluded_items=set()):
        return [n.prefix()[-items_num:] for n in dfs(self, deepth=items_num)]


def dfs(n, deepth=-1):
    if (len(n.children) == 0 and deepth == -1) or deepth == 0:
        return [n]
    else:
        r = [] if (deepth > 0 or n.parent is None) else [n]
        for c in n.children.values():
            r.extend(dfs(c, deepth - 1))
        return r


class EventSequenceCounter:
    def __init__(self, sessions: List[List[int]], n_items: int, item_frequencies: np.ndarray = None):
        self._items = set()
        self.n_items = n_items
        update_frequencies = False
        if item_frequencies is None:
            self.item_counts = np.zeros((n_items))
            update_frequencies = True
        else:
            self.item_counts = item_frequencies.copy()

        self.root = Node(None, None)
        for s in sessions:
            node = self.root
            for ev in s:
                if ev != 0:
                    node = node.get_and_update(ev)
                    if update_frequencies:
                        self.item_counts[ev] += 1

    def __getitem__(self, seq: List[int]):
        return self.root[seq]

    def next_items_counts(self, prefix: List[int], extended=False):
        res = np.zeros((self.n_items))
        total = 0
        node = self.root.find(prefix)
        for k, n in node.children.items():
            res[k] = n.count
            total += n.count
        return (res, total, len(node.children)) if extended else res

    def next_negative_items(self, prefix):
        counts, total, n_items = self.next_items_counts(prefix, extended=True)
        if n_items >= self.n_items - 1:
            p = counts / total
            p[1:] = (1 - p[1:]) / (self.n_items - 2)
        else:
            zero_counts_mask = 1 - np.sign(counts)
            p = self.item_counts * zero_counts_mask
            p = p / p.sum()

        return p

    def travels(self, prefix_len=-1):
        return dfs(self.root, prefix_len)


class SubSeqSampler(BatchedSampler):
    def __init__(self, rec: DMLSessionRecommender, batch_size=32, pos_len=5, neg_len=5, pad=0, warmup_calls=0):
        super().__init__(rec, pad)
        self.warmup_calls = warmup_calls
        self.neg_len = neg_len
        self.pos_len = pos_len
        self.batch_size = batch_size
        self.event_sequences_counter = EventSequenceCounter(self.train_sessions, self.items_num)
        self.train_max_len = self.max_len

    def sample(self):
        all_x1 = []
        all_x2 = []
        all_x3 = []
        batch_x1 = []
        batch_x2 = []
        set_x2 = set()
        c = Counter()
        skipped = 0
        skipped_inter = 0
        all_seq = 0

        if self.warmup_calls > 0 and ((self.calls // self.warmup_calls) < self.train_max_len):
            self.max_len = (self.calls // self.warmup_calls) + 1
            print('Warmup sampling with len: ', self.max_len)
        else:
            self.max_len = self.train_max_len

        for l in range(1, self.max_len + 1):
            nodes = self.event_sequences_counter.travels(l)
            indices = list(np.random.permutation(np.arange(len(nodes))))
            print(f'Prefix len: {l} nodes: {len(indices)}')
            for i in indices:
                n = nodes[i]
                x1 = n.prefix()
                for x2 in n.next_items(self.pos_len, set_x2):
                    all_seq += 1
                    # add padding
                    c[len(x1)] += 1
                    x1 = self.session_padding(x1)
                    batch_x1.append(x1)
                    batch_x2.append(x2)
                    if len(batch_x1) == self.batch_size:
                        _batch_x1 = np.array(batch_x1, dtype=np.int32)
                        _batch_x2 = np.array(batch_x2, dtype=np.int32)
                        all_x1.append(_batch_x1)
                        all_x2.append(_batch_x2)

                        # negative sampling
                        _batch_x3 = np.zeros((self.batch_size, self.neg_len), dtype=np.long)
                        for _i in range(self.batch_size):
                            p = self.event_sequences_counter.next_negative_items(batch_x1[_i])
                            _batch_x3[_i] = np.random.choice(np.arange(len(p)), self.neg_len, False, p)
                        all_x3.append(_batch_x3)

                        # clean
                        batch_x1 = []
                        batch_x2 = []
                        set_x2 = set()

        print('Sessions len hist.:', [v for k, v in sorted(list(c.items()))], end=' ')
        print(
            f'Skipped pos: {skipped} ({skipped / all_seq:0.02f}%), intersect: {skipped_inter} ({skipped_inter / all_seq:0.02f}%)',
            end=''
        )
        return all_x1, all_x2, all_x3
