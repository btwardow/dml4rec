import csv
import os
import json
import gzip
import math
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from rec.dataset.dataset import Dataset
import rec.model as m

directory = 'data/retailrocket/'
directory_input = directory + 'raw/'
input_path_events = directory_input + 'sorted_events.csv'
input_path_items = [
    directory_input + 'sorted_item_properties_part1.csv', directory_input + 'sorted_item_properties_part2.csv'
]
input_category_tree = directory_input + 'category_tree.csv'

items_jsonl_path = directory + 'items'
events_jsonl_path = directory + 'sessions'
delimiter = ','

datasets = 5
datasets_dir_prefix = 'data/dataset/RR'
datasets_dirs = []
timestamp_first_event = 1430622004384
timestamp_last_event = 1442545187788


class RetailRocket:
    def __init__(self):
        self.items = dict()
        self.category_tree = dict()
        self.users_sessions = dict()
        self.next_session_id = 0
        self.items_in_datasets = dict()
        self.items_all_properties = set()
        self.items_mutable_properties = set()
        for i in range(datasets):
            self.items_in_datasets[i] = set()

    def prepare_items(self):
        self._read_category_tree()
        for input_path in input_path_items:
            self._add_items_properties(input_path)
        self._find_immutable_properties()

    def generate_events_file(self):
        rows = self._prepare_events()
        data = self._filter_events(rows)
        self._save_events_to_file(data)

    def save_items_to_file(self):
        print('Saving all items...')
        with gzip.open(f'{datasets_dir_prefix}/items.jsonl.gz', 'wt') as f:
            for item in self.items.values():
                f.write(item.transform_into_jsonl_format())
                f.write('\n')

        print('Saving splited items...')
        for i in range(datasets):
            items_set = self.items_in_datasets[i]
            with gzip.open(f'{datasets_dir_prefix}-{i+1}/items.jsonl.gz', 'wt') as f:
                for item_id in items_set:
                    item_jsonl = self.items[item_id].transform_into_jsonl_format()
                    f.write(item_jsonl)
                    f.write('\n')

    def _prepare_events(self):
        rows = []
        with open(input_path_events) as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            next(csv_reader, None)

            for line in csv_reader:
                event_jsonl = self._prepare_event_in_jsonl(line)
                if event_jsonl is not None:
                    ev_dict = json.loads(event_jsonl)
                    file_no = self.calculate_file_no(ev_dict['timestamp'])
                    row = [ev_dict['sessionId'], ev_dict['clickedItem'], ev_dict['timestamp'], event_jsonl, file_no]
                    rows.append(row)
        return rows

    def _filter_events(self, rows):
        columns = ['session_id', 'item_id', 'timestamp', 'event_jsonl', 'file_no']
        return self._filter_data(pd.DataFrame(rows, columns=columns))

    def _save_events_to_file(self, data):
        for i in range(datasets):
            d = f'{datasets_dir_prefix}-{i+1}'
            os.makedirs(d, exist_ok=True)
            datasets_dirs.append(d)

        os.makedirs(datasets_dir_prefix, exist_ok=True)
        datasets_dirs.append(datasets_dir_prefix)

        print('Saving all events dataset...')
        with gzip.open(f'{datasets_dir_prefix}/sessions.jsonl.gz', 'wt') as f:
            for _, row in data.iterrows():
                f.write(row['event_jsonl'] + '\n')

        print('Saving splited events datasets...')
        outputs = [gzip.open(f'{datasets_dir_prefix}-{i+1}/sessions.jsonl.gz', 'wt') for i in range(datasets)]
        for _, row in data.iterrows():
            if row['file_no'] < datasets:
                if row['item_id'] in self.items:
                    outputs[row['file_no']].write(row['event_jsonl'] + '\n')
                    self.items_in_datasets[row['file_no']].add(row['item_id'])
                else:
                    print(f'Item id: {row.item_id} is clicked but not in items dataset')
        map(lambda f: f.close(), outputs)

    def _add_items_properties(self, path):
        with open(path) as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            next(csv_reader, None)
            for line in csv_reader:
                self._add_item_property(line)

    def _add_item_property(self, line):
        assert len(line) == 4
        timestamp = int(line[0])
        item_id = line[1]
        property_name = line[2]
        value = line[3].strip().split(' ')
        if len(value) == 1:  # single value, no array is neccessary
            value = value[0]

        if item_id not in self.items.keys():
            self.items[item_id] = Item(item_id)

        self.items[item_id].add_property(property_name, timestamp, value)

        if property_name == "categoryid" and value in self.category_tree:
            category_path_ids = self._read_path_to_root(value)
            self.items[item_id].add_property("category_path_ids", timestamp, category_path_ids)

    def _read_path_to_root(self, leaf):
        current_node = leaf
        result = deque([current_node])

        while self.category_tree[current_node] != current_node:
            current_node = self.category_tree[current_node]
            result.appendleft(current_node)

        return result

    def _read_category_tree(self):
        with open(input_category_tree) as input_file:
            csv_reader = csv.reader(input_file, delimiter=delimiter)
            next(csv_reader, None)

            for line in csv_reader:
                if line[1] != "":
                    self.category_tree[int(line[0])] = int(line[1])
                else:  # when line describes root category
                    self.category_tree[int(line[0])] = int(line[0])

    def _find_immutable_properties(self):
        for item_id, item in self.items.items():
            for k, v in item.properties.items():  # k = property name, v = list of tuples (timestamp, value)
                self.items_all_properties.add(k)
                if len(v) > 1:  # if for all timestamps there is the same value => not muttable
                    for el in v:
                        if el[1] != v[0][1]:
                            self.items_mutable_properties.add(k)
                            break

        print(
            f'All items properties number: {len(self.items_all_properties)}, mutable: {len(self.items_mutable_properties)}'
        )
        for item_id, item in self.items.items():
            for k, v in item.properties.items():
                if k in self.items_mutable_properties:
                    item.mutable_properties[k] = v
                else:
                    item.immutable_properties[k] = v[0][1]  # take first value

    @staticmethod
    def normalize_context(r):
        d = dict()
        attribs = []
        for k, values in r.items():
            if not isinstance(values, list):
                values = [values]
            for v in values:
                if v.startswith('n'):  # number
                    f = float(v[1:])
                    if math.isinf(f):
                        print(f'Infinity! Bad value for {k} : {v}. Skipping...')
                        continue
                    d[k] = f
                else:
                    attribs.append(f'{k}|{v}')
        d['properties'] = attribs
        return d

    def _prepare_event_in_jsonl(self, line):
        def converter(o):
            if isinstance(o, datetime):
                return o.__str__()

        timestamp = int(line[0])
        user_id = int(line[1])
        item_id = line[3]

        if user_id not in self.users_sessions:
            self.users_sessions[user_id] = [timestamp, self.next_session_id]
            self.next_session_id += 1
        else:
            if timestamp - self.users_sessions[user_id][0] > 30 * 60 * 1000:  # 30 min * 60s * 1000ms
                self.users_sessions[user_id] = [timestamp, self.next_session_id]
                self.next_session_id += 1
            else:
                self.users_sessions[user_id][0] = timestamp  # update last activity in session

        if item_id in self.items:
            data = {
                m.TIMESTAMP: timestamp,
                m.EVENT_USER_ID: user_id,
                m.EVENT_TYPE: line[2],
                m.EVENT_ITEM: item_id,
                m.EVENT_SESSION_ID: self.users_sessions[user_id][1]
            }
            context = self._prepare_context(item_id, timestamp)
            if len(context) > 0:
                data[m.EVENT_CONTEXT] = RetailRocket.normalize_context(context)
            return json.dumps(data, default=converter, separators=(',', ':'))

    def _prepare_context(self, item_id, timestamp):
        context = {}
        for property, values in self.items[item_id].mutable_properties.items():
            ts, val = 0, 0
            for time, value in values:
                if timestamp >= time > ts:
                    ts = time
                    val = value
            if ts > 0:
                context[property] = val
        return context

    @staticmethod
    def _filter_data(data):  # based on 130L session-rec/preprocessing/preprocess_retailrocket.py

        session_lengths = data.groupby('session_id').size()
        data = data[np.in1d(data.session_id, session_lengths[session_lengths > 1].index)]

        item_supports = data.groupby('item_id').size()
        data = data[np.in1d(data.item_id, item_supports[item_supports >= 5].index)]

        session_lengths = data.groupby('session_id').size()
        data = data[np.in1d(data.session_id, session_lengths[session_lengths >= 2].index)]

        return data

    @staticmethod
    def calculate_file_no(ts):
        return int((ts - timestamp_first_event) / (1000 * 60 * 60 * 24 * 27))  # 1000ms * 60s * 60min * 24h * 27d


class Item:
    def __init__(self, id):
        self.id = str(id)
        self.properties = dict()  # all properties
        self.immutable_properties = dict()  # add to items.jsonl
        self.mutable_properties = dict()  # add to sessions.jsonl in context field

    def add_property(self, property, timestamp, value):
        if property not in self.properties.keys():
            self.properties[property] = list()
        self.properties[property].append((timestamp, value))

    def transform_into_jsonl_format(self):
        dt = {m.ITEM_ID: self.id}
        dt.update(RetailRocket.normalize_context(self.immutable_properties))
        return json.dumps(dt, separators=(',', ':'))


if __name__ == '__main__':
    items = RetailRocket()
    items.prepare_items()
    items.generate_events_file()
    items.save_items_to_file()

    for ds_dir in datasets_dirs:
        ds_name = ds_dir.split('/')[-1]
        ds = Dataset(ds_name).load_from_file(f'{ds_dir}/sessions.jsonl.gz', f'{ds_dir}/items.jsonl.gz')
        ds.write_to_file(f'{ds_dir}/dataset.pkl.gz')
        ds = Dataset(ds_name).load_with_no_context(f'{ds_dir}/sessions.jsonl.gz')
        ds.write_to_file(f'{ds_dir}/dataset_no_context.pkl.gz')
