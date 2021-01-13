import gzip
import pickle
import logging
import math
import random
try:
    import ujson as json
except ImportError:
    import json as json
from collections import defaultdict, Counter
from itertools import chain

import numpy as np
from scipy.sparse import diags
from tqdm import tqdm

from rec.model import Session, Item, EVENT_ITEM, EVENT_SESSION_ID, EVENT_USER_HASH, EVENT_USER_ID, \
    EVENT_TYPE, ITEM_ID, PRESENTED_ITEMS, TIMESTAMP, EVENT_CONTEXT
from rec.utils import current_milli_time

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, name='', create_indices=True):
        self.create_indices = create_indices
        self.name = name
        self.sessions = defaultdict(dict)
        self.items = {}

    def _create_indexes(self):
        self.items_id_to_idx = {v: idx for idx, v in enumerate(self.items.keys())}
        self.items_idx_to_id = {v: k for k, v in list(self.items_id_to_idx.items())}
        self.sessions_id_to_idx = {
            v: idx
            for idx, v in enumerate(
                [(uId, sId) for uId, uSessions in list(self.sessions.items()) for sId, s in list(uSessions.items())]
            )
        }
        self.sessions_idx_to_id = {v: k for k, v in list(self.sessions_id_to_idx.items())}
        self.items_idx_distrib = [
            self.items_id_to_idx[i] for us in list(self.sessions.values()) for s in list(us.values())
            for i in s.clicked_items_list() if i is not None
        ]

    def load_from_file(
        self, sessions_data_file, items_data_file, unique_items_in_session=False, load_only_clicked_items=False
    ):
        self._load_sessions(sessions_data_file, unique_items_in_session)
        self._load_items(items_data_file, load_only_clicked_items)
        if self.create_indices:
            self._create_indexes()
        return self

    def load_with_no_context(self, sessions_data_file, unique_items_in_session=False):
        '''
        Loads dataset created by interactions with items. 
        No additional (context) attributes are loaded.

        This is adequate for 2D recommenders - like pure CF (kNN, MF, GRU4Rec,...)
        '''
        start_time = current_milli_time()
        self._load_sessions_no_context(sessions_data_file, unique_items_in_session)
        # items are only ids
        self.items = {
            i: 1
            for i in
            set([i for sessions in self.sessions.values() for s in sessions.values() for i in s.clicked_items_set()])
        }
        self._create_indexes()
        logger.info('Load time: {t}(s).'.format(t=((current_milli_time() - start_time) / 1000)))
        logger.info(str(self))
        return self

    def _load_sessions(self, sessions_data_file, unique_items_in_session):
        logger.info(f'Loading sessions data from file: {sessions_data_file}')
        self.omitted_events = 0
        self.loaded_events = 0
        logger.info('Loading session events...')
        with Dataset._open_file(sessions_data_file) as f:
            i = 0
            for line in tqdm(f, desc="Loading session events"):
                i += 1
                try:
                    session_event = json.loads(line)
                    if EVENT_ITEM in session_event and session_event[EVENT_ITEM] is None:
                        del session_event[EVENT_ITEM]
                    session_id = session_event[EVENT_SESSION_ID]
                    user_hash = session_event[EVENT_USER_HASH] if EVENT_USER_HASH in session_event else session_id
                    user_id = session_event[EVENT_USER_ID] if EVENT_USER_ID in session_event else None
                    if session_id not in self.sessions[user_hash]:
                        self.sessions[user_hash][session_id] = Session(session_id, user_hash, user_id)
                    ds_s = self.sessions[user_hash][session_id]
                    if unique_items_in_session and EVENT_ITEM in session_event:
                        item_id = session_event[EVENT_ITEM]
                        event_type = session_event[EVENT_TYPE]
                        if ds_s.is_in_session(event_type, item_id):
                            self.omitted_events += 1
                        else:
                            ds_s.create_and_add_event(session_event)
                    else:
                        ds_s.create_and_add_event(session_event)

                    self.loaded_events += 1
                except ValueError:
                    import sys
                    self.omitted_events += 1
                    print(f"Unexpected error while parsing line {i}: ---\n{line}\n---\n", sys.exc_info()[0])

            if unique_items_in_session:
                logger.info(
                    "{} events omitted due to non unique event_type,item_id or parsing errors".format(
                        self.omitted_events
                    )
                )
        self.sessions.default_factory = None  # don't extend by default
        logger.info("{} events processed".format(self.loaded_events))

    def _load_sessions_no_context(self, sessions_data_file, unique_items_in_session=False):
        logger.info(f'Loading sessions clicks data from file: {sessions_data_file}')
        i = 0
        logger.info('Loading session VIEW events')
        with Dataset._open_file(sessions_data_file) as f:
            for line in tqdm(f, desc="Loading session events"):
                # only items views
                try:
                    session_event = json.loads(line)
                except ValueError:
                    e = sys.exc_info()[0]
                    logger.error(f'Error: {e}\n\nLine:\n--\n{line}\n--\n')
                if EVENT_ITEM in session_event and session_event[EVENT_ITEM] is not None:
                    session_id = session_event[EVENT_SESSION_ID]
                    user_hash = session_event[EVENT_USER_HASH] if EVENT_USER_HASH in session_event else session_id
                    user_id = session_event[EVENT_USER_ID] if EVENT_USER_ID in session_event else None
                    item_id = session_event[EVENT_ITEM]
                    if EVENT_CONTEXT in session_event:
                        del session_event[EVENT_CONTEXT]
                    if session_id not in self.sessions[user_hash]:
                        self.sessions[user_hash][session_id] = Session(session_id, user_hash, user_id)
                    ds_s = self.sessions[user_hash][session_id]
                    if unique_items_in_session:
                        if item_id not in ds_s.clicked_items_set():
                            ds_s.create_and_add_event(session_event)
                    else:
                        ds_s.create_and_add_event(session_event)
                    i += 1
                    if i % 1000000 == 0:
                        logger.info('{n} session events loaded.'.format(n=i))
        self.sessions.default_factory = None  # don't extend by default
        logger.info('{n} session events loaded.'.format(n=i))
        logger.info('{n} unique user\'s hashes'.format(n=self.users_num()))

    def _load_items(self, items_data_file, load_only_event_items=False):
        logger.info(f'Loading items data from file: {items_data_file}')
        i = 0
        logger.info('Loading items...')
        if load_only_event_items:
            logger.info("Only clicked items will be loaded.")
            events_item_ids = {
                clicked_id
                for us in self.sessions.values() for s in us.values() for clicked_id in s.clicked_items_list()
            }
            logger.info("{} events items.".format(len(events_item_ids)))

        with Dataset._open_file(items_data_file) as f:
            for line in tqdm(f, desc="Loading items"):
                item = json.loads(line)
                id = str(item[ITEM_ID])
                if load_only_event_items and id not in events_item_ids:
                    continue
                self.items[id] = item
                i += 1
                if i % 100000 == 0:
                    logger.info(f'{i} items loaded.')
            logger.info(f'{i} items loaded.')

    def split_left_n_sessions(self, n=2):
        train = Dataset()
        test = Dataset()
        train.items = self.items
        test.items = self.items

        for u, u_sessions in self.sessions.items():
            if len(u_sessions) <= n:
                train.sessions[u] = u_sessions
            else:
                se_by_timestamp = sorted(list(u_sessions.values()), key=lambda s: s.timestamp_start)
                train.sessions[u] = {s.id: s for s in se_by_timestamp[:-n]}
                test.sessions[u] = {s.id: s for s in se_by_timestamp[-n:]}

        train._create_indexes()
        test._create_indexes()
        return train, test

    def split_sessions(self, train_size, test_size):
        train = Dataset()
        test = Dataset()
        train.items = self.items
        test.items = self.items

        n = 1
        for u, u_sessions in self.sessions.items():
            if n <= train_size:
                train.sessions[u] = u_sessions
            elif n <= train_size + test_size:
                test.sessions[u] = u_sessions
            else:
                break

            n += len(u_sessions)

        train._create_indexes()
        test._create_indexes()
        return train, test

    def most_popular_items_ids(self, n):
        # most popular items
        items_clicks = Counter()
        for item_id in [
            clicked_id for us in self.sessions.values() for s in us.values() for clicked_id in s.clicked_items_list()
        ]:
            items_clicks[item_id] += 1
        return np.array([i for i, c in items_clicks.most_common(min(n, len(items_clicks)))], dtype=str)

    def users(self):
        return list(self.sessions.keys())

    def users_num(self):
        return len(self.users())

    def items_num(self):
        return len(self.items)

    def sessions_num(self):
        return sum([len(sessions) for sessions in self.sessions.values()])

    def all_sessions_list(self):
        # list all sessions with appropriate order
        all_sessions = []
        for us in list(self.sessions.values()):
            for s in list(us.values()):
                all_sessions.append(s)
        return all_sessions

    def events_num(self, event_types=None):
        """
        Counts events in dataset.

        :param event_types: list of event type to count
        :type event_types: list[string]
        :return:
        """
        if event_types is None:
            return sum([s.events_num() for user_sessions in self.sessions.values() for s in user_sessions.values()])
        else:
            i = 0
            for user_sessions in self.sessions.values():
                for s in user_sessions.values():
                    for e in s.events:
                        if e.event_type in event_types:
                            i += 1
            return i

    def create_sample_by_users(self, users_num=1000, all_items=False):
        """
        Create random sampled dataset given user number.

        :param users_num: number of users in sample
        :param all_items: contains all items or only ones in sampled sessions
        :return:
        """
        sampled_users = random.sample(list(self.sessions), users_num)

        sample = Dataset()
        sample.sessions = {u: self.sessions[u] for u in sampled_users}
        sample.items = dict()

        if all_items:
            sample.items = self.items
        else:
            sample_items = set()
            for session_items in [
                s.clicked_items_set() | s.all_presented_items() for sessions in sample.sessions.values()
                for s in sessions.values()
            ]:
                sample_items |= session_items
            for item_id in sample_items:
                sample.items[item_id] = self.items[item_id]

        sample._create_indexes()
        return sample

    def create_sample_by_sessions(self, sessions_num=1000000, all_items=False):
        """
        Create random sampled dataset given sessions.

        :param sessions_num: number of sessions in sample
        :param all_items: contains all items or only ones in sampled sessions
        :return:
        """
        all_sessions = chain.from_iterable([s.values() for s in self.sessions.values()])
        sampled_sessions = random.sample(all_sessions, sessions_num)

        sample = Dataset()
        sample.sessions = {0: {s.id: s for s in sampled_sessions}}
        sample.items = dict()

        if all_items:
            sample.items = self.items
        else:
            sample_items = set()
            for session_items in [
                s.clicked_items_set() | s.all_presented_items() for sessions in sample.sessions.values()
                for s in sessions.values()
            ]:
                sample_items |= session_items
            for item_id in sample_items:
                sample.items[item_id] = self.items[item_id]

        sample._create_indexes()
        return sample

    def write_to_file(self, file_name='dataset.pkl.gz'):
        logger.info(f'Saving dataset to file: {file_name}')
        with Dataset._open_file(file_name, mode='wb') as f:
            pickle.dump(self, f, protocol=2)

    def items_one_hot_encoding_item_id(self):
        return list(self.items.keys()), diags(np.ones(self.items_num(), dtype=np.int), 0, format='csr')

    @staticmethod
    def _open_file(file_name, mode='r'):
        return gzip.open(file_name, mode) if '.gz' in file_name else open(file_name, mode)

    @staticmethod
    def read_from_file(file_name='dataset.pkl.gz'):
        logger.info('Reading dataset from file: {file_name}'.format(file_name=file_name))
        with Dataset._open_file(file_name, mode='rb') as f:
            dataset = pickle.load(f)
        return dataset

    @staticmethod
    def generate_test_data_with_density(sessions_num=100, items_num=100, density=0.01, name='generated-test-dataset'):
        logger.info(f"Generating test dataset with {sessions_num} sessions, {items_num} items and {density} density.")
        dataset = Dataset(name)
        dataset.items = {f'item-{i}': Item(f'item-{i}') for i in range(items_num)}
        for i in range(sessions_num):
            s = Session(i, i, i)
            clicked_id = random.randint(0, items_num - 1)
            s.create_and_add_event({TIMESTAMP: i, EVENT_TYPE: 'VIEW', EVENT_ITEM: f'item-{clicked_id}'})
            dataset.sessions[i][i] = s
        events_num = int(density * sessions_num * items_num)
        for i in range(sessions_num, events_num):
            session_id = random.randint(0, sessions_num - 1)
            items_id = random.randint(0, items_num - 1)
            dataset.sessions[session_id][session_id].create_and_add_event(
                {
                    TIMESTAMP: i,
                    EVENT_TYPE: 'VIEW',
                    EVENT_ITEM: f'item-{items_id}'
                }
            )
        dataset._create_indexes()
        return dataset

    @staticmethod
    def generate_test_data(
        sessions_num=100, items_num=100, events_in_session=10, sessions_by_user=1, name='generated-test-dataset'
    ):
        logger.info(
            "Generating test dataset with {sessions_num} sessions, {items_num} items and {events_in_session} events "
            "in session.".format(sessions_num=sessions_num, items_num=items_num, events_in_session=events_in_session)
        )
        dataset = Dataset(name)
        item_id_offset = random.randint(0, 100)
        dataset.items = {f'item-{i + item_id_offset}': Item(f'item-{i + item_id_offset}') for i in range(items_num)}
        for i in range(sessions_num):
            s = Session(i, i, i)
            for e in range(events_in_session):
                ts = 1 + i * events_in_session + e
                item_id = random.randint(0, items_num - 1) + item_id_offset
                s.create_and_add_event(dict(timestamp=ts, eventType='VIEW', clickedItem=f'item-{item_id}'))
                dataset.sessions[math.floor(i / sessions_by_user)][i] = s
        dataset._create_indexes()
        return dataset

    def __str__(self):
        density = round(100. * float(self.events_num(['VIEW'])) / float(self.items_num() * self.sessions_num()), 4)
        return 'Dataset: {}, sessions: {}, items: {}, events: {} (density: {}%){}'.format(
            self.name, self.sessions_num(), self.items_num(), self.events_num(), density,
            '{} events omitted due to non unique type/item in session'.format(self.omitted_events)
            if hasattr(self, 'omitted_events') and self.omitted_events > 0 else ''
        )

    def __repr__(self):
        return super(Dataset, self).__repr__()
