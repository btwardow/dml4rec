import random
import logging
from math import sqrt

from rec.dataset.dataset import Dataset
from rec.recommender.base import SessionAwareRecommender
from collections import defaultdict, Counter
import tqdm


class SessionKnnRecommender(SessionAwareRecommender):
    def __init__(self, k=100, sample_size=1000, similarity='cosine', sampling='recent'):
        super(SessionKnnRecommender, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.k = k
        self.sample_size = sample_size
        self.similarity = similarity
        assert self.similarity in ['cosine', 'jaccard', 'sorensen_dice']
        self._similarity_func = getattr(SetSimilarities, self.similarity)
        self.sampling = sampling
        assert self.sampling in ['random', 'recent', 'common']
        self._sampling_func = getattr(self, f'_sampling_{self.sampling}')
        # indexes
        self.item_session_map = defaultdict(set)
        self.session_item_map = dict()
        self.session_ts = dict()
        self.items_distribution = dict()

    def fit(self, train_dataset: Dataset, valid_data=None, valid_measures=None):
        assert isinstance(train_dataset, Dataset)
        self.logger.info('Creating recommender indexes...')
        self.create_item_session_maps(train_dataset)
        self.create_items_distribution(train_dataset)
        self.logger.debug("Model preparation completed.")

    def create_items_distribution(self, train_dataset):
        counter = {train_dataset.items_idx_to_id[k]: v for k, v in Counter(train_dataset.items_idx_distrib).items()}
        for item in train_dataset.items.keys():
            if item not in counter:
                counter[item] = 0
        self.items_distribution = counter

    def predict_single_session(self, session, n=10):
        neighbours = self.nearest_neighbours(session)
        result = self.session_items_rank(neighbours)
        result.sort(key=lambda x: (x[1], self.items_distribution[x[0]]), reverse=True)
        return [item[0] for item in result][:min(n, len(result))]

    def nearest_neighbours(self, session):
        sessions = self.possible_neighbours(session)
        items = session.clicked_items_set()
        rank = [(other, self.session_similarity(items, other)) for other in sessions]
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank[0:min(self.k, len(rank))]

    def create_item_session_maps(self, train_dataset: Dataset):
        for session in tqdm.tqdm(train_dataset.all_sessions_list()):
            item_set = session.clicked_items_set()
            self.session_item_map[session.id] = item_set
            self.session_ts[session.id] = session.timestamp_start
            for item in item_set:
                self.item_session_map[item].add(session.id)
        self.item_session_map.default_factory = None

    def session_items_rank(self, neighbours):
        items = self.items_from_sessions(neighbours)
        return [(item, self.item_rank(item, neighbours)) for item in items]

    def item_rank(self, item, neighbours):
        item_sessions = self.item_session_map[item]
        return sum([x[1] for x in neighbours if x[0] in item_sessions])

    def items_from_sessions(self, sessions):
        items = set()
        for session in sessions:
            items |= self.session_item_map[session[0]]
        return items

    def possible_neighbours(self, session):
        """
        to gain some performance sample possible nearest neighbours
        session which have at least one common viewed item
        :param session: Session
        :return: Set
        """
        items = session.clicked_items_set()
        common_sessions = set()
        for item in items:
            if item in self.item_session_map:
                common_sessions |= self.item_session_map[item]
        return self._sampling_func(common_sessions, session)

    def session_similarity(self, this_items, other_session):
        """
        calculate similarity between two sessions
        using method specified in class constructor
        :param this_items: Set
        :param other_session: Session
        :return:
        """
        other_items = self.session_item_map[other_session]
        return self._similarity_func(this_items, other_items)

    def _sampling_random(self, sessions, session):
        sample_size = min(self.sample_size, len(sessions))
        return random.sample(sessions, sample_size)

    def _sampling_recent(self, sessions, session):
        """
        get most recent sessions based on session timestamp
        :param session: Session
        :param sessions: Set
        :return: set of self.sample_size most recent sessions
        """
        rank = [(sid, self.session_ts[sid]) for sid in sessions]
        rank.sort(key=lambda x: x[1], reverse=True)
        result = [x[0] for x in rank]
        sample_size = min(self.sample_size, len(sessions))
        return result[:sample_size]

    def _sampling_common(self, sessions, session):
        """
        get sessions with most common items set
        :param session: Session
        :param sessions: Set
        :return:
        """
        rank = [(ses, len(self.session_item_map[ses] & session.clicked_items_set())) for ses in sessions]
        rank.sort(key=lambda x: x[1])
        result = [x[0] for x in rank]
        sample_size = min(self.sample_size, len(sessions))
        return result[:sample_size]

    def __str__(self):
        return 'SKNN'


class SetSimilarities:
    @staticmethod
    def cosine(first, second):
        len_first = len(first)
        len_second = len(second)
        len_inner_join = len(first & second)
        return len_inner_join / (sqrt(len_first) * sqrt(len_second))

    @staticmethod
    def jaccard(first, second):
        len_first = len(first)
        len_second = len(second)
        len_inner_join = len(first & second)
        return len_inner_join / (len_first + len_second - len_inner_join)

    @staticmethod
    def sorensen_dice(first, second):
        len_inter = len(first & second)
        len_both_sum = len(first) + len(second)
        return len_inter / len_both_sum