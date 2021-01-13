import logging
import random
from collections import defaultdict
from math import sqrt, log10

import numpy as np
import tqdm

from rec.dataset.dataset import Dataset
from rec.recommender.base import SessionAwareRecommender
from rec.recommender.sknn import SetSimilarities


class VMSessionKnnRecommender(SessionAwareRecommender):
    def __init__(
        self,
        k=100,
        sample_size=1000,
        similarity='cosine',
        sampling='recent',
        weighting='div',
        weighting_score='div',
        idf_weighting=10,
        normalize=True
    ):  
        super(VMSessionKnnRecommender, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.k = k
        self.sample_size = sample_size
        self.similarity = similarity
        assert self.similarity in ['cosine', 'jaccard', 'sorensen_dice']
        self._similarity_func = getattr(SetSimilarities, self.similarity)
        self.sampling = sampling
        self._sampling_func = getattr(self, f'_sampling_{self.sampling}')

        self.weighting = weighting
        assert self.weighting in ['linear', 'same', 'div', 'log', 'quadratic']
        self._weighting_func = getattr(SetWeightingFunction, self.weighting)

        self.weighting_score = weighting_score
        assert self.weighting_score in ['linear', 'same', 'div', 'log', 'quadratic']
        self._weighting_score_func = getattr(SetWeightingScoreFunction, self.weighting_score)

        self.idf_weighting = idf_weighting
        self.normalize = normalize

        self.item_session_map = defaultdict(set)
        self.session_item_map = dict()
        self.session_ts = dict()

    def fit(self, train_dataset: Dataset, valid_data=None, valid_measures=None):
        assert isinstance(train_dataset, Dataset)
        self.logger.info('Creating recommender indexes...')
        self.create_item_session_maps(train_dataset)
        self.logger.debug("Model preparation completed.")

    def predict_single_session(self, session, n=10, return_scores=False):
        neighbours = self.nearest_neighbours(session)
        result = self.session_items_rank(neighbours, session)
        result.sort(key=lambda x: x[1], reverse=True)
        r = [item[0] for item in result][:min(n, len(result))]
        if return_scores:
            return r, [item[1] for item in result][:min(n, len(result))]
        return r

    def nearest_neighbours(self, session):
        sessions = self.possible_neighbours(session)
        items = session.clicked_items_list()
        rank = self.calc_similarity(items, sessions)
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank[0:min(self.k, len(rank))]

    def session_items_rank(self, neighbours, session):
        items = session.clicked_items_list()
        scores = dict()

        for other in neighbours:
            other_items = self.session_item_map[other[0]]
            step = 1
            for item in reversed(items):
                if item in other_items:
                    decay = self._weighting_score_func(step)
                    break
                step += 1

            for item in other_items:
                old_score = scores.get(item)
                new_score = other[1]
                if self.idf_weighting:
                    new_score = new_score + (new_score * self.idf[item] * self.idf_weighting)

                new_score = new_score * decay

                if not old_score is None:
                    new_score = old_score + new_score

                scores.update({item: new_score})

        rank = list()
        for k, v in scores.items():
            rank.append((k, v))
        return rank

    def create_item_session_maps(self, train_dataset: Dataset):
        for session in tqdm.tqdm(train_dataset.all_sessions_list()):
            item_set = session.clicked_items_set()
            self.session_item_map[session.id] = item_set
            self.session_ts[session.id] = session.timestamp_start
            for item in item_set:
                self.item_session_map[item].add(session.id)
        self.item_session_map.default_factory = None

        if self.idf_weighting:
            all_session_num = len(self.session_item_map)
            self.idf = {
                i: np.log(all_session_num / len(self.item_session_map[i]))
                for i in self.item_session_map.keys()
            }

    def possible_neighbours(self, session):
        items = session.clicked_items_set()
        common_sessions = set()
        for item in items:
            if item in self.item_session_map:
                common_sessions |= self.item_session_map[item]
        return self._sampling_func(common_sessions, session)

    def calc_similarity(self, session_items, possible_neighbours):
        pos_map = dict()
        length = len(session_items)

        count = 1
        for item in session_items:
            if self.weighting is not None:
                pos_map[item] = self._weighting_func(count, length)
                count += 1
            else:
                pos_map[item] = 1  # it's equal 'same' decay function

        items = set(session_items)
        neighbours = []
        count = 0
        for other in possible_neighbours:
            count += 1
            other_items = self.session_item_map[other]
            similarity = self.vec(items, other_items, pos_map)
            neighbours.append([other, similarity])

        return neighbours

    def vec(self, first, second, map):
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]

        result = sum / len(map)
        return result

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
        return 'VSKNN'



class SetWeightingFunction:  # 656-669L VSKNN session-rec
    @staticmethod
    def linear(i, length):
        return 1 - (0.1 * (length - i)) if i <= 10 else 0

    @staticmethod
    def same(i, length):
        return 1

    @staticmethod
    def div(i, length):
        return i / length

    @staticmethod
    def log(i, length):
        return 1 / (log10((length - i) + 1.7))

    @staticmethod
    def quadratic(i, length):
        return (i / length)**2


class SetWeightingScoreFunction:  # 641-654L VSKNN session-rec
    @staticmethod
    def linear(i):
        return 1 - (0.1 * i) if i <= 100 else 0

    @staticmethod
    def same(i):
        return 1

    @staticmethod
    def div(i):
        return 1 / i

    @staticmethod
    def log(i):
        return 1 / (log10(i + 1.7))

    @staticmethod
    def quadratic(i):
        return 1 / (i * i)