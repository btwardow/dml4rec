import operator
import random
from collections import Counter
from rec.dataset.dataset import Dataset
from rec.recommender.base import SessionAwareRecommender


class RandomRecommender(SessionAwareRecommender):
    """
    Random recommender.
    """
    def predict_single_session(self, session, n=10):
        return random.sample(self.items, n)

    def fit(self, train_dataset, valid_data=None, valid_measures=None):
        self.train_dataset = train_dataset
        self.items = list(train_dataset.items.keys())

    def __str__(self):
        return 'RND'


class MostPopularRecommender(SessionAwareRecommender):
    """
    Simple Popularity Recommender.
    Always return top-n most popular items in training dataset.
    """
    def __init__(self, most_common=100):
        self.most_common = most_common
        self.popular_items = None
        self.items_clicks = Counter()
        super(MostPopularRecommender, self).__init__()

    def predict_single_session(self, session, n=10):
        if n > self.most_common:
            raise RuntimeError("Recommender is prepared to return max. {} most popular items.".format(self.most_common))
        return self.popular_items[:n]

    def fit(self, train_dataset, valid_data=None, valid_measures=None):
        """

        Args:
            train_dataset (object): 
        """
        assert isinstance(train_dataset, Dataset)
        for i in [
            i for sessions in train_dataset.sessions.values() for s in sessions.values() for i in s.clicked_items_list()
        ]:
            self.items_clicks[i] += 1
        self.popular_items = [i for i, c in self.items_clicks.most_common(self.most_common)]

    def __str__(self):
        return 'POP'


class SessionMostPopularRecommender(MostPopularRecommender):
    """
    Session Popularity Recommender.
    Returns top-N most popular items in training dataset which was already presented in the session.
    If presented items is less than required N, recommender can fill the missing items using most popular
    items in the training dataset.
    """
    def __init__(self, fill_to_top_n=True):
        self.fill_to_top_n = fill_to_top_n
        super(SessionMostPopularRecommender, self).__init__()

    def predict_single_session(self, session, n=10):
        items_pop = {i: self.items_clicks[i] for i in session.all_presented_items()}
        items_pop = list(reversed(sorted(list(items_pop.items()), key=operator.itemgetter(1))))
        result = [k for k, v in items_pop[:n]]
        # if there is not enought items - add the most popular one's
        missing_items = n - len(result)
        if self.fill_to_top_n and missing_items > 0:
            result += self.popular_items[:missing_items]
        return result


class PopularityInSessionRecommender(MostPopularRecommender):
    def __init__(self, fill_to_top_n=True):
        self.fill_to_top_n = fill_to_top_n
        super(PopularityInSessionRecommender, self).__init__()

    def predict_single_session(self, session, n=10):
        s_items_count = Counter(session.clicked_items_list()).most_common(n)
        result = [i for i, c in s_items_count]
        missing_items = n - len(result)
        # if there is not enough items - add the most popular one's
        if self.fill_to_top_n and missing_items > 0:
            result += self.popular_items[:missing_items]
        return result

    def __str__(self):
        return 'SPOP'
