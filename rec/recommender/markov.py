from collections import Counter
from rec.dataset.dataset import Dataset
from rec.recommender.base import SessionAwareRecommender


class MarkovRecommender(SessionAwareRecommender):
    def __init__(self, L=1):
        super().__init__()
        self.L = L
        assert self.L >= 1
        self.history = dict()

    def predict_single_session(self, session, n=10):
        ids = session.clicked_items_list()
        if len(ids) < self.L:
            return []

        response = []
        for i in range(len(ids) - self.L, 0, -1):
            key = tuple(ids[i:i + self.L])
            if key in self.history:
                response.extend([k for k, v in self.history[key].most_common(n)])
                if len(response) >= n:
                    break
        return response

    def fit(self, train_dataset: Dataset, valid_data=None, valid_measures=None):
        assert isinstance(train_dataset, Dataset)
        for s in train_dataset.all_sessions_list():
            ids = s.clicked_items_list()
            if len(ids) > self.L + 1:
                for i in range(len(ids) - self.L):
                    key = tuple(ids[i:i + self.L])
                    next_id = ids[i + self.L]
                    if key not in self.history:
                        self.history[key] = Counter()
                    self.history[key][next_id] += 1
        assert len(self.history) > 0, 'There is no history events for {}-order Markov!'.format(self.L)

    def __str__(self):
        return 'MARKOV-{}'.format(self.L)
