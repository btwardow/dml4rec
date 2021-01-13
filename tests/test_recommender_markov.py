from unittest import TestCase

from rec.dataset.dataset import Dataset
from rec.recommender.markov import MarkovRecommender


class TestLOrderMarkovRecommender(TestCase):
    def test_first_order(self):
        # given
        ds = Dataset.generate_test_data()
        r = MarkovRecommender()
        r.fit(ds)
        n = 3
        s = ds.all_sessions_list()[0]
        # when
        prediction = r.predict_single_session(s, n)
        # then
        self.assertEqual(len(prediction), n)