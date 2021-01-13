from unittest import TestCase
from unittest.case import expectedFailure

from rec.dataset.dataset import Dataset
from rec.recommender.baseline import PopularityInSessionRecommender
from rec.recommender.baseline import MostPopularRecommender, SessionMostPopularRecommender


class TestMostPopularRecommender(TestCase):
    def setUp(self):
        self.dataset = Dataset('sample', create_indices=False).load_from_file(
            'tests/data/sample_sessions.json', 'tests/data/sample_items.json'
        )

    def test_simple_popularity_recommender(self):
        # given
        r = MostPopularRecommender()
        r.fit(self.dataset)
        n = 3
        # when
        prediction = r.predict_single_session(None, n)
        # then
        self.assertEqual(len(prediction), n)
        self.assertListEqual(prediction, ['2', '5', '7'])

    def test_session_popularity_recommender(self):
        # given
        r = SessionMostPopularRecommender()
        r.fit(self.dataset)
        n = 3
        # when
        prediction = r.predict_single_session(list(list(self.dataset.sessions.values())[0].values())[0], n)
        # then
        self.assertEqual(len(prediction), 3)
        self.assertEqual(prediction[0], '2')
        self.assertSetEqual(set(prediction), {'2', '7', '5'})

    def test_get_config(self):
        # give
        r = SessionMostPopularRecommender(fill_to_top_n=False)
        # when
        conf = r.get_config()
        # then
        self.assertDictEqual(conf, dict(name='SessionMostPopularRecommender', fill_to_top_n=False))

    def test_popularity_in_session_recommender(self):
        # given
        r = PopularityInSessionRecommender()
        r.fit(self.dataset)
        n = 3
        # when
        prediction = r.predict_single_session(list(list(self.dataset.sessions.values())[0].values())[0], n)
        # then
        self.assertEqual(len(prediction), 3)
        self.assertListEqual(prediction, ['5', '7', '2'])

    @expectedFailure
    def test_get_more_that_given_most_common(self):
        # given
        most_common = 10
        top_n = 99
        r = MostPopularRecommender(most_common)
        r.fit(self.dataset)
        # when -> raise error
        r.predict_single_session(list(list(self.dataset.sessions.values())[0].values())[0], top_n)