from rec.utils import seed_everything
from unittest import TestCase
from rec.dataset.dataset import Dataset
from rec.recommender.baseline import MostPopularRecommender, SessionMostPopularRecommender

seed_everything()


class TestMostPopularRecommenderSalesIntelligenceData(TestCase):
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
        selected_session = self.dataset.all_sessions_list()[0]

        # when
        prediction = r.predict_single_session(selected_session, n)

        # then
        self.assertEqual(len(prediction), n)
        self.assertListEqual(prediction[:1], ['2'])
