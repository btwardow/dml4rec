from rec import dataset
from unittest import TestCase

from rec.dataset.dataset import Dataset
from rec.recommender.sknn import SessionKnnRecommender


class TestSessionKnnRecommenderSalesIntelligenceData(TestCase):
    def setUp(self):
        self.dataset = Dataset.generate_test_data()

    def test_session_knn_recommender(self):
        # given
        r = SessionKnnRecommender()
        given_n = 2
        selected_session = self.dataset.all_sessions_list()[0]
        # when
        r.fit(self.dataset)
        prediction = r.predict_single_session(selected_session, given_n)
        # then
        self.assertIsInstance(prediction, list)
        self.assertTrue(len(prediction))
        for i in prediction:
            self.assertIn(i, self.dataset.items)
        
