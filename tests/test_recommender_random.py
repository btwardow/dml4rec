from unittest import TestCase

from rec.dataset.dataset import Dataset
from rec.recommender.baseline import RandomRecommender


class TestRandomRecommender(TestCase):
    def test_predict(self):
        dataset = Dataset.generate_test_data(100, 100, 10, 4)
        train, test = dataset.split_left_n_sessions(1)
        recommender = RandomRecommender()
        recommender.fit(train)

        n = 10
        test_sessions = list(test.sessions.values())
        test_session1 = test_sessions[0]
        test_session2 = test_sessions[1]
        predict1 = recommender.predict_single_session(test_session1, n)
        predict2 = recommender.predict_single_session(test_session2, n)

        self.assertEqual(len(predict1), 10)
        self.assertEqual(len(predict2), 10)
        self.assertNotEqual(
            set(predict1), set(predict2), 'Random recommender returned the same result. Highly improbable.'
        )
