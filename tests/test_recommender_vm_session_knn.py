from unittest import TestCase

import numpy as np

from rec.dataset.dataset import Dataset
from rec.recommender.vsknn import VMSessionKnnRecommender
from tests.dataset_toolkit import create_session, create_dataset


def prepare_dataset():
    np.random.seed(123)

    def next_item():
        return np.random.randint(10)

    sessions = [create_session("s" + str(i), 1, 2) for i in range(400)]
    sessions.extend([create_session("s" + str(i), 1, 5) for i in range(300)])
    sessions.extend([create_session("s" + str(i), 2, 3) for i in range(200)])
    sessions.extend([create_session("s" + str(i), 8, 3) for i in range(100)])
    sessions.extend([create_session("s" + str(i), next_item(), next_item()) for i in range(50)])
    np.random.shuffle(sessions)

    dataset = create_dataset(*sessions)
    return dataset


class TestVMSessionKnnRecommender(TestCase):
    def test_predict(self):
        # given
        dataset = Dataset.generate_test_data(sessions_num=2)
        r = VMSessionKnnRecommender()
        r.fit(dataset)

        # when
        session = list(list(dataset.sessions.values())[0].values())[0]

        # then
        self.assertEqual(len(r.predict_single_session(session, 1)), 1)
        self.assertEqual(len(r.predict_single_session(session, 2)), 2)
        self.assertEqual(len(r.predict_single_session(session, 10)), 10)

    def test_predicting_simple_bahavior(self):
        # given
        dataset = prepare_dataset()
        r = VMSessionKnnRecommender()
        r.fit(dataset)

        # when
        predictions = r.predict_single_session(create_session("test", 1), n=3)

        # then
        self.assertEqual(3, len(predictions))
        self.assertEqual(5, predictions[0])
        self.assertEqual(1, predictions[1])
        self.assertEqual(6, predictions[2])  # if seed = 123

        self.assertNotEqual(2, predictions[2])
        self.assertNotEqual(5, predictions[2])

    def test_prediction_weighting(self):
        # given
        dataset = prepare_dataset()
        weighting = ['linear', 'same', 'div', 'log', 'quadratic']
        recommenders = [VMSessionKnnRecommender(weighting=w) for w in weighting]
        [r.fit(dataset) for r in recommenders]

        # when
        predictions = [r.predict_single_session(create_session("test", 1, 2, 8), n=3) for r in recommenders]

        # then
        self.assertEqual(predictions[0][0], 2)
        self.assertEqual(predictions[1][0], 2)
        self.assertEqual(predictions[2][0], 8)
        self.assertEqual(predictions[3][0], 8)
        self.assertEqual(predictions[4][0], 8)

        self.assertEqual(predictions[0][1], 1)
        self.assertEqual(predictions[1][1], 1)
        self.assertEqual(predictions[2][1], 3)
        self.assertEqual(predictions[3][1], 3)
        self.assertEqual(predictions[4][1], 3)

    def test_prediction_weighting_score(self):
        # given
        dataset = prepare_dataset()
        weighting_score = ['linear', 'same', 'div', 'log', 'quadratic']
        recommenders = [VMSessionKnnRecommender(weighting_score=w) for w in weighting_score]
        [r.fit(dataset) for r in recommenders]

        # when
        predictions = [r.predict_single_session(create_session("test", 1, 3, 8), n=3) for r in recommenders]

        # then
        val = 3
        self.assertEqual(predictions[0][0], 3)
        self.assertEqual(predictions[1][0], 3)
        self.assertEqual(predictions[2][0], 8)
        self.assertEqual(predictions[3][0], 3)
        self.assertEqual(predictions[4][0], 8)

        self.assertEqual(predictions[0][1], 8)
        self.assertEqual(predictions[1][1], 8)
        self.assertEqual(predictions[2][1], 3)
        self.assertEqual(predictions[3][1], 8)
        self.assertEqual(predictions[4][1], 3)
