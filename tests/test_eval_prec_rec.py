import json
from unittest import TestCase

import numpy as np
from pprint import pprint

from rec.eval import PrecisionRecall, PrecisionRecallAtN, MeanReciprocalRank, PrecisionRecallUpToN
from rec.model import Session, TIMESTAMP, EVENT_ITEM, EVENT_TYPE


class TestPrecisionRecall(TestCase):
    def test_compute_precision_and_recall(self):
        gt = [1, 2, 3, 5]
        p = [1, 6]
        eval_measure = PrecisionRecall()
        precision, recall = eval_measure.compute(gt, p)

        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.25)

    def test_compute_precision_and_recall_for_no_predictions(self):
        gt = [1]
        p = []
        eval_measure = PrecisionRecall()
        precision, recall = eval_measure.compute(gt, p)

        self.assertEqual(precision, 0)
        self.assertEqual(recall, 0)

    def test_precision_and_recall_at_n(self):
        gt = [[1, 2, 3, 5], [10, 11]]
        p = [[1, 6], [10]]
        eval_measure = PrecisionRecallAtN(4)
        result = eval_measure.compute(gt, p)
        precision, recall = result['prec'], result['rec']

        self.assertEqual(precision, 0.75)
        self.assertEqual(recall, 0.375)

    def test_precision_and_recall_on_single_event_gt(self):
        gt = [[1], [2], [3]]
        p = [[1, 2, 3], [1, 3, 4], [1, 2, 3]]
        eval_measure = PrecisionRecallAtN(3)
        result = eval_measure.compute(gt, p)
        precision, recall = result['prec'], result['rec']

        self.assertAlmostEqual(precision, .22, 2)
        self.assertAlmostEqual(recall, .67, 2)

    def test_session_length_rec(self):
        gt = [[1], [2], [3], [1], [1]]
        p = [[1, 2, 3], [1, 3, 4], [1, 2, 3], [1], [2]]
        sessions = [
            self._create_dmmy_session_with_events_num(3),
            self._create_dmmy_session_with_events_num(3),
            self._create_dmmy_session_with_events_num(3),
            self._create_dmmy_session_with_events_num(1),
            self._create_dmmy_session_with_events_num(1),
        ]
        max_events = 5
        eval_measure = PrecisionRecallAtN(top_n=10, max_events=max_events)
        result = eval_measure.compute(gt, p, sessions)
        sl_rec_expected = np.zeros(max_events)
        sl_rec_expected[3] = .666
        sl_rec_expected[1] = .5

        for rec1, rec2 in zip(result['sl_rec'], sl_rec_expected):
            self.assertAlmostEqual(rec1, rec2, 2)

    def test_prec_rec_up_to_k(self):
        gt = [[1], [2], [3], [1], [1]]
        p = [[1, 2, 3], [1, 3, 4], [1, 2, 3], [1], [2]]
        sessions = [
            self._create_dmmy_session_with_events_num(3),
            self._create_dmmy_session_with_events_num(3),
            self._create_dmmy_session_with_events_num(3),
            self._create_dmmy_session_with_events_num(1),
            self._create_dmmy_session_with_events_num(1),
        ]
        max_events = 5
        top_n = 10
        eval_measure = PrecisionRecallUpToN(top_n=top_n, max_events=max_events)
        result = eval_measure.compute(gt, p, sessions)
        result = result['rec@k']
        self.assertJsonSerialization(result)
        self.assertEqual(len(result), top_n - 1)
        for k in range(top_n - 1):
            self.assertIsInstance(result[k]['sl_rec'], list)
            self.assertEqual(len(result[k]['sl_rec']), max_events)
            self.assertGreaterEqual(result[k]['rec'], .0)
            self.assertLessEqual(result[k]['rec'], 1.)
            self.assertGreaterEqual(result[k]['prec'], .0)
            self.assertLessEqual(result[k]['prec'], 1.)

    def _create_dmmy_session_with_events_num(self, events_num):
        s = Session(1, 1)
        for i in range(events_num):
            s.create_and_add_event({TIMESTAMP: i, EVENT_ITEM: i, EVENT_TYPE: 'CLICK'})
        return s

    def assertJsonSerialization(self, data):
        json_dump = json.dumps(data)
        data_out = json.loads(json_dump)
        self.assertListEqual(data, data_out)
