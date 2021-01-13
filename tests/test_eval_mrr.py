import json
from unittest import TestCase

import numpy as np

from rec.eval import MeanReciprocalRank, MeanReciprocalRankUpToN
from rec.model import Session, TIMESTAMP, EVENT_ITEM, EVENT_TYPE


class TestMeanReciprocalRank(TestCase):
    def test_compute_mrr(self):
        gt = [[1], [2], [3]]
        p = [[1, 2, 3], [1, 3, 4], [1, 2, 3]]
        eval_measure = MeanReciprocalRank()
        mrr = eval_measure.compute(gt, p)
        self.assertAlmostEqual(mrr['mrr'], .44, 2)

    def test_all_relevant(self):
        gt = [[1], [1, 2], [3]]
        p = [[1], [1, 2], [3]]
        eval_measure = MeanReciprocalRank()
        mrr = eval_measure.compute(gt, p)

        self.assertAlmostEqual(mrr['mrr'], 1.0, 2)

    def test_none_relevant(self):
        gt = [[1], [1, 2], [3]]
        p = [[2], [3, 4, 5], [23]]
        eval_measure = MeanReciprocalRank()
        mrr = eval_measure.compute(gt, p)

        self.assertAlmostEqual(mrr['mrr'], .0, 2)

    def test_session_length_mrr(self):
        gt = [[1], [2], [3], [1], [2]]
        p = [[1, 2, 3], [1, 3, 4], [1, 2, 3], [1], [1, 2]]
        sessions = [
            self._create_dmmy_session_with_events_num(5),
            self._create_dmmy_session_with_events_num(5),
            self._create_dmmy_session_with_events_num(5),
            self._create_dmmy_session_with_events_num(1),
            self._create_dmmy_session_with_events_num(2)
        ]
        max_events = 10
        eval_measure = MeanReciprocalRank(max_events=max_events)
        mrr = eval_measure.compute(gt, p, sessions)
        sl_mrr_expected = np.zeros(max_events)
        sl_mrr_expected[5] = .44
        sl_mrr_expected[1] = 1.0
        sl_mrr_expected[2] = .5

        for mrr1, mrr2 in zip(mrr['sl_mrr'], sl_mrr_expected):
            self.assertAlmostEqual(mrr1, mrr2, 2)

    def test_up_to_n_mrr(self):
        gt = [[1], [2], [3], [1], [2]]
        p = [[1, 2, 3], [1, 3, 4], [1, 2, 3], [1], [1, 2]]
        sessions = [
            self._create_dmmy_session_with_events_num(5),
            self._create_dmmy_session_with_events_num(5),
            self._create_dmmy_session_with_events_num(5),
            self._create_dmmy_session_with_events_num(1),
            self._create_dmmy_session_with_events_num(2)
        ]
        max_events = 10
        n = 7
        eval_measure = MeanReciprocalRankUpToN(n=n, max_events=max_events)
        result = eval_measure.compute(gt, p, sessions)
        result = result['mrr@k']
        self.assertJsonSerialization(result)
        self.assertEqual(len(result), n - 1)
        for k in range(n - 1):
            self.assertIsInstance(result[k]['sl_mrr'], list)
            self.assertEqual(len(result[k]['sl_mrr']), max_events)
            self.assertGreaterEqual(result[k]['mrr'], .0)
            self.assertLessEqual(result[k]['mrr'], 1.)

    def _create_dmmy_session_with_events_num(self, events_num):
        s = Session(1, 1)
        for i in range(events_num):
            s.create_and_add_event({TIMESTAMP: i, EVENT_ITEM: i, EVENT_TYPE: 'CLICK'})
        return s

    def assertJsonSerialization(self, data):
        json_dump = json.dumps(data)
        data_out = json.loads(json_dump)
        self.assertListEqual(data, data_out)
