from unittest import TestCase

from rec.eval import MeanAveragePrecisionAtN


class TestMeanAveragePrecision(TestCase):
    def test_all_relevant(self):
        gt = [[1], [1, 2], [3]]
        p = [[1], [1, 2], [3]]
        eval_measure = MeanAveragePrecisionAtN(1)
        m = eval_measure.compute(gt, p)

        self.assertAlmostEqual(m['map'], 1.0, 2)

    def test_none_relevant(self):
        gt = [[1], [1, 2], [3]]
        p = [[2], [3, 4, 5], [23]]
        eval_measure = MeanAveragePrecisionAtN()
        m = eval_measure.compute(gt, p)

        self.assertAlmostEqual(m['map'], .0, 1)
