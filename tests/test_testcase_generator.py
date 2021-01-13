from unittest import TestCase

from rec.dataset.dataset import Dataset
from rec.dataset.testcase_generator import LeftNEventsTestCaseGenerator, SubsequentEventTestCaseGenerator, \
    AllViewedItemsTestCaseGenerator
from rec.model import Session


class TestIdentityTestCaseGenerator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset('test', create_indices=True).load_from_file(
            'tests/data/sample_sessions.json', 'tests/data/sample_items.json'
        )

    def test_generate(self):
        # given
        generator = AllViewedItemsTestCaseGenerator()

        # wehn
        t_list, gt_list = generator.generate(self.dataset)

        # then
        self.assertIsInstance(t_list, list)
        self.assertIsInstance(gt_list, list)
        for t, gt in zip(t_list, gt_list):
            self.assertIsInstance(t, Session)
            self.assertIsInstance(gt, list)
            self.assertEqual(len(t.clicked_items_set()), len(gt))


class TestLeftNEventsTestCaseGenerator(TestCase):
    def test_generate(self):
        # given
        dataset = Dataset.generate_test_data(10, 10, 4)
        n = 2
        generator = LeftNEventsTestCaseGenerator(n)

        # when
        t, gt = generator.generate(dataset)

        # then
        self.assertIsInstance(t, list)
        self.assertIsInstance(gt, list)
        for s in t:
            self.assertIsInstance(s, Session)
            self.assertGreater(s.events_num(), 0)

        for item_ids in gt:
            self.assertIsInstance(item_ids, list)
            self.assertGreater(len(item_ids), 0)
            self.assertLessEqual(len(item_ids), n)


class TestSubsequentEventTestCaseGenerator(TestCase):
    def test_generate(self):
        # given
        dataset = Dataset.generate_test_data(10, 10, 4)
        generator = SubsequentEventTestCaseGenerator()

        # wehn
        t, gt = generator.generate(dataset)

        # then
        self.assertIsInstance(t, list)
        self.assertIsInstance(gt, list)

        self.assertGreaterEqual(len(t), 20)
        self.assertGreaterEqual(len(gt), 20)

        for s in t:
            self.assertIsInstance(s, Session)
            self.assertGreater(s.events_num(), 0)

        for item_ids in gt:
            self.assertIsInstance(item_ids, list)
            self.assertGreaterEqual(len(item_ids), 1)

    def test_generate_only_new(self):
        # given
        dataset = Dataset.generate_test_data(10, 10, 12)
        generator = SubsequentEventTestCaseGenerator(only_new=True)

        # wehn
        t, gt = generator.generate(dataset)

        # then
        self.assertIsInstance(t, list)
        self.assertIsInstance(gt, list)

        self.assertLessEqual(len(t), 100)
        self.assertLessEqual(len(gt), 100)

        for s in t:
            self.assertIsInstance(s, Session)
            self.assertGreater(s.events_num(), 0)

        for item_ids in gt:
            self.assertIsInstance(item_ids, list)
            self.assertGreaterEqual(len(item_ids), 1)
