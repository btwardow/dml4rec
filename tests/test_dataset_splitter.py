from unittest import TestCase

from rec.dataset.split import TimestampSessionSplitter, LastNPercentOfSessionsInDataset
from rec.dataset.dataset import Dataset
from rec.dataset.split import PreciseUserNumberDatasetSplitter, RandomSessionSplitter


class TestRandomDatasetSplitter(TestCase):
    def test_split(self):
        # given
        dataset = Dataset.generate_test_data()
        train_size = 5
        test_size = 2
        splitter = PreciseUserNumberDatasetSplitter(train_size, test_size)

        # when
        train, test = splitter.split(dataset)

        # then
        self.assertIsInstance(train, Dataset)
        self.assertIsInstance(test, Dataset)

        self.assertEqual(train.items_num(), 100)
        self.assertEqual(test.items_num(), 100)

        self.assertEqual(train.sessions_num(), 6)
        self.assertEqual(test.sessions_num(), 2)


class TestRandomSessionSplitter(TestCase):
    def test_split(self):
        # given
        dataset = Dataset.generate_test_data()
        train_ratio = 0.7
        splitter = RandomSessionSplitter(train_ratio=train_ratio)

        # when
        train, test = splitter.split(dataset)

        # then
        self.assertIsInstance(train, Dataset)
        self.assertIsInstance(test, Dataset)

        self.assertEqual(train.items_num(), 100)
        self.assertEqual(test.items_num(), 100)

        self.assertAlmostEqual(
            1.0 * train.sessions_num() / (test.sessions_num() + train.sessions_num()), train_ratio, 1
        )


class TestTimestampSessionSplitter(TestCase):
    def test_split(self):
        # given
        dataset = Dataset.generate_test_data(10, 10, 10)
        splitter = TimestampSessionSplitter(split_sec=15)

        # when
        train, test = splitter.split(dataset)

        # then
        self.assertIsInstance(train, Dataset)
        self.assertIsInstance(test, Dataset)

        self.assertEqual(train.items_num(), 10)
        self.assertEqual(test.items_num(), 10)

        self.assertEqual(test.sessions_num(), 2)
        self.assertEqual(train.sessions_num(), 8)


class TestLastNPercentOfSessionsInDataset(TestCase):
    def test_split(self):
        # given
        dataset = Dataset.generate_test_data(10, 10, 10)
        splitter = LastNPercentOfSessionsInDataset(.2)

        # when
        train, test = splitter.split(dataset)

        # then
        self.assertIsInstance(train, Dataset)
        self.assertIsInstance(test, Dataset)

        self.assertEqual(train.items_num(), 10)
        self.assertEqual(test.items_num(), 10)

        self.assertEqual(test.sessions_num(), 2)
        self.assertEqual(train.sessions_num(), 8)

        train_max = max([s.timestamp_start for s in train.all_sessions_list()])
        test_min = min([s.timestamp_start for s in test.all_sessions_list()])
        self.assertLessEqual(train_max, test_min)
