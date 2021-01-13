import os
import random
from unittest import TestCase

from rec.dataset.dataset import Dataset


class TestDataset(TestCase):
    def test_generate_test_data_with_density(self):
        sessions_num = 127
        items_num = 17
        density = .1
        ds = Dataset.generate_test_data_with_density(sessions_num, items_num, density)

        self.assertEqual(ds.sessions_num(), sessions_num)
        self.assertEqual(ds.items_num(), items_num)
        self.assertEqual(ds.events_num(), int(sessions_num * items_num * density))
        self.assertEqual(len(ds.sessions_idx_to_id), sessions_num)
        self.assertIsNotNone(len(ds.sessions_idx_to_id[sessions_num - 1]))

    def test_generate_test_data(self):
        sessions_num = 111
        items_num = 21
        events_num = 13
        ds = Dataset.generate_test_data(sessions_num, items_num, events_num)

        self.assertEqual(ds.sessions_num(), sessions_num)
        self.assertEqual(ds.items_num(), items_num)
        self.assertEqual(ds.events_num(), sessions_num * events_num)
        for us in ds.sessions.values():
            for s in us.values():
                s.events_num()

    def test_load_dataset_with_no_context(self):
        sample_dataset = Dataset().load_with_no_context('tests/data/sample_sessions.json')

        self.assertEqual(sample_dataset.users_num(), 2)
        self.assertEqual(sample_dataset.items_num(), 3)
        self.assertEqual(sample_dataset.sessions_num(), 4)

    def test_split_left_n_sessions(self):
        dataset = Dataset.generate_test_data(100, 100, 10, 4)
        n = 2
        train, test = dataset.split_left_n_sessions(n)

        self.assertIsInstance(train, Dataset)
        self.assertIsInstance(test, Dataset)

        # both dataset shouln't be empty
        self.assertGreater(len(train.sessions), 0)
        self.assertGreater(len(test.sessions), 0)

        self.assertGreaterEqual(
            train.users_num(), test.users_num(),
            'Users number in fit dataset should be grater of equal to num of users in test dataset'
        )

        # it should be last 2 sessions
        for u in test.users():
            test_sessions = test.sessions[u]
            self.assertLessEqual(len(test_sessions), 2)
            self.assertGreater(len(test_sessions), 0)
            for ts in test_sessions.values():
                ts.timestamp_start > max([tts.timestamp_start for tts in train.sessions[u].values()])

    def test_write_load_using_pickle(self):
        # given
        dataset = Dataset.generate_test_data_with_density(100, 100, .1)
        file_name = 'sample-ds.pkl.gz'
        dataset.write_to_file(file_name)
        dataset2 = Dataset.read_from_file(file_name)

        assert isinstance(dataset2, Dataset)

        self.assertEqual(dataset.items_num(), dataset2.items_num())
        self.assertEqual(dataset.sessions_num(), dataset2.sessions_num())
        os.remove(file_name)

    def test_one_hot_encoding_of_items_ids(self):
        # given
        dataset = Dataset.generate_test_data(100, 100, 10)
        # when
        idx, encoded_items = dataset.items_one_hot_encoding_item_id()
        # then
        self.assertEqual(len(idx), dataset.items_num())
        self.assertEqual(encoded_items.shape, (dataset.items_num(), dataset.items_num()))
        self.assertEqual(encoded_items[0, 0], 1)

    def test_random_sample_by_users(self):
        # when
        random.seed(1)
        dataset = Dataset.generate_test_data(100, 100, 10)
        sample = dataset.create_sample_by_users(2, True)

        # then
        assert isinstance(sample, Dataset)
        self.assertEqual(sample.users_num(), 2)
        self.assertEqual(sample.sessions_num(), 2)
        self.assertEqual(sample.items_num(), 100)

    def test_split_by_session(self):
        # given
        random.seed(1)
        ds = Dataset.generate_test_data(100, 10, 10)

        # when
        test, train = ds.split_sessions(60, 40)

        # then
        assert isinstance(test, Dataset)
        assert isinstance(train, Dataset)
        self.assertEqual(test.sessions_num(), 60)
        self.assertEqual(train.sessions_num(), 40)

    def test_create_session_and_items_indexes(self):
        ds = Dataset.generate_test_data(100, 10, 10)
        self.assertIsInstance(ds.items_id_to_idx, dict)
        self.assertIsInstance(ds.sessions_id_to_idx, dict)
        self.assertIsInstance(ds.sessions_idx_to_id, dict)
        self.assertEqual(len(ds.items_id_to_idx), ds.items_num())
        self.assertEqual(len(ds.items_idx_to_id), ds.items_num())
        self.assertEqual(len(ds.sessions_id_to_idx), ds.sessions_num())
        self.assertEqual(len(ds.sessions_idx_to_id), ds.sessions_num())

        self.assertListEqual(
            [(id, idx) for id, idx in ds.items_id_to_idx.items()],
            [(id, idx) for idx, id in ds.items_idx_to_id.items()],
        )

    def test_str(self):
        ds = Dataset.generate_test_data(100, 10, 10)
        s = ds.__str__()
        self.assertEqual(s, "Dataset: generated-test-dataset, sessions: 100, items: 10, events: 1000 (density: 100.0%)")