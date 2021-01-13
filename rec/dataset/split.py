from abc import abstractmethod

from numpy import random

from rec.base import ParametrizedObject
from rec.dataset.dataset import Dataset


class DatasetSplitter(ParametrizedObject):
    @abstractmethod
    def split(self, dataset):
        assert isinstance(dataset, Dataset)
        pass

    def _prepare_target_datasets(self, dataset):
        train = Dataset(dataset.name)
        test = Dataset(dataset.name)
        train.items = dataset.items
        test.items = dataset.items
        return train, test


class IdentitySplitter(DatasetSplitter):
    """
    Do not split dataset at all.
    It returns for both, train and test, the same object.

    This implementation is mainly for testing purpose.
    It shouldn't be used in a real-life training schedule.
    """
    def split(self, dataset):
        return dataset, dataset


class PreciseUserNumberDatasetSplitter(DatasetSplitter):
    def __init__(self, train_size=0, test_size=0):
        super(PreciseUserNumberDatasetSplitter, self).__init__()
        self.train_size = train_size
        self.test_size = test_size

    def split(self, dataset):
        super(PreciseUserNumberDatasetSplitter, self).split(dataset)
        train, test = self._prepare_target_datasets(dataset)

        n = 0
        for u, u_sessions in list(dataset.sessions.items()):
            if n <= self.train_size:
                train.sessions[u] = u_sessions
            elif n <= self.train_size + self.test_size:
                test.sessions[u] = u_sessions
            else:
                break

            n += len(u_sessions)

        train._create_indexes()
        test._create_indexes()
        return train, test


class RandomSessionSplitter(DatasetSplitter):
    def __init__(self, train_ratio=0.7):
        super(RandomSessionSplitter, self).__init__()
        self.test_ratio = train_ratio

    def split(self, dataset):
        super(RandomSessionSplitter, self).split(dataset)
        train, test = self._prepare_target_datasets(dataset)
        test_session_num = self.test_ratio * dataset.sessions_num()

        user_session_ids = []
        for u, u_sessions in list(dataset.sessions.items()):
            for sid in u_sessions.keys():
                user_session_ids.append((u, sid))

        random.shuffle(user_session_ids)
        for n in range(len(user_session_ids)):
            u, sid = user_session_ids[n]
            out_dataset = train if n <= test_session_num else test
            out_dataset.sessions[u][sid] = dataset.sessions[u][sid]

        train._create_indexes()
        test._create_indexes()
        return train, test


class TimestampSessionSplitter(DatasetSplitter):
    def __init__(self, split_sec=24 * 60 * 60):
        super(TimestampSessionSplitter, self).__init__()
        self.split_sec = split_sec

    def split(self, dataset):
        super(TimestampSessionSplitter, self).split(dataset)
        train, test = self._prepare_target_datasets(dataset)
        max_ts = self._get_max_timestamp(dataset)
        threshold = max_ts - self.split_sec
        for u, u_sessions in list(dataset.sessions.items()):
            for sid, session in list(u_sessions.items()):
                out_dataset = train if session.timestamp_end < threshold else test
                out_dataset.sessions[u][sid] = dataset.sessions[u][sid]

        train._create_indexes()
        test._create_indexes()
        return train, test

    def _get_max_timestamp(self, dataset):
        max_ts = 0
        for u, u_sessions in list(dataset.sessions.items()):
            for sid, session in list(u_sessions.items()):
                if session.timestamp_end > max_ts:
                    max_ts = session.timestamp_end
        return max_ts


class LastNPercentOfSessionsInDataset(DatasetSplitter):
    def __init__(self, split_percent=.05):
        self.split_percent = split_percent

    def split(self, dataset):
        all_sessions = dataset.all_sessions_list()
        sorted(all_sessions, key=lambda s: s.timestamp_start)
        split_num = len(all_sessions) * self.split_percent
        train, test = self._prepare_target_datasets(dataset)

        # iterate from last event till split is filled
        for s in reversed(all_sessions):
            out_dataset = train
            if split_num > 0:
                split_num -= 1
                out_dataset = test
            out_dataset.sessions[s.user_id][s.id] = s

        train._create_indexes()
        test._create_indexes()
        return train, test
