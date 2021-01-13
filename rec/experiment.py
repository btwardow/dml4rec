import datetime
import json
import logging
import os
import pprint
import socket

import numpy as np

from rec.dataset.dataset import Dataset
from rec.dataset.split import DatasetSplitter, LastNPercentOfSessionsInDataset
from rec.dataset.testcase_generator import SubsequentEventTestCaseGenerator
from rec.eval import evaluate_recommender, PrecisionRecallAtN, MeanReciprocalRank, PrecisionRecallUpToN, \
    MeanReciprocalRankUpToN, HitRateAtN
from rec.recommender.base import SessionAwareRecommender
from .utils import current_milli_time, init_tb, tb

DEFAULT_TOP_N = 20
DEFAULT_EVAL_MEASURES = [
    PrecisionRecallAtN(DEFAULT_TOP_N),
    MeanReciprocalRank(max_events=DEFAULT_TOP_N),
    PrecisionRecallUpToN(DEFAULT_TOP_N),
    MeanReciprocalRankUpToN(DEFAULT_TOP_N)
]


class Experiment(object):
    def __init__(
        self,
        dataset,
        recommender,
        evaluation_measures=DEFAULT_EVAL_MEASURES,
        iteration_num=1,
        top_n=DEFAULT_TOP_N,
        train_test_splitter=LastNPercentOfSessionsInDataset(split_percent=0.1),
        test_case_generator=SubsequentEventTestCaseGenerator(),
        test_batch_size=64,
        train_valid_splitter=None,
        experiment_id=os.getpid(),
        use_tensorboard=False,
        results_dir='results'
    ):
        self.results_dir = results_dir
        assert isinstance(dataset, Dataset)
        assert isinstance(recommender, SessionAwareRecommender)

        self.experiment_id = experiment_id
        self.train_valid_splitter = train_valid_splitter
        self.timestamp_dt = datetime.datetime.now()
        self.timestamp = self.timestamp_dt.isoformat()
        self.dataset = dataset
        self.top_n = top_n
        self.recommender = recommender
        self.iteration_num = iteration_num
        self.evaluation_measures = evaluation_measures
        self.evaluation_results = dict()
        self.train_test_splitter = train_test_splitter
        self.test_case_generator = test_case_generator
        self.test_batch_size = test_batch_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_data = dict(experiment_id=experiment_id)
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            dir = 'logs/tensorboard'
            os.makedirs(dir, exist_ok=True)
            self.tf_log_dir = f'{dir}/{self.dataset.name}_{str(self.recommender)}_{self.experiment_id}_{self.iteration_num}'
            self.writer = init_tb(self.tf_log_dir)
        else:
            self.writer = tb

    def run(self, test_dataset=None):

        self.logger.info("Running experiment {} :\n{}".format(self.experiment_id, self))

        for experiment_num in range(1, self.iteration_num + 1):

            self.logger.info('Running experiment iteration: {experiment_num}'.format(experiment_num=experiment_num))
            self.writer.add_text('config', str(pprint.pformat(self.get_config())), 1)

            self.logger.info('Preparing train/test splits...')
            if test_dataset is None:
                train, test = self.train_test_splitter.split(self.dataset)
            else:
                train = self.dataset
                test = test_dataset

            self.logger.info('Preparing test evaluation cases...')
            test_sessions, test_ground_truth = self.prepare_evaluation_cases(test)

            # prepare validation dataset
            valid = None
            valid_data = None
            if self.train_valid_splitter:
                assert isinstance(self.train_valid_splitter, DatasetSplitter)
                self.logger.debug("Preparing validation dataset...")
                train, valid = self.train_valid_splitter.split(train)
                valid_data = self.prepare_evaluation_cases(valid)
            
            self.logger.info("Train dataset: {}".format(train))
            self.logger.info("Valid dataset: {}".format(valid))
            self.logger.info("Test  dataset: {}".format(test))

            self.logger.info('Training model...')
            model_fit_start = current_milli_time()
            self.recommender.fit(train, valid_data=valid_data)
            model_fit_sec = (current_milli_time() - model_fit_start) / 1000

            self.logger.info('Model trained in: {} sec.'.format(model_fit_sec))
            self.add_to_log(train_sec=model_fit_sec)

            self.evaluation_results, predictions, predict_sec, eval_sec = evaluate_recommender(
                self.recommender, test_sessions, test_ground_truth, self.evaluation_measures, self.top_n,
                self.test_batch_size
            )
            predictions = [train.items_id_to_idx[i] for p in predictions if len(p) > 0 for i in p]
            unique_pred_num = np.unique(predictions).size
            self.writer.add_histogram('test/predictions', np.array(predictions), 1)
            self.add_to_log(test_sessions_num=len(test_sessions))
            self.add_to_log(eval_start=eval_sec)
            self.add_to_log(test_sec=predict_sec)
            self.add_to_log(pred_unique_items=unique_pred_num)
            self.update_tensorboard(self._consolidate_results(self.evaluation_results, self.top_n), 'test')

    def prepare_evaluation_cases(self, test):
        self.logger.info('Prepare evaluation sessions and ground truth...')
        testcase_gen_start = current_milli_time()
        test_sessions, ground_truth = self.test_case_generator.generate(test)
        assert len(test_sessions) == len(ground_truth)
        testcase_gen_sec = (current_milli_time() - testcase_gen_start) / 1000
        self.add_to_log(testcase_gen_sec=testcase_gen_sec)
        self.logger.info('Test cases gen. in: {} sec.'.format(testcase_gen_sec))
        self.logger.info('Number of testing sessions: {num}'.format(num=len(test_sessions)))
        self.logger.info('Number of ground truth items: {num}'.format(num=sum([len(e) for e in ground_truth])))
        return test_sessions, ground_truth

    def add_to_log(self, **kwargs):
        self.log_data.update(kwargs)

    def get_config(self):
        return dict(
            timestamp=self.timestamp,
            dataset_name=self.dataset.name,
            iteration_num=self.iteration_num,
            test_batch_size=self.test_batch_size,
            recommender=self.recommender.get_config(),
            train_info=self.recommender.train_info if hasattr(self.recommender, 'train_info') else None,
            splitter=None if self.train_test_splitter is None else self.train_test_splitter.get_config(),
            test_case_generator=None if self.test_case_generator is None else self.test_case_generator.get_config(),
            top_n=self.top_n,
            hostname=socket.gethostname(),
            rec_str=str(self.recommender)
        )

    def get_data(self):
        experiment_data = self.get_config()
        experiment_data['results'] = self.evaluation_results
        experiment_data.update(self.log_data)
        return experiment_data

    def to_json(self):
        return json.dumps(self.get_data())

    def default_results_filename(self, suffix='json'):
        ts = self.timestamp_dt.strftime("%Y%m%d_%H%M")
        return f'{self.results_dir}/{self.dataset.name}_{str(self.recommender)}_{self.experiment_id}_{ts}.{suffix}'

    def update_tensorboard(self, d, prefix='', step=1):
        if isinstance(d, dict):
            for k, v in d.items():
                n = f"{prefix}/{k}" if prefix else str(k)
                self.update_tensorboard(v, n, step)
        elif isinstance(d, float):
            self.writer.add_scalar(prefix, d, step)
        elif isinstance(d, list):
            for i, _v in enumerate(d):
                self.update_tensorboard(_v, prefix, i + 1)
        else:
            self.logger.warning(f"Unsupported type: {type(d)}, values: {d}")

    def _consolidate_results(self, d, topn):
        r = {}
        for k, v in d.items():
            if '@k' in k:
                n = k.split('@k')[0]
                r[f"{n}@N"] = [_v[n] for _v in v]
                _topn = len(v)
                r[f"{n}@{_topn}-SL"] = v[_topn - 1][f'sl_{n}']
            else:
                r[k] = v
        return r

    def save_results(self, filename=None):
        if filename is None:
            os.makedirs(self.results_dir, exist_ok=True)
            filename = self.default_results_filename()
        with open(filename, 'w+') as f:
            json.dump(self.get_data(), f, sort_keys=True, indent=2)

    def run_and_save_results(self):
        self.run()
        self.save_results()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.get_data())
