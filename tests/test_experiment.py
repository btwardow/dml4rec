import json
import os
from unittest import TestCase

from rec.dataset.dataset import Dataset
from rec.dataset.split import RandomSessionSplitter
from rec.eval import PrecisionRecallAtN, MeanReciprocalRank
from rec.experiment import Experiment
from rec.recommender.baseline import MostPopularRecommender
from rec.recommender.markov import MarkovRecommender
from rec.utils import seed_everything

seed_everything()


class TestExperiment(TestCase):
    def test_run(self):
        dataset = Dataset.generate_test_data()
        splitter = RandomSessionSplitter(0.7)
        experiment = Experiment(
            dataset=dataset,
            recommender=MarkovRecommender(),
            train_test_splitter=splitter,
            evaluation_measures=[PrecisionRecallAtN(5), MeanReciprocalRank()],
            iteration_num=1,
            use_tensorboard=True,
            top_n=20
        )
        experiment.run()

        self.assertGreaterEqual(experiment.evaluation_results['prec'], .0)
        self.assertGreaterEqual(experiment.evaluation_results['rec'], .0)
        self.assertGreaterEqual(experiment.evaluation_results['mrr'], .0)

    def test_experiment_to_json(self):
        dataset = Dataset.generate_test_data(name='test-DS')
        splitter = RandomSessionSplitter(0.7)
        experiment = Experiment(
            dataset=dataset,
            recommender=MostPopularRecommender(),
            train_test_splitter=splitter,
            evaluation_measures=[PrecisionRecallAtN(5), MeanReciprocalRank()],
            iteration_num=1
        )
        experiment.run()
        json_dump = experiment.to_json()
        data = json.loads(json_dump)

        self.assertEqual(data['dataset_name'], 'test-DS')
        self.assertGreater(data['results']['mrr'], 0.0)
        for v1 in data['results']['sl_mrr']:
            self.assertGreaterEqual(v1, .0)
            self.assertLessEqual(v1, 1.)
        for v1 in data['results']['sl_rec']:
            self.assertGreaterEqual(v1, .0)
            self.assertLessEqual(v1, 1.)

    def test_experiment_has_repr(self):
        dataset = Dataset.generate_test_data()
        splitter = RandomSessionSplitter(0.7)
        experiment = Experiment(
            dataset=dataset,
            recommender=MostPopularRecommender(),
            train_test_splitter=splitter,
            evaluation_measures=[PrecisionRecallAtN(5), MeanReciprocalRank()],
            iteration_num=1
        )
        repr_value = '{}'.format(experiment)

        self.assertIsInstance(repr_value, str)
        self.assertTrue(repr_value.startswith('Experiment'))

    def test_default_filename(self):
        dataset = Dataset.generate_test_data()
        splitter = RandomSessionSplitter(0.7)
        experiment = Experiment(
            dataset=dataset,
            recommender=MostPopularRecommender(),
            train_test_splitter=splitter,
            evaluation_measures=[PrecisionRecallAtN(5), MeanReciprocalRank()],
            iteration_num=1,
            use_tensorboard=True
        )

        self.assertRegex(experiment.default_results_filename(), 'results/generated-test-dataset_POP_(.*)_(.*).json')

    def test_save_to_file(self):
        dataset = Dataset.generate_test_data()
        splitter = RandomSessionSplitter(0.7)
        experiment = Experiment(
            dataset=dataset,
            recommender=MostPopularRecommender(),
            train_test_splitter=splitter,
            evaluation_measures=[PrecisionRecallAtN(5), MeanReciprocalRank()],
            iteration_num=1
        )
        file_name = 'test.json'
        experiment.save_results(file_name)

        def extract_dict(dict_a, dict_b):
            return dict([(k, dict_b[k]) for k in dict_a.keys() if k in dict_b.keys()])

        # check if file is appropriate json
        with open(file_name, 'r') as f:
            data = json.load(f)
            rec_dict = dict(rec_str='POP')
            self.assertEqual(rec_dict, extract_dict(rec_dict, data))

        # clean up the mess...
        os.remove(file_name)
