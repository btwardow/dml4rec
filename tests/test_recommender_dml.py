from unittest import TestCase

from rec.dataset.dataset import Dataset
from rec.dataset.split import LastNPercentOfSessionsInDataset
from rec.dataset.testcase_generator import SubsequentEventTestCaseGenerator
from rec.experiment import Experiment
from rec.recommender.dml.dml import DMLSessionRecommender
from rec.utils import seed_everything

seed_everything(1, pytorch=True)


class TestDMLRecommender(TestCase):
    def setUp(self):
        self.dataset = Dataset.generate_test_data()
        splitter = LastNPercentOfSessionsInDataset(split_percent=0.1)
        self.train, test = splitter.split(self.dataset)
        test_case_generator = SubsequentEventTestCaseGenerator()
        self.test_sessions, _ = test_case_generator.generate(test)

    def test_dml4rec_experiment(self):
        # given
        n = 5
        r = DMLSessionRecommender(epochs=10,
                                  emb_dim=13,
                                  sampler={'name': 'SessionPosNegSampler', 'batch_size': 8},
                                  loss={'name': 'PML', 'loss': 'ContrastiveLoss', 'neg_margin': 1.01,
                                        'miner': {'name': 'BatchHardMiner'}}
                                  )
        e = Experiment(self.dataset, r, top_n=n, use_tensorboard=True)

        # when
        e.run_and_save_results()

        # check outcome
        assert e.evaluation_results is not None

    def test_predict_subset_of_items(self):
        n = 5
        r = DMLSessionRecommender(epochs=10,
                                  emb_dim=2,
                                  sampler={'name': 'SessionPosNegSampler', 'batch_size': 8},
                                  topk_items=0.8
                                  )
        r.fit(self.train)

        predictions = r.predict_single_session(self.test_sessions[0], n)

        self.assertEqual(str(r), 'DMLSession-MaxPool-TripletLoss')
        self.assertEqual(n, len(predictions))
        for p in predictions:
            self.assertIn(p, self.train.items)

    def test_predict_contrastive(self):
        n = 5
        r = DMLSessionRecommender(epochs=10,
                                  emb_dim=13,
                                  sampler={'name': 'SessionPosNegSampler', 'batch_size': 8},
                                  loss={'name': 'ContrastiveLoss'},
                                  )
        r.fit(self.train)

        predictions = r.predict_single_session(self.test_sessions[0], n)

        self.assertEqual(str(r), 'DMLSession-MaxPool-ContrastiveLoss')
        self.assertEqual(n, len(predictions))
        for p in predictions:
            self.assertIn(p, self.train.items)

    def test_predict_bpr(self):
        n = 5
        r = DMLSessionRecommender(epochs=10,
                                  emb_dim=13,
                                  sampler={'name': 'SessionPosNegSampler', 'batch_size': 8},
                                  loss={'name': 'BPRLoss'},
                                  )
        r.fit(self.train)

        predictions = r.predict_single_session(self.test_sessions[0], n)

        self.assertEqual(str(r), 'DMLSession-MaxPool-BPRLoss')
        self.assertEqual(n, len(predictions))
        for p in predictions:
            self.assertIn(p, self.train.items)

    def test_predict_weighted(self):
        n = 5
        r = DMLSessionRecommender(
            epochs=10, emb_dim=13,
            s_net={'name': 'WeightedPool', 'activ': 'tanh'},
            sampler={'name': 'SessionPosNegSampler', 'batch_size': 8}
        )
        r.fit(self.train)

        predictions = r.predict_single_session(self.test_sessions[0], n)

        self.assertEqual(str(r), 'DMLSession-WeightedPool-TripletLoss')
        self.assertEqual(n, len(predictions))
        for p in predictions:
            self.assertIn(p, self.train.items)

    def test_predict_triplet_weighted(self):
        n = 5
        r = DMLSessionRecommender(
            epochs=10, emb_dim=13,
            loss={'name': 'TripletLoss', 'weighting': True},
            sampler={'name': 'SessionPosNegSampler', 'batch_size': 8}
        )
        r.fit(self.train)

        predictions = r.predict_single_session(self.test_sessions[0], n)

        self.assertEqual(str(r), 'DMLSession-MaxPool-TripletLoss')
        self.assertEqual(n, len(predictions))
        for p in predictions:
            self.assertIn(p, self.train.items)

    def test_predict_weighted_SDML(self):
        n = 5
        r = DMLSessionRecommender(
            epochs=10, emb_dim=13,
            s_net={'name': 'WeightedPool', 'activ': 'tanh'},
            loss={'name': 'SDMLLoss', 'smoothing_parameter': 0.3},
            sampler={'name': 'SessionPosNegSampler', 'batch_size': 8}
        )
        r.fit(self.train)

        predictions = r.predict_single_session(self.test_sessions[0], n)

        self.assertEqual(str(r), 'DMLSession-WeightedPool-SDMLLoss')
        self.assertEqual(n, len(predictions))
        for p in predictions:
            self.assertIn(p, self.train.items)

    def test_predict_SDMLAllLoss(self):
        n = 5
        r = DMLSessionRecommender(
            epochs=10, emb_dim=13,
            loss={'name': 'SDMLAllLoss', 'smoothing_parameter': 0.7, 'label_softmax': True},
            sampler={'name': 'SessionPosNegSampler', 'batch_size': 8}
        )
        r.fit(self.train)

        predictions = r.predict_single_session(self.test_sessions[0], n)

        self.assertEqual(str(r), 'DMLSession-MaxPool-SDMLAllLoss')
        self.assertEqual(n, len(predictions))
        for p in predictions:
            self.assertIn(p, self.train.items)

    def test_predict_PML(self):

        losses = [
            {'name': 'PML', 'loss': 'ContrastiveLoss', 'neg_margin': 1.01},
            {'name': 'PML', 'loss': 'TripletMarginLoss', 'margin': 1.0},
            {'name': 'PML', 'loss': 'NPairsLoss'},
            {'name': 'PML', 'loss': 'MultiSimilarityLoss', 'alpha': 0.1, 'beta': 40, 'base': 0.5},
            {'name': 'PML', 'loss': 'FastAPLoss', 'num_bins': 1000},
            {'name': 'PML', 'loss': 'AngularLoss', 'alpha': 35},
            {'name': 'PML', 'loss': 'SignalToNoiseRatioContrastiveLoss', 'pos_margin': 0.1, 'neg_margin': 1.0,},
        ]

        for loss in losses:
            print('Testing loss', loss)
            n = 5
            r = DMLSessionRecommender(
                epochs=10, emb_dim=13,
                loss=loss,
                sampler={'name': 'SessionPosNegSampler', 'batch_size': 8}
            )
            r.fit(self.train)

            predictions = r.predict_single_session(self.test_sessions[0], n)

            self.assertEqual(str(r), 'DMLSession-MaxPool-PML:{}'.format(loss['loss']))
            self.assertEqual(n, len(predictions))
            for p in predictions:
                self.assertIn(p, self.train.items)

    def test_predict_PML_with_labels(self):

        losses = [
            {'name': 'PML', 'loss': 'TripletMarginLoss', 'margin': 1.0},
            {'name': 'PML', 'loss': 'ProxyNCALoss', 'num_classes': 90, 'embedding_size': 13},
            {'name': 'PML', 'loss': 'NormalizedSoftmaxLoss', 'temperature': 1, 'num_classes': 90, 'embedding_size': 13},
        ]

        for loss in losses:
            print('Testing loss', loss)
            n = 5
            r = DMLSessionRecommender(
                epochs=10, emb_dim=13,
                loss=loss,
                sampler={'name': 'BalancedPerItemSampler', 'batch_items': 4, 'instances': 4},
                topk_items=90
            )
            r.fit(self.train)

            predictions = r.predict_single_session(self.test_sessions[0], n)

            self.assertEqual(str(r), 'DMLSession-MaxPool-PML:{}'.format(loss['loss']))
            self.assertEqual(n, len(predictions))
            for p in predictions:
                self.assertIn(p, self.train.items)

    def test_different_s_nets(self):
        s_nets = [
            {'name': 'OnlyLastItem', 'activ': 'tanh'},
            {'name': 'TagSpace'},
            {'name': 'CNN1D'},
            {'name': 'TextCNN'},
            {'name': 'RNN'}
        ]
        loss = {'name': 'ContrastiveLoss'}
        sampler = {'name': 'SessionPosNegSampler', 'batch_size': 8}
        for s_net in s_nets:
            n = 5
            r = DMLSessionRecommender(epochs=10, emb_dim=13, s_net=s_net, loss=loss, sampler=sampler)
            r.fit(self.train)

            predictions = r.predict_single_session(self.test_sessions[0], n)

            self.assertEqual(str(r), 'DMLSession-{}-ContrastiveLoss'.format(s_net['name']))
            self.assertEqual(n, len(predictions))
            for p in predictions:
                self.assertIn(p, self.train.items)

    def test_different_samplers(self):
        samplers = [
            {'name': 'SessionPosNegSampler', 'batch_size': 8},
            {'name': 'SlidingWindowSampler', 'batch_size': 8},
            {'name': 'SubSeqSampler', 'batch_size': 8},

        ]
        for sampler in samplers:
            print('Sampler: ', sampler)
            n = 5
            r = DMLSessionRecommender(epochs=5, emb_dim=13, sampler=sampler, max_len=7)
            r.fit(self.train)
            predictions = r.predict_single_session(self.test_sessions[0], n)

            self.assertEqual(str(r), 'DMLSession-MaxPool-TripletLoss')
            self.assertEqual(n, len(predictions))
            for p in predictions:
                self.assertIn(p, self.train.items)

    def test_sampler_with_warmup(self):
        samplers = [
            {'name': 'SubSeqSampler', 'batch_size': 8, 'warmup_calls': 1},
        ]
        for sampler in samplers:
            n = 5
            r = DMLSessionRecommender(epochs=10, emb_dim=13, sampler=sampler, max_len=7, warmup_epochs=5)
            r.fit(self.train)
            predictions = r.predict_single_session(self.test_sessions[0], n)

            self.assertEqual(str(r), 'DMLSession-MaxPool-TripletLoss')
            self.assertEqual(n, len(predictions))
            for p in predictions:
                self.assertIn(p, self.train.items)
