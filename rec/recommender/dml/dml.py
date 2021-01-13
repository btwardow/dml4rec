import logging
from collections import Counter
from copy import deepcopy
from time import time

import numpy as np
from rec.utils import tb
import torch
import torch.nn as nn
import tqdm

from rec.dataset.dataset import Dataset
from rec.dataset.testcase_generator import SubsequentEventTestCaseGenerator
from rec.eval import PrecisionRecallAtN, MeanReciprocalRank, evaluate_recommender, HitRateAtN
from rec.recommender.base import SessionAwareRecommender
from rec.recommender.dml.utils import minibatch, create, Identity
from rec.utils import current_milli_time

ACTIVATIONS = {'iden': Identity(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigm': nn.Sigmoid()}
VALID_MEASURES = (
    PrecisionRecallAtN(top_n=20),
    MeanReciprocalRank(max_events=20, single_next_item_only=True),
    HitRateAtN(20, single_next_item_only=True),
)


class DMLSessionRecommender(SessionAwareRecommender):
    def __init__(
        self,
        sampler={
            'name': 'SessionPosNegSampler',
            'batch_size': 32,
            'pos_len': 5,
            'neg_len': 5
        },
        optimizer={
            'name': 'Adam',
            'weight_decay': 0.0,
            'lr': 0.001
        },
        epochs=100,
        emb_dim=100,
        max_len=5,
        common_items_emb=False,
        use_cuda=torch.cuda.is_available(),
        max_emb_norm=None,
        i_activ="tanh",
        s_net={
            'name': 'MaxPool',
            'activ': 'tanh'
        },
        loss={
            'name': 'TripletLoss',
            's_margin': 1.0
        },
        val_freq=5,
        train_val_freq=None,
        log_debug=False,
        topk_items=1.0,
        lr_patience=3,
        lr_factor=10,
        lr_patience_improvement_rate=0.005,
        warmup_epochs=0,
        item_emb_dim=None,
        batch_adjust=True
    ):
        super().__init__()
        self.batch_adjust = batch_adjust
        self.lr_patience_improvement_rate = lr_patience_improvement_rate
        self.train_val_freq = train_val_freq
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.warmup_epochs = warmup_epochs
        self.log_debug = log_debug
        self.val_freq = val_freq
        self.logger = logging.getLogger(self.__class__.__name__)
        # params
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self.common_items_emb = common_items_emb
        self.max_len = max_len
        self.epochs = epochs
        self.emb_dim = emb_dim
        self.max_emb_norm = max_emb_norm

        self.item_emb_dim = item_emb_dim
        if item_emb_dim is None:
            self.item_emb_dim = item_emb_dim

        self.sampler = sampler

        # state
        self.topk_items = topk_items
        self.items_id_to_idx = None
        self.items_idx_to_id = None
        self.items_num = 0
        self.items_emb = None
        self.train_sessions = None

        self.loss = loss
        self.optimizer = optimizer

        # model
        self.s_net = s_net
        self._s_net: torch.nn.Module = None
        self.i_activ = i_activ
        self.i_net: torch.nn.Module = None
        self._optimizer: torch.optim.Optimizer = None
        self._lr_scheduler = None

        self._iteration = 0
        self.train_skip_num = 0
        self._train_records = 0
        self._train_sessions = []
        self._train_pos_items = []

        self.train_metric_data = None
        self.train_dataset = None

        self.epoch_end_callback = None

        self._val_mrr = []
        self._val_rec = []

        self._best_model = None
        self._best_model_mrr = 0.0

    def _create_model(self):
        self._s_net = create('rec.recommender.dml.s_nets', self, **self.s_net)
        self.i_net = torch.nn.Sequential(
            self._s_net.embed
            if self.common_items_emb else torch.nn.Embedding(self.items_num, self.emb_dim, max_norm=self.max_emb_norm),
            torch.nn.Linear(self.emb_dim, self.emb_dim), ACTIVATIONS[self.i_activ],
            torch.nn.Linear(self.emb_dim, self.emb_dim), ACTIVATIONS[self.i_activ]
        )

        self.loss_object = create('rec.recommender.dml.losses', **self.loss)

        self.i_net.to(self._device)
        self._s_net.to(self._device)
        self.loss_object.to(self._device)

    def _create_optim(self):
        optim_group = [dict(params=self._s_net.parameters())]

        i_net_params = []
        start_idx = 1 if self.common_items_emb else 0
        for i in range(start_idx, len(self.i_net)):
            i_net_params.extend(self.i_net[i].parameters())
        optim_group.append(dict(params=self.loss_object.parameters()))
        self._optimizer = create('torch.optim', optim_group, **self.optimizer)

    def fit(self, train_dataset, valid_data=None, valid_measures=VALID_MEASURES):
        self.train_dataset = train_dataset
        if self.items_emb is None:
            self._prepare_dataset(train_dataset)
            self._create_model()

        self._create_sampler()
        self._create_optim()

        self._prepare_train_dataset_for_validation(train_dataset)
        print('#' * 10)
        print('WARMUP TRAIN')
        print('#' * 10)
        self._train(train_dataset, valid_data, valid_measures, self.warmup_epochs)
        print('#' * 10)
        print('TRAIN')
        print('#' * 10)
        self._create_optim()
        self._train(train_dataset, valid_data, valid_measures, self.epochs)

    def _restore_best_model(self):
        if self._best_model:
            self.s_net, self.i_net = self._best_model
            if self.common_items_emb:
                if self._s_net.embed != self.i_net[0]:
                    print('Embedding should be the same. Changing reference...')
                    self._s_net.embed = self.i_net[0]

    def _train(self, train_dataset, valid_data, valid_measures, epochs):
        patience = self.lr_patience
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')

            if patience == 0:
                patience = self.lr_patience

                # chek if training is still possible
                if self.sampler['pos_len'] < 2 or self._sampler.pos_len < 2:
                    print("Cannot lower pos_len more. Ending training!")
                    break

                print('No more patience. Adjusting learning method...')

                if self.batch_adjust:
                    self._sampler.batch_size = int(self._sampler.batch_size * 0.8)
                    self._sampler.pos_len = self._sampler.pos_len - 1
                    self._sampler.neg_len = self._sampler.neg_len - 1
                    print(f'Batch size: {self._sampler.batch_size} pos_len: {self._sampler.pos_len}')

                # restart optimizer & start from best model
                if self._best_model:
                    self.s_net, self.i_net = self._best_model
                    if self.common_items_emb:
                        if self._s_net.embed != self.i_net[0]:
                            print('Embedding should be the same. Changing reference...')
                            self._s_net.embed = self.i_net[0]

                # new optimizer
                self.optimizer['lr'] = self.optimizer['lr'] * 0.8
                print(f'Lowering lr to: ', self.optimizer['lr'])
                self._create_optim()

            t1 = time()
            self.i_net.train()  # make sure we are on train after validation
            self._s_net.train()
            self._sampler()
            indices = torch.arange(len(self._sampler)).to(self._device)
            epoch_batch = 0
            epoch_loss = 0.0
            for batch_idx in tqdm.tqdm(indices):
                batch = self._sampler[batch_idx]

                session_items, pos_item, neg_item = batch[:3]
                if len(batch) == 5:
                    ses_pos_coo, pos_coo = batch[3:5]
                    ses_pos_coo = ses_pos_coo.to(self._device)
                    pos_coo = pos_coo.to(self._device)
                else:
                    ses_pos_coo, pos_coo = None, None

                if neg_item.dim() > 1:
                    pseudo_labels = None
                else:
                    # session_items, pos_item = batch
                    pseudo_labels = neg_item.flatten()
                    neg_item = None

                epoch_batch += 1
                self._iteration += 1

                # train step
                self._optimizer.zero_grad()
                s_emb = self._s_net(session_items)
                i_pos_emb = self.i_net(pos_item)
                i_neg_emb = self.i_net(neg_item) if neg_item is not None else None
                loss = self.loss_object(s_emb, i_pos_emb, i_neg_emb, ses_pos_coo, pos_coo, labels=pseudo_labels)
                epoch_loss += loss.item()
                tb.add_scalar('loss', epoch_loss, self._iteration)

                loss.backward()
                if self.log_debug:
                    self._log_grad_norm()
                # self._clip_grad_norm()
                self._optimizer.step()

                if self.log_debug:
                    tb.add_histogram('train/s_emb-l2', torch.norm(s_emb, dim=1).detach().cpu().numpy(), self._iteration)
                    tb.add_histogram(
                        'train/i_pos_emb-l2',
                        torch.norm(i_pos_emb, dim=1).detach().cpu().numpy(), self._iteration
                    )
                    if i_neg_emb is not None:
                        tb.add_histogram(
                            'train/i_neg_emb-l2',
                            torch.norm(i_neg_emb, dim=1).detach().cpu().numpy(), self._iteration
                        )

            t2 = time()
            print(f"Epoch {epoch + 1:03d} [{t2 - t1:.1f} s] \tloss={epoch_loss:.6f} ", end=' ')
            tb.add_scalar('epoch_loss', epoch_loss, epoch + 1)
            if self.val_freq is not None and ((((epoch % self.val_freq) == 0) or ((epoch + 1) ==epochs)) \
                    and valid_data is not None):
                rec, mrr, hr = self._validate(valid_data, valid_measures, epoch)
                print(f"Val HR: {hr:0.03f} Rec: {rec:0.03f} MRR: {mrr:0.03f}", end=' ')
                tb.add_scalar('val/rec', rec, epoch + 1)
                tb.add_scalar('val/mrr', mrr, epoch + 1)
                tb.add_scalar('val/hr', hr, epoch + 1)

                # check patience
                if len(self._val_mrr) > 0:
                    improvement_rate = ((mrr / self._val_mrr[-1]) - 1.0) if self._val_mrr[-1] != 0.0 else mrr
                    if improvement_rate < self.lr_patience_improvement_rate:
                        patience -= 1
                        print("Val MRR: ", self._val_mrr)
                        print(
                            f"MRR not improving enough. Diff: {improvement_rate} is lower than {self.lr_patience_improvement_rate}"
                        )
                    else:
                        patience = self.lr_patience

                if mrr > self._best_model_mrr:
                    print('Saving best model')
                    self._best_model = deepcopy((self.s_net, self.i_net))
                    self._best_model_mrr = mrr

                print(f"Patience: {patience}")
                self._val_mrr.append(mrr)
                self._val_rec.append(rec)

            if (self.train_val_freq is not None and epoch >= self.train_val_freq) and \
                    ( epoch > 0 and (((epoch % self.train_val_freq) == 0)) and epoch == (epochs + 1)):
                train_rec, train_mrr, train_hr = self._validate(self.train_metric_data, valid_measures, epoch)
                print(f"; Train HR: {train_hr:0.03f} Rec: {train_rec:0.03f} MRR: {train_mrr:0.03f}", end=' ')
                tb.add_scalar('train/rec', train_rec, epoch + 1)
                tb.add_scalar('train/mrr', train_mrr, epoch + 1)

            if self.epoch_end_callback is not None:
                self.epoch_end_callback(epoch)

        # after training
        self._restore_best_model()
        if valid_data is not None:
            # last validation
            rec, mrr, hr = self._validate(valid_data, valid_measures, self.epochs + 1)
            print(f"Last val Val HR: {hr:0.03f} Rec: {rec:0.03f} MRR: {mrr:0.03f}", end=' ')
        self._prepare_itmes_emb(verbose=True)

    def _clip_grad_norm(self):
        for pg in self._optimizer.param_groups:
            for p in pg['params']:
                torch.nn.utils.clip_grad_norm(p, 0.2)

    def _log_grad_norm(self):
        total_norm = 0.0
        for pg in self._optimizer.param_groups:
            for p in pg['params']:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item()**2
            total_norm = total_norm**(1. / 2)
        tb.add_scalar('train/total_norm', total_norm, self._iteration)

    def _validate(self, valid_data, valid_measures, epoch):
        self._prepare_itmes_emb()
        evaluation_results, predictions, predict_sec, eval_sec = evaluate_recommender(
            self, valid_data[0], valid_data[1], valid_measures, 20, 64
        )
        predictions = [self.items_id_to_idx[i] for p in predictions if len(p) > 0 for i in p]
        pred_unique_num = np.unique(predictions).size
        rec, mrr, hr = [evaluation_results[m] for m in ['rec', 'mrr', 'hr']]
        tb.add_histogram('test/predictions', predictions, epoch)
        print(f"Pred unique items: {pred_unique_num}. Eval sec: {eval_sec}")
        return rec, mrr, hr

    def predict_single_session(self, session, n=20, return_session_emb=False):
        if self._s_net is None:
            raise RuntimeError('First fit() the model!')
        self._s_net.eval()
        with torch.no_grad():
            session_items = session.clicked_items_list()[:self.max_len]
            session_items = [self.items_id_to_idx[i] for i in session_items]
            if len(session_items) == 0:
                return []
            # add padding if necessary
            session_items = self._sampler.session_padding(session_items)
            session_items = torch.LongTensor([session_items]).to(self._device)
            sesssio_emb = self._s_net(session_items)
            z_s = sesssio_emb.expand(self.items_emb.size())
            predictions = self.loss_object.dist(z_s, self.items_emb)
            top_n_item_indexes = predictions.argsort()[:n + 1]
            items_ids = [self.items_idx_to_id[idx] for idx in top_n_item_indexes.cpu().numpy() if idx != 0][:n]
            if return_session_emb:
                return items_ids, sesssio_emb.cpu().numpy()
            return items_ids

    def _prepare_dataset(self, train_dataset: Dataset):
        if self.topk_items > 1.0:
            self.selected_num = self.topk_items
        else:
            self.selected_num = int(self.topk_items * train_dataset.items_num())
        print(f'Selecting most popular items: {self.selected_num} will be taken')
        counter = Counter(i for s in train_dataset.all_sessions_list() for i in s.clicked_items_list())
        self.items_id_to_idx = {k: idx + 1 for idx, (k, v) in enumerate(counter.most_common(self.selected_num))}
        self.items_idx_to_id = {v: k for k, v in self.items_id_to_idx.items()}
        self.items_num = len(self.items_idx_to_id) + 1
        # will be mapped, but not used by model - mapped to the same idx (last one)
        less_popular = {
            k: self.items_num - 1
            for k in (train_dataset.items_id_to_idx.keys() - self.items_id_to_idx.keys())
        }
        self.items_id_to_idx.update(less_popular)

        self.train_sessions = [
            [self.items_id_to_idx[i] for i in s.clicked_items_list()] for s in train_dataset.all_sessions_list()
        ]
        assert max(max(s) for s in self.train_sessions) <= self.selected_num

        self.M, self.c = self.get_cooccurrence_matrix_with_support()
        print(f'Number of training sessions: {len(self.train_sessions)}')

    def _prepare_itmes_emb(self, verbose=False):
        self.i_net.eval()
        with torch.no_grad():
            indices = torch.arange(self.items_num).to(self._device)
            if verbose:
                print(f'Embedding all {indices.size(0)} items...')
            results = []
            for b in minibatch(indices, batch_size=32):
                i_z = self.i_net(b)
                results.append(i_z)
            self.items_emb = torch.cat(results, axis=0).to(self._device)
            if verbose:
                print(f'Embedding ended. Shape: {self.items_emb.shape}')
            assert self.items_emb.shape == (self.items_num, self.emb_dim)

    def __str__(self):
        n = self.__class__.__name__.split('Recommender')[0]
        s = f"{n}-{self.s_net['name']}-{self.loss['name']}"
        if self.loss['name'] == 'PML':
            s += f":{self.loss['loss']}"
        return s

    def get_cooccurrence_matrix_with_support(self):
        M_shape = (self.items_num, self.items_num)
        print("Cooccurrence matrix shape: {M_shape}".format(M_shape=M_shape))
        print("Building cooccurrence matrix...")
        start_time = current_milli_time()
        M = np.zeros(M_shape, dtype=np.uint16)
        c = np.zeros(self.items_num, dtype=np.uint16)
        for items_list in self.train_sessions:
            items_set = set(items_list)
            for i1 in items_list:
                for i2 in items_set:
                    if i1 != i2:
                        M[i1, i2] += 1
                c[i1] += 1

        print("Cooccurrence matrix build in time: {}(s)".format(float(current_milli_time() - start_time) / 1000))
        return M, c

    def _create_sampler(self):
        self._sampler = create('rec.recommender.dml.sampler', self, **self.sampler)

    def _prepare_train_dataset_for_validation(self, train_dataset):
        print('Preparing train data for metric...')
        tc = SubsequentEventTestCaseGenerator()
        self.train_metric_data = tc.generate(train_dataset)
        print('Preparing train data for metric...')
