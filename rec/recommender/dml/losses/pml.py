import copy

import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.miners import BaseMiner

from rec.recommender.dml.losses.losses import squared_diffs
from rec.recommender.dml.utils import create


class PML(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = copy.deepcopy(kwargs)

        self.miner: BaseMiner = None
        if 'miner' in kwargs:
            self.miner = create('pytorch_metric_learning.miners', **kwargs['miner'])
            del kwargs['miner']

        kwargs['name'] = kwargs['loss']
        del kwargs['loss']
        self.loss: losses.BaseMetricLossFunction = create('pytorch_metric_learning.losses', *args, **kwargs)
        self.dist = squared_diffs

    def __call__(self, s, p, n=None, sp_coo=None, p_coo=None, labels=None):
        B, N, D = p.size()
        if n is not None:
            embeddings = torch.cat((s, p.reshape(B * N, D), n.reshape(B * N, D)), dim=0)
            labels_s = torch.arange(B, device=s.device, dtype=torch.long)
            labels_p = labels_s.repeat_interleave(N)
            labels_n = torch.arange(B * N, device=s.device, dtype=torch.long) + B
            _labels = torch.cat((labels_s, labels_p, labels_n))
        else:
            L = labels.size(0)
            S, D = s.size()
            SB = S // L
            embeddings = torch.cat((s, p.reshape(B * N, D)), dim=0)
            _labels = torch.cat((labels.repeat_interleave(SB), labels.repeat_interleave(N)), dim=0)

        selected_pairs = None
        if self.miner:
            selected_pairs = self.miner(embeddings, _labels)

        return self.loss(embeddings, _labels, indices_tuple=selected_pairs)

    def __str__(self):
        return f"PML:{self.loss.__class__.__name__}"
