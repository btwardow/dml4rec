import torch
import torch.nn.functional as F


class SDMLLoss(torch.nn.Module):
    r"""

    Implementation based on:
    https://github.com/apache/incubator-mxnet/pull/17298/files

    Calculates Batchwise Smoothed Deep Metric Learning (SDML) Loss given two input tensors and a smoothing weight
    SDM Loss learns similarity between paired samples by using unpaired samples in the minibatch
    as potential negative examples.
    The loss is described in greater detail in
    "Large Scale Question Paraphrase Retrieval with Smoothed Deep Metric Learning."
    - by Bonadiman, Daniele, Anjishnu Kumar, and Arpit Mittal.  arXiv preprint arXiv:1905.12786 (2019).
    URL: https://arxiv.org/pdf/1905.12786.pdf
    According to the authors, this loss formulation achieves comparable or higher accuracy to
    Triplet Loss but converges much faster.
    The loss assumes that the items in both tensors in each minibatch
    are aligned such that x1[0] corresponds to x2[0] and all other datapoints in the minibatch are unrelated.
    `x1` and  `x2` are minibatches of vectors.

    Parameters
    ----------
    smoothing_parameter : float
        Probability mass to be distributed over the minibatch. Must be < 1.0.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    Inputs:
        - **x1**: Minibatch of data points with shape (batch_size, vector_dim)
        - **x2**: Minibatch of data points with shape (batch_size, vector_dim)
          Each item in x2 is a positive sample for the same index in x1.
          That is, x1[0] and x2[0] form a positive pair, x1[1] and x2[1] form a positive pair - and so on.
          All data points in different rows should be decorrelated
    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """
    def __init__(self, smoothing_parameter=0.3, dist='ssd', d_pred='softmax'):
        super().__init__()
        self.kl_loss = torch.nn.KLDivLoss(reduction='mean')  # reduction="sum")
        self.smoothing_parameter = smoothing_parameter  # Smoothing probability mass
        self.labels_cache = dict()
        if dist == 'ssd':
            self.dist = squared_diffs
        elif dist == 'cos':
            self.dist = lambda x, y: (1.0 - (F.normalize(x, p=2, dim=1) * F.normalize(y, p=2, dim=1)).sum(-1)) / 2
        else:
            raise RuntimeError(f'Unknow distance function: {dist}')

        if d_pred == 'softmax':
            self.d_pred = lambda d: torch.log_softmax(-d, dim=1)
        elif d_pred == 'sigmoid':
            self.d_pred = lambda d: torch.log(1.0 - torch.sigmoid(d))
        else:
            raise RuntimeError(f'Unknow distance function: {dist}')

    def _compute_labels(self, batch_size, device):
        """
        The function creates the label matrix for the loss.
        It is an identity matrix of size [BATCH_SIZE x BATCH_SIZE]
        labels:
            [[1, 0]
             [0, 1]]
        after the process the labels are smoothed by a small amount to
        account for errors.
        labels:
            [[0.9, 0.1]
             [0.1, 0.9]]
        Pereyra, Gabriel, et al. "Regularizing neural networks by penalizing
        confident output distributions." arXiv preprint arXiv:1701.06548 (2017).
        """
        gold = torch.eye(batch_size, device=device)
        labels = gold * (1 - self.smoothing_parameter) + (1 - gold) * self.smoothing_parameter / (batch_size - 1)
        return labels

    def __call__(self, x1, x2, neg=None, ses_pos_coo=None, pos_coo=None, labels=None):
        """
        the function computes the kl divergence between the negative distances
        (internally it compute a softmax casting into probabilities) and the
        identity matrix.
        This assumes that the two batches are aligned therefore the more similar
        vector should be the one having the same id.
        Batch1                                Batch2
        President of France                   French President
        President of US                       American President
        Given the question president of France in batch 1 the model will
        learn to predict french president comparing it with all the other
        vectors in batch 2
        """
        x2 = x2[:, 0]  # Take only first for now and don't care about negative
        batch_size = x1.size(0)
        if batch_size not in self.labels_cache:
            self.labels_cache[batch_size] = self._compute_labels(batch_size, x1.device)

        labels = self.labels_cache[batch_size]
        distances = dist_mat(x1, x2, self.dist)  # here we can use any pair-wise distance: euclid/cosine/ssd
        if batch_size == 1:
            return distances - distances

        log_probabilities = self.d_pred(distances)
        loss = self.kl_loss(log_probabilities, labels) * batch_size  # wasserstein?
        return loss

    def __str__(self):
        return self.__class__.__name__


def block_diag(*arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

    shapes = torch.tensor([a.shape for a in arrs])
    out = torch.zeros(torch.sum(shapes, dim=0).tolist(), dtype=arrs[0].dtype, device=arrs[0].device)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


class SDMLAllLoss(SDMLLoss):
    def __init__(self, smoothing_parameter=0.3, dist='ssd', d_pred='softmax', normalize=False, label_softmax=False):
        super().__init__(smoothing_parameter, dist, d_pred)
        self.label_softmax = label_softmax
        self.normalize = normalize

    def _compute_sp(self, b, n, device):
        m = torch.ones((n, n)) * (1.0 - self.smoothing_parameter) + torch.eye(n, n) * self.smoothing_parameter
        # mask = torch.eye(n, n).byte()
        # m.masked_fill_(mask, 1.0)
        m = block_diag(*([m] * b))
        m = m + self.smoothing_parameter
        return m.to(device)

    def __call__(self, s, p, n=None, sp_coo=None, p_coo=None, labels=None):
        B, N, D = p.size()
        _p = p.reshape(B * N, D)
        _n = n.reshape(B * N, D)

        if self.normalize:
            s = F.normalize(s, p=2, dim=1)
            _p = F.normalize(_p, p=2, dim=1)
            _n = F.normalize(_n, p=2, dim=1)
        s = s.repeat_interleave(repeats=N, dim=0)

        d_sp = dist_mat(s, _p, self.dist)
        d_sn = dist_mat(s, _n, self.dist)
        distances = torch.cat((d_sp, d_sn), dim=-1)

        if B * N not in self.labels_cache:
            self.labels_cache[B * N] = self._compute_sp(B, N, s.device)
        labels_sp = self.labels_cache[B * N]
        labels = torch.cat((labels_sp, torch.zeros_like(d_sn)), dim=-1)
        if self.label_softmax:
            labels = torch.softmax(labels, dim=1)
        log_probabilities = self.d_pred(distances)
        loss = self.kl_loss(log_probabilities, labels) * B  # wasserstein?
        return loss


class SDMLAllWithCooLoss(SDMLAllLoss):
    def __init__(
        self,
        smoothing_parameter=0.3,
        dist='ssd',
        d_pred='softmax',
        normalize=False,
        label_softmax=False,
        alpha=1.0,
        beta=1.0
    ):
        super().__init__(smoothing_parameter, dist, d_pred, normalize, label_softmax)
        self.beta = beta
        self.alpha = alpha

    def __call__(self, s, p, n=None, sp_coo=None, p_coo=None, labels=None):
        B, N, D = p.size()
        _p = p.reshape(B * N, D)
        _n = n.reshape(B * N, D)
        # _p_coo = p_coo.reshape(B * N, D)

        if self.normalize:
            s = F.normalize(s, p=2, dim=1)
            _p = F.normalize(_p, p=2, dim=1)
            _n = F.normalize(_n, p=2, dim=1)
        s = s.repeat_interleave(repeats=N, dim=0)

        # Session - pos/neg
        d_sp = dist_mat(s, _p, self.dist)
        d_sn = dist_mat(s, _n, self.dist)
        distances = torch.cat((d_sp, d_sn), dim=-1)
        if B * N not in self.labels_cache:
            self.labels_cache[B * N] = self._compute_sp(B, N, s.device)
        labels_sp = self.labels_cache[B * N]
        labels = torch.cat((labels_sp, torch.zeros_like(d_sn)), dim=-1)
        if self.label_softmax:
            labels = torch.softmax(labels, dim=1)
        log_probabilities = self.d_pred(distances)
        loss_s_pn = self.kl_loss(log_probabilities, labels) * B  # wasserstein?

        # Positive items
        pp_dist = dist_mat(_p, _p, self.dist)
        log_probabilities = self.d_pred(pp_dist)
        loss_p = self.kl_loss(log_probabilities, p_coo.float())

        # Negative items
        pn_dist = dist_mat(_p, _n, self.dist)
        log_probabilities = self.d_pred(pn_dist)
        loss_n = F.relu(pp_dist - pn_dist + 0.5).mean(dim=-1).sum()

        return loss_s_pn + self.alpha * loss_p + self.beta * loss_n


class TripletLoss(torch.nn.Module):
    def __init__(self, s_margin=1.0, i_margin=1.0, normalize=True, swap=False, pdist='euclidian', weighting=False):
        super().__init__()
        self.weighting = weighting
        self.normalize = normalize
        self.i_margin = i_margin
        self.s_margin = s_margin
        self.s_triplet_loss = torch.nn.TripletMarginLoss(
            margin=s_margin, p=2, swap=swap, reduction='none' if weighting else 'mean'
        )
        if pdist == 'euclidian':
            if self.normalize:
                self.dist = lambda x, y: torch.pairwise_distance(
                    F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1), p=2
                )
            else:
                self.dist = lambda x, y: torch.pairwise_distance(x, y, p=2)
        elif pdist == 'cos':
            if self.normalize:
                self.dist = lambda x, y: 1.0 - (F.normalize(x, p=2, dim=1) * F.normalize(y, p=2, dim=1)).sum(-1)
            else:
                self.dist = lambda x, y: 1.0 - (x * y).sum(-1)
        else:
            raise RuntimeError(f'Bad pdist function: {pdist}')

    def __call__(self, s, p, n, sp_coo=None, p_coo=None, labels=None):
        B, N, D = p.size()
        _p = p.reshape(B * N, D)
        _n = n.reshape(B * N, D)
        if self.normalize:
            s = F.normalize(s, p=2, dim=1)
            _p = F.normalize(_p, p=2, dim=1)
            _n = F.normalize(_n, p=2, dim=1)
        s = s.repeat_interleave(repeats=N, dim=0)
        loss = self.s_triplet_loss(s, _p, _n)
        if self.weighting:
            # last ones should receive more attention
            if N == 0:
                print('Number of positives examples is zero! Skipping weighting...')
                return loss.mean()
            w = torch.sqrt(1.0 / (1.0 + torch.arange(N, device=s.device)).repeat(B))
            # w = (5.0 - torch.sqrt(1.0 / (1.0 + torch.arange(N, device=s.device))).repeat(B))
            loss = (loss * w).mean()
        return loss

    def __str__(self):
        return self.__class__.__name__


"""s
Tensor triplet_margin_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative, double margin,
                           double p, double eps, bool swap, int64_t reduction) {
  auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
  auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);
  if (swap) {
    auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
    dist_neg = at::min(dist_neg, dist_swap);
  }
  auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
  return apply_loss_reduction(output, reduction);
}

Tensor margin_ranking_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  auto output =  (-target * (input1 - input2) + margin).clamp_min_(0);
  return apply_loss_reduction(output, reduction);
}
"""


class Contrastive2Loss(torch.nn.Module):
    def __init__(
        self, alpha=2, beta=0.5, ss_margin=.5, si_margin=.1, normalize=True, swap=False, pdist='cos', weighting=False
    ):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.swap = swap
        self.si_margin = si_margin
        self.ss_margin = ss_margin
        self.normalize = normalize
        self._call = 0

        if pdist == 'euclidian':
            if self.normalize:
                self.dist = lambda x, y: torch.pairwise_distance(
                    F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1), p=2
                )
            else:
                self.dist = lambda x, y: torch.pairwise_distance(x, y, p=2)
        elif pdist == 'cos':
            if self.normalize:
                self.dist = lambda x, y: 1.0 - (F.normalize(x, p=2, dim=1) * F.normalize(y, p=2, dim=1)).sum(-1)
            else:
                self.dist = lambda x, y: 1.0 - (x * y).sum(-1)
        else:
            raise RuntimeError(f'Bad pdist function: {pdist}')

    def __call__(self, s, p, n, sp_coo=None, p_coo=None, labels=None):
        self._call += 1
        B, N, D = p.size()
        _p = p.reshape(B * N, D)
        _n = n.reshape(B * N, D)
        if self.normalize:
            s = F.normalize(s, p=2, dim=1)
            _p = F.normalize(_p, p=2, dim=1)
            _n = F.normalize(_n, p=2, dim=1)

        # session-item
        sr = s.repeat_interleave(repeats=N, dim=0)
        # si_loss = self.s_triplet_loss(sr, _p, _n)

        dist_pos = self.dist(sr, _p)
        dist_neg = self.dist(sr, _n)
        # if self.swap:
        dist_pos_neg = self.dist(_p, _n)
        # dist_neg = at::min(dist_neg, dist_pos_neg)

        # session-session
        dist_s = dist_mat(s, s, self.dist)
        w = torch.sqrt(1.0 / (1.0 + torch.arange(N, device=s.device)).repeat(B))

        loss_sp = (dist_pos * w).clamp_min(0).mean()
        loss_sn = (self.si_margin - dist_neg).clamp_min(0).mean()
        # loss_si = (dist_pos - dist_neg + self.si_margin).clamp_min(0).mean()
        loss_pn = -1.0 * (w * dist_pos_neg).clamp_min(0).mean() / 1000.0
        loss_ss = (self.ss_margin - dist_s.fill_diagonal_(0)).clamp_min(0).mean()

        # tb.add_scalar('train/loss-si', loss_sp.item(), self._call)
        # tb.add_scalar('train/loss-si', loss_sn.item(), self._call)
        # tb.add_scalar('train/loss-pn', loss_pn.item(), self._call)
        # tb.add_scalar('train/loss-ss', loss_ss.item(), self._call)
        #
        # print(f"Loss sp: {loss_sp} sn: {loss_sn} pn: {loss_pn} ss: {loss_ss}")
        return loss_sp + loss_sn + self.alpha * loss_pn + self.beta * loss_ss

    def __str__(self):
        return self.__class__.__name__


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, pos_margin=0, neg_margin=1, normalize=True, avg_non_zero_only=True):
        super().__init__()
        self.avg_non_zero_only = avg_non_zero_only
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin
        self.normalize = normalize
        if self.normalize:
            self.dist = lambda x, y: (F.normalize(x, p=2, dim=1) * F.normalize(y, p=2, dim=1)).sum(-1)
        else:
            self.dist = lambda x, y: torch.pairwise_distance(x, y, p=2)

    def __call__(self, s, p, n, sp_coo=None, p_coo=None, labels=None):
        B, N, D = p.size()
        _p = p.reshape(B * N, D)
        _n = n.reshape(B * N, D)
        s = s.repeat_interleave(repeats=N, dim=0)
        pos_dist = self.dist(s, _p)
        neg_dist = self.dist(s, _n)
        ploss = torch.relu(pos_dist - self.pos_margin)
        nloss = torch.relu(self.neg_margin - neg_dist)
        return self._reduce(ploss) + self._reduce(nloss)

    def _reduce(self, loss):
        num_non_zero_pairs = (loss > 0).nonzero().size(0)
        if self.avg_non_zero_only:
            loss = torch.sum(loss) / (num_non_zero_pairs + 1e-16)
        else:
            loss = torch.mean(loss)
        return loss

    def __str__(self):
        return self.__class__.__name__


class BPRLoss(torch.nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
        if self.normalize:
            self.dist = lambda x, y: (F.normalize(x, p=2, dim=1) * F.normalize(y, p=2, dim=1)).sum(-1)
        else:
            self.dist = lambda x, y: (x * y).sum(-1)

    def __call__(self, s, p, n, sp_coo=None, p_coo=None, labels=None):
        B, N, D = p.size()
        _p = p.reshape(B * N, D)
        _n = n.reshape(B * N, D)
        s = s.repeat_interleave(repeats=N, dim=0)
        pos_pred = self.dist(s, _p)
        net_pred = self.dist(s, _n)
        loss = 1.0 - torch.sigmoid(pos_pred - net_pred)
        return loss.mean()

    def __str__(self):
        return self.__class__.__name__


def test_tf_compute_distances():
    import numpy as np
    import tensorflow as tf

    x1 = tf.constant([[1, 2], [2, 2]])
    x2 = tf.constant([[1, 2], [3, 1]])

    # extracting sizes expecting [batch_size, dim]
    assert x1.shape == x2.shape
    batch_size, dim = x1.shape
    # expanding both tensor form [batch_size, dim] to [batch_size, batch_size, dim]
    x1_ = tf.broadcast_to(tf.expand_dims(x1, 1), [batch_size, batch_size, dim])
    x2_ = tf.broadcast_to(tf.expand_dims(x2, 0), [batch_size, batch_size, dim])
    squared_diffs = (x1_ - x2_)**2
    # sum of squared differences distance
    r = tf.reduce_sum(squared_diffs, axis=2)

    assert np.array_equal(r.numpy(), np.array([[0, 5], [1, 2]]))


def squared_diffs(x, y):
    return ((x - y)**2).sum(axis=-1)


def ssd(x, y):
    assert x.size() == y.size()
    n, d = x.size()
    x = x.unsqueeze(1).expand(n, n, d)
    y = y.unsqueeze(0).expand(n, n, d)
    squared_diffs = (x - y)**2
    return squared_diffs.sum(axis=2)


def dist_mat(x, y, dist_func):
    assert x.size() == y.size()
    n, d = x.size()
    x = x.unsqueeze(1).expand(n, n, d)
    y = y.unsqueeze(0).expand(n, n, d)
    return dist_func(x, y)


def test_ssd():
    input1 = torch.randn(100, 128)
    input2 = torch.randn(100, 128)
    output = ssd(input1, input2)
    assert output.size() == (100, 100)


def test_ssd2():
    import numpy as np
    x = torch.tensor([[1, 2], [2, 2]])
    y = torch.tensor([[1, 2], [3, 1]])

    assert np.array_equal(ssd(x, y).numpy(), np.array([[0, 5], [1, 2]]))


def test_SDML():
    input1 = torch.randn(100, 128)
    input2 = torch.randn(100, 128)
    output = SDMLLoss()(input1, input2)
    assert output > 0.0
