"""

Module contains classes and functions necessary for the recommender system
evaluation. All of them are well-known evaluation metrics. However, here there
are prepared for the usage with the session-aware recommender systems.

"""

import logging
from collections import Counter

import numpy as np

from rec.utils import current_milli_time
from tqdm import tqdm

ROUND_FLOAT = 4
EPSILON = 10e-8

logger = logging.getLogger(__name__)


class Evaluation(object):
    """
    Main class for the evaluation metric computation. 
    """
    def __init__(self):
        super(Evaluation, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute(self, ground_truth, predictions, sessions):
        """
        Compute the metric using the predictions and known ground truth values.
        
        Args:
            ground_truth (list[list[int]]): list of the ground truth items ids
            predictions (list[list[int]]): list of recommended items ids
            sessions (list[Sessions]): list of sessions used for evaluation. 
             Used to compute some of the metrics, i.e. those which breaks the results
             due to the session length or used events.

        Returns: metric, which structure depends on the calculation

        """
        assert len(ground_truth) == len(predictions)
        if sessions is not None:
            assert len(ground_truth) == len(sessions)


class PrecisionRecall(Evaluation):
    """
    Precision and Recall calculated as known from the Information Retrieval.
    """
    def compute(self, ground_truth, predictions, sessions=None):
        # super(PrecisionRecall, self).compute(ground_truth, predictions, sessions)
        assert len(ground_truth) > 0
        if len(predictions) == 0:
            return 0, 0
        hits = list(set(ground_truth) & set(predictions))
        precision = len(hits) / float(len(predictions))
        recall = len(hits) / float(len(ground_truth))
        return precision, recall


class PrecisionRecallAtN(Evaluation):
    """
    Precision and Recall with respect to ranking and up to N first items.
    Additionally, if the sessions are given, per session length results are returned.
    """
    def __init__(self, top_n, max_events=30):
        super(PrecisionRecallAtN, self).__init__()
        self.max_events = max_events
        self.top_n = top_n

    def compute(self, ground_truth, predictions, sessions=None):
        super(PrecisionRecallAtN, self).compute(ground_truth, predictions, sessions)
        n = len(predictions)
        p = np.zeros(n)
        r = np.zeros(n)

        hits_num = 0
        all_gt = 0
        all_pred = 0

        # per session length
        if sessions is not None:
            sl_hits_num = np.zeros(self.max_events)
            sl_gt = np.zeros(self.max_events)
            sl_pred = np.zeros(self.max_events)

        for i in range(0, n):
            gt = ground_truth[i][:self.top_n]
            pred = predictions[i][:self.top_n]
            n_hits = len(list(set(gt) & set(pred)))
            n_gt = len(gt)
            n_pred = len(pred)

            # global
            hits_num += n_hits
            all_gt += n_gt
            all_pred += n_pred
            if n_pred > 0:
                p[i] = n_hits / float(n_pred)
            r[i] = n_hits / float(n_gt)

            # per session length
            if sessions is not None:
                sl = sessions[i].events_num()
                if sl < self.max_events:
                    sl_hits_num[sl] += n_hits
                    sl_gt[sl] += n_gt
                    sl_pred[sl] += n_pred

        result = dict(prec=round(p.mean(), ROUND_FLOAT), rec=round(r.mean(), ROUND_FLOAT))
        logger.info("REC@{}: {}".format(self.top_n, round(r.mean(), ROUND_FLOAT)))
        if sessions is not None:
            sl_recall = sl_hits_num / (sl_gt + 1e-10)
            # print 'Session-Length Recall: {}'.format(sl_recall)
            result.update(sl_rec=sl_recall.tolist())

        return result


class HitRateAtN(Evaluation):
    """
    Hit rate with respect to ranking and up to N first items.
    Additionally, if the sessions are given, per session length results are returned.
    """
    def __init__(self, top_n, max_events=30, single_next_item_only=False):
        super(HitRateAtN, self).__init__()
        self.max_events = max_events
        self.top_n = top_n
        self.single_next_item_only = single_next_item_only

    def compute(self, ground_truth, predictions, sessions=None):
        super(HitRateAtN, self).compute(ground_truth, predictions, sessions)
        n = len(predictions)
        p = np.zeros(n)
        r = np.zeros(n)

        hits_num = 0

        # per session length
        if sessions is not None:
            sl_hits_num = np.zeros(self.max_events)
            sl_gt = np.zeros(self.max_events)

        for i in range(0, n):
            gt = ground_truth[i][:1] if self.single_next_item_only else ground_truth[i]

            pred = predictions[i][:self.top_n]
            hit = int(len(list(set(gt) & set(pred))) > 0)

            # global
            hits_num += hit
            r[i] = hit

            # per session length
            if sessions is not None:
                sl = sessions[i].events_num()
                if sl < self.max_events:
                    sl_hits_num[sl] += hit
                    sl_gt[sl] += 1

        result = dict(hr=round(r.mean(), ROUND_FLOAT))
        logger.info("HR@{}: {}".format(self.top_n, round(r.mean(), ROUND_FLOAT)))
        if sessions is not None:
            sl_hr = sl_hits_num / (sl_gt + 1e-10)
            result.update(sl_hr=sl_hr.tolist())

        return result


class PrecisionRecallUpToN(Evaluation):
    """
    Precission and Recall calculation up to top-n items at once. 
    The calculations are very similar to the previous class, but here
    the results are also break by the different top-n values. This 
    gives more a more insight information in case of choosing the N 
    value for a new application.
    """
    def __init__(self, top_n=20, max_events=30):
        super(PrecisionRecallUpToN, self).__init__()
        self.max_events = max_events
        self.top_n = top_n

    def compute(self, ground_truth, predictions, sessions=None):
        super(PrecisionRecallUpToN, self).compute(ground_truth, predictions, sessions)

        test_num = len(predictions)

        hits = np.zeros(self.top_n)
        gt = np.zeros(self.top_n)
        pred = np.zeros(self.top_n)

        sl_hits_num = np.zeros((self.top_n, self.max_events))
        sl_gt = np.zeros((self.top_n, self.max_events))
        sl_pred = np.zeros((self.top_n, self.max_events))

        for i in range(0, test_num):
            t_gt = ground_truth[i][:self.top_n]
            t_pred = predictions[i][:self.top_n]

            # compute stats per every k
            for k in range(self.top_n):
                gt_k = t_gt[:k + 1]
                pred_k = t_pred[:k + 1]
                n_hits = len(list(set(gt_k) & set(pred_k)))
                n_gt = len(gt_k)
                n_pred = len(pred_k)

                # collect stats
                hits[k] += n_hits
                gt[k] += n_gt
                pred[k] += n_pred

                # per session length
                if sessions is not None:
                    sl = sessions[i].events_num()
                    if sl < self.max_events:
                        sl_hits_num[k, sl] += n_hits
                        sl_gt[k, sl] += n_gt
                        sl_pred[k, sl] += n_pred

        rec = np.around(hits / (gt + EPSILON), ROUND_FLOAT)
        prec = np.around(hits / (pred + EPSILON), ROUND_FLOAT)

        result = []

        for k in range(1, self.top_n):
            r = dict(prec=round(prec[k], ROUND_FLOAT), rec=round(rec[k], ROUND_FLOAT))
            if sessions is not None:
                sl_recall = sl_hits_num[k] / (sl_gt[k] + 1e-10)
                # self.logger.debug('Session-Length Recall@{}: {}'.format(k, sl_recall))
                r.update(sl_rec=np.around(sl_recall, ROUND_FLOAT).tolist())
            result.append(r)
        return {'rec@k': result}


class MeanAveragePrecisionAtN(Evaluation):
    def __init__(self, top_n=20, max_events=30):
        super(MeanAveragePrecisionAtN, self).__init__()
        self.max_events = max_events
        self.top_n = top_n

    def compute(self, ground_truth, predictions, sessions=None):
        super(MeanAveragePrecisionAtN, self).compute(ground_truth, predictions, sessions)

        test_num = len(predictions)
        map_k = 0.0
        n = 0

        for i in range(0, test_num):
            t_gt = ground_truth[i][:self.top_n]
            t_pred = predictions[i][:self.top_n]

            ap = 0.0

            if len(t_pred) > 0:
                last_recall = 0.0
                for k in range(self.top_n):
                    pred_k = t_pred[:k + 1]
                    rec = len(set(t_gt) & set(pred_k)) / len(t_gt)
                    prec = len(set(t_gt) & set(pred_k)) / self.top_n

                    ap += prec * (rec - last_recall)
                    last_recall = rec
                map_k += ap
                n += 1

        map_k = map_k / n
        return {'map': map_k}


class MeanReciprocalRank(Evaluation):
    """
    Mean Reciprocal Rank (MRR) for the recommendation ranking evaluation.
    """
    def __init__(self, max_events=30, single_next_item_only=False):
        super(MeanReciprocalRank, self).__init__()
        self.max_events = max_events
        self.single_next_item_only = single_next_item_only

    def compute(self, ground_truth, predictions, sessions=None):
        super(MeanReciprocalRank, self).compute(ground_truth, predictions, sessions)
        n = len(predictions)
        if n == 0:
            return .0

        rr_sum = .0
        # session length mrr
        sl_rr_sum = np.zeros(self.max_events)
        sl_rr_num = np.zeros(self.max_events)

        for i in range(n):
            if sessions is not None:
                sl = sessions[i].events_num()
                if sl < self.max_events:
                    sl_rr_num[sl] += 1

            gt = ground_truth[i][:1] if self.single_next_item_only else ground_truth[i]
            for p in range(len(predictions[i])):
                if predictions[i][p] in gt:
                    rr = 1.0 / (p + 1)
                    rr_sum += rr
                    if sessions is not None and sl < self.max_events:
                        sl_rr_sum[sl] += rr
                    break
        result = dict(mrr=round(rr_sum / n, ROUND_FLOAT))
        if sessions is not None:
            sl_mrr = sl_rr_sum / (sl_rr_num + 1e-10)
            result.update(sl_mrr=np.around(sl_mrr, ROUND_FLOAT).tolist())
        return result


class MeanReciprocalRankAtN(Evaluation):
    """
    Mean Reciprocal Rank (MRR) for the recommendation ranking evaluation.
    """
    def __init__(self, top_n=20, max_events=30, single_next_item_only=False):
        super(MeanReciprocalRankAtN, self).__init__()
        self.max_events = max_events
        self.single_next_item_only = single_next_item_only
        self.top_n = top_n

    def compute(self, ground_truth, predictions, sessions=None):
        super(MeanReciprocalRankAtN, self).compute(ground_truth, predictions, sessions)
        n = len(predictions)
        if n == 0:
            return .0

        rr_sum = .0
        # session length mrr
        sl_rr_sum = np.zeros(self.max_events)
        sl_rr_num = np.zeros(self.max_events)

        for i in range(n):
            if sessions is not None:
                sl = sessions[i].events_num()
                if sl < self.max_events:
                    sl_rr_num[sl] += 1

            gt = ground_truth[i][:1] if self.single_next_item_only else ground_truth[i]
            for p in range(len(predictions[i][:self.top_n])):
                if predictions[i][p] in gt:
                    rr = 1.0 / (p + 1)
                    rr_sum += rr
                    if sessions is not None and sl < self.max_events:
                        sl_rr_sum[sl] += rr
                    break
        result = dict(mrr=round(rr_sum / n, ROUND_FLOAT))
        if sessions is not None:
            sl_mrr = sl_rr_sum / (sl_rr_num + 1e-10)
            result.update(sl_mrr=np.around(sl_mrr, ROUND_FLOAT).tolist())
        return result


class MeanReciprocalRankUpToN(Evaluation):
    def __init__(self, n=20, max_events=30, single_next_item_only=False):
        super(MeanReciprocalRankUpToN, self).__init__()
        self.n = n
        self.max_events = max_events
        self.single_next_item_only = single_next_item_only

    def compute(self, ground_truth, predictions, sessions=None):
        super(MeanReciprocalRankUpToN, self).compute(ground_truth, predictions, sessions)
        test_num = len(predictions)
        if test_num == 0:
            return 0.0

        rr_sum = np.zeros(self.n, dtype='float32')
        rr_num = np.zeros(self.n, dtype='float32')
        # session length mrr
        sl_rr_sum = np.zeros((self.n, self.max_events))
        sl_rr_num = np.zeros((self.n, self.max_events))

        for i in range(test_num):
            gt = ground_truth[i][:1] if self.single_next_item_only else ground_truth[i]
            for k in range(1, self.n):
                k_predictions = predictions[i][:k]
                rr_num[k] += 1
                if sessions is not None:
                    sl = sessions[i].events_num()
                    if sl < self.max_events:
                        sl_rr_num[k, sl] += 1
                for r in range(len(k_predictions)):
                    if k_predictions[r] in gt:
                        rr = 1.0 / (r + 1)
                        rr_sum[k] += rr
                        if sessions is not None and sl < self.max_events:
                            sl_rr_sum[k, sl] += rr
                        break
        mrr = rr_sum / (rr_num + 1e-10)

        result = []
        for k in range(1, self.n):
            r = dict(mrr=round(float(mrr[k]), ROUND_FLOAT))
            if sessions is not None:
                sl_mrr = sl_rr_sum[k] / (sl_rr_num[k] + 1e-10)
                # self.logger.debug('Session-Length MRR@{}: {}'.format(k, sl_mrr))
                r.update(sl_mrr=np.around(sl_mrr, ROUND_FLOAT).tolist())
            result.append(r)
        return {'mrr@k': result}


def evaluate_recommender(recommender, sessions, ground_truth, evaluation_measures, top_n, batch_size):
    test_start = current_milli_time()
    predictions = []
    test_batch_number = 0
    batch_slice = slice(test_batch_number * batch_size, (test_batch_number + 1) * batch_size)
    with tqdm(total=len(sessions)) as pbar:
        while batch_slice.start < len(sessions):
            test_batch = sessions[batch_slice]
            predictions.extend(recommender.predict(test_batch, top_n))
            test_batch_number += 1
            batch_slice = slice(test_batch_number * batch_size, (test_batch_number + 1) * batch_size)
            pbar.update(batch_slice.start - pbar.n)
    predict_sec = (current_milli_time() - test_start) / 1000
    logger.info("Test prediction time: {} sec.".format(predict_sec))
    logger.info('Eveluation - test sessions num: {test_num}'.format(test_num=len(sessions)))
    gt_items_num = sum([len(i) for i in ground_truth])
    logger.info('Evaluation - ground truth items num: {gt_items}'.format(gt_items=gt_items_num))
    pred_items_num = sum([len(i) for i in predictions])
    logger.info('Evaluation - predictions items num: {p_items}'.format(p_items=pred_items_num))
    items_pred_count = Counter()
    for pred in predictions:
        items_pred_count.update(pred)
    logger.info('Unique items predicted: {}'.format(len(items_pred_count)))
    eval_start = current_milli_time()
    evaluation_results = dict()
    for eval_measure in evaluation_measures:
        assert isinstance(eval_measure, Evaluation)
        result = eval_measure.compute(ground_truth, predictions, sessions)
        evaluation_results.update(result)
    eval_sec = (current_milli_time() - eval_start) / 1000
    logger.info("Evaluation time: {} sec.".format(eval_sec))
    return evaluation_results, predictions, predict_sec, eval_sec
