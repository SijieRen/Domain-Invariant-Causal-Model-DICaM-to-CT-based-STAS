import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn as nn

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count if self.count > 0 else (self.count + 1))

import numbers
class AUCMeter(object):
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.
    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """

    def __init__(self):
        super(AUCMeter, self).__init__()
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def update(self, output, target):
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy().reshape(-1)
        else:
            output = np.atleast_1d(np.asarray(output)).reshape(-1)
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy().reshape(-1)
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        else:
            target = np.atleast_1d(np.asarray(target)).reshape(-1)
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)

    def get_auc(self):
        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return (0.5, 0.0, 0.0)

        # sorting the arrays
        scores, sortind = torch.sort(torch.from_numpy(
            self.scores), dim=0, descending=True)
        scores = scores.numpy()
        sortind = sortind.numpy()

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        # calculating area under curve using trapezoidal rule
        n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return area #(area, tpr, fpr)
    # def get_sensi_speci(self):


# class AUCMeter(object):
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.score_labels = []
#         self.pos_count = 0
#         self.neg_count = 0
#
#     def update(self, pos_scores, target):
#         target = target.cpu().numpy()
#         self.pos_count += np.sum(target == 1)
#         self.neg_count += np.sum(target == 0)
#         n = target.shape[0]
#         for i in range(n):
#             self.score_labels.append((pos_scores[i], target[i]))
#
#     def get_auc(self):
#         if self.pos_count == 0 or self.neg_count == 0:
#             return 0
#         self.score_labels.sort()
#         pos_rank_sum = 0
#         for (i, (score, label)) in enumerate(self.score_labels):
#             if label == 1:
#                 pos_rank_sum += (i + 1)
#         # print(self.pos_count, self.neg_count, pos_rank_sum, (self.pos_count * (self.pos_count + 1)) * 0.5, self.pos_count * self.neg_count)
#         return 100.0 * (pos_rank_sum - (self.pos_count * (self.pos_count + 1)) * 0.5) / (self.pos_count * self.neg_count)

