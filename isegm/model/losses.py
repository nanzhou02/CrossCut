import numpy as np
import jittor
import jittor.nn as nn
from isegm.utils import misc

class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def execute(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = jittor.sigmoid(pred)

        alpha = jittor.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = jittor.where(sample_weight, 1.0 - jittor.abs(label - pred), jittor.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = jittor.sum(sample_weight, dims=(-2, -1), keepdims=True)
        beta_sum = jittor.sum(beta, dims=(-2, -1), keepdims=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = jittor.clamp_max(beta, self._max_mult)

        with jittor.no_grad():
            ignore_area = jittor.sum(label == self._ignore_label, dims=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = jittor.mean(mult, dims=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                # beta_pmax, _ = jittor.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = jittor.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        # loss = -alpha * beta * jittor.log(jittor.min(pt + self._eps, jittor.ones(1, dtype=jittor.float).to(pt.device)))
        loss = -alpha * beta * jittor.log(jittor.minimum(pt + self._eps, jittor.ones(1, dtype=jittor.float)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = jittor.sum(sample_weight, dims=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = jittor.sum(loss, dims=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = jittor.sum(loss, dims=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def execute(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = jittor.sigmoid(pred)

        alpha = jittor.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = jittor.where(sample_weight, 1.0 - jittor.abs(label - pred), jittor.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * jittor.log(jittor.min(pt + self._eps, jittor.ones(1, dtype=jittor.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = jittor.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = jittor.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = jittor.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def execute(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = jittor.sigmoid(pred)

        loss = 1.0 - jittor.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (jittor.sum(jittor.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def execute(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = jittor.where(sample_weight, label, jittor.zeros_like(label))

        if not self._from_sigmoid:
            loss = jittor.relu(pred) - pred * label + nn.softplus(-jittor.abs(pred))
        else:
            eps = 1e-12
            loss = -(jittor.log(pred + eps) * label
                     + jittor.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return jittor.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class BinaryDiceLoss(nn.Module):
    """ Dice Loss for binary segmentation
    """

    def execute(self, pred, label):
        batchsize = pred.size(0)

        # convert probability to binary label using maximum probability
        input_pred, input_label = pred.max(1)
        input_pred *= input_label.float()

        # convert to floats
        input_pred = input_pred.float()
        target_label = label.float()

        # convert to 1D
        input_pred = input_pred.view(batchsize, -1)
        target_label = target_label.view(batchsize, -1)

        # compute dice score
        intersect = jittor.sum(input_pred * target_label, 1)
        input_area = jittor.sum(input_pred * input_pred, 1)
        target_area = jittor.sum(target_label * target_label, 1)

        sum = input_area + target_area
        epsilon = jittor.tensor(1e-6)

        # batch dice loss and ignore dice loss where target area = 0
        batch_loss = jittor.tensor(1.0) - (jittor.tensor(2.0) * intersect + epsilon) / (sum + epsilon)
        loss = batch_loss.mean()

        return loss