import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from custom_transforms import edge_contour


def one_hot(index_tensor, cls_num):
    b, h, w = index_tensor.size()
    index_tensor = index_tensor.view(b, 1, h, w)
    one_hot_tensor = torch.cuda.FloatTensor(b, cls_num, h, w).zero_()
    one_hot_tensor = one_hot_tensor.cuda(index_tensor.get_device())
    target = one_hot_tensor.scatter_(1, index_tensor.long(), 1)

    return target


class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing=0.1, nclasses=6):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.nclasses = nclasses
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            target = one_hot(target, self.nclasses)
            logprobs = self.log_softmax(x)
            nll_loss = -logprobs * target

            nll_loss = nll_loss.sum(1)
            smooth_loss = -logprobs.mean(1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return nn.CrossEntropyLoss(x, target)


class Edge_weak_loss(nn.Module):
    def __init__(self):
        super(Edge_weak_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, scale_pred, target):
        edge = edge_contour(target).long()
        edge_loss = (torch.mul(self.ce_loss(scale_pred, target), torch.where(
            edge == 0, torch.tensor([1.]).cuda(), torch.tensor([2.0]).cuda()))).mean()

        return edge_loss


class Edge_loss(nn.Module):

    def __init__(self, ignore_index=255):
        super(Edge_loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # h, w = label.size(1), label.size(2)
        pos_num = torch.sum(label == 1, dtype=torch.float)
        neg_num = torch.sum(label == 0, dtype=torch.float)
        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)

        weights = torch.Tensor([weight_neg, weight_pos])
        edge_loss = F.cross_entropy(pred, label,
                                weights.cuda(), ignore_index=self.ignore_index)

        return edge_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super(CrossEntropyLoss, self).__init__()
        if weights is not None:
            weights = torch.from_numpy(np.array(weights)).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, weight=weights)

    def forward(self, prediction, label):
        loss = self.ce_loss(prediction, label)

        return loss


class CrossEntropyLoss_binary(nn.Module):
    def __init__(self, weights=None, binary_class=None):
        super(CrossEntropyLoss_binary, self).__init__()
        self.binary_class = binary_class
        if weights is not None:
            weights = torch.from_numpy(np.array(weights)).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, weight=weights)

    def forward(self, prediction, label):
        if self.binary_class is not None:
            label[label != self.binary_class] = 0
            label[label == self.binary_class] = 1
        loss = self.ce_loss(prediction, label)

        return loss


def One_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        # input_flat = pred[:, 1, :, :].contiguous().view(n, -1)
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = One_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


class IOULoss(nn.Module):
    def __init__(self,smooth):
        super(IOULoss,self).__init__()
        self.smooth = smooth

    def forward(self,y_pred,y_true,weight=None):
        axis = identify_axis(y_pred.shape)
        y_pred = nn.Softmax(dim=1)(y_pred)
        if weight is not None:
            weight = weight.to(y_pred.device)
        tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis,weight=weight)
        inter = tp
        union = tp + fp + fn
        iou = 1 - (inter + self.smooth) / (union + self.smooth)
        return iou.mean()


class SoftIoULoss(nn.Module):
    def __init__(self, weights=None):
        super(SoftIoULoss, self).__init__()
        if weights is not None:
            self.weights = torch.from_numpy(np.array(weights)).float().cuda()
    def forward(self, pred, target):
        # Old One
        # pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        loss = 1 - loss.mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        # input = torch.argmax(input, dim=1)
        # # 选择一个类别（例如，前景0 边界1）的概率作为预测结果
        # # 假设类别1是我们感兴趣的类别
        input_flat = input[:, 1, :, :].contiguous().view(N, -1)

        # input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    input: N C H W
    target: N H W

    """

    def __init__(self, weights=None, cls_num=None):
        super(MulticlassDiceLoss, self).__init__()
        self.weights = weights
        self.cls_num = cls_num

    def forward(self, input, target):

        target = one_hot(target, cls_num=self.cls_num)
        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if self.weights is not None:
                diceLoss *= self.weights[i]
            totalLoss += diceLoss

        return totalLoss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class Lovasz_loss(nn.Module):
    def __init__(self):
        super(Lovasz_loss, self).__init__()

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        from itertools import filterfalse as ifilterfalse
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def forward(self, probas, labels, classes='present', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        if per_image:
            loss = self.mean(
                self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss




def to_onehot(y_pred, y_true):
    shp_x = y_pred.shape
    shp_y = y_true.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(y_pred.shape, y_true.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y_true
        else:
            y_true = y_true.long()
            y_onehot = torch.zeros(shp_x, device=y_pred.device)
            y_onehot.scatter_(1, y_true, 1)
    return y_onehot

def get_tp_fp_fn_tn(net_output, gt, axes=None, square=False, weight=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    y_onehot = to_onehot(net_output, gt)

    if weight is None:
        weight = 1
    tp = net_output * y_onehot * weight
    fp = net_output * (1 - y_onehot) * weight
    fn = (1 - net_output) * y_onehot * weight
    tn = (1 - net_output) * (1 - y_onehot) * weight

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def identify_axis(shape):
    """
    Helper function to enable loss function to be flexibly used for
    both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
    """
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]
    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')




