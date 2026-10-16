"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import cv2
from custom_transforms import edge_contour
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
        self.theta0 = 3
        self.theta = 5


    def Accuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return precision

    def meanPrecision(self):
        precision = self.Precision()
        mPrecision = np.nanmean(precision)
        return mPrecision

    def Recall(self):
        # Recall = (TP) / (TP + FN)
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return recall

    def meanRecall(self):
        recall = self.Recall()
        mRecall = np.nanmean(recall)
        return mRecall

    def F1(self):
        # 2*precision*recall / (precision + recall)
        f1 = 2 * self.Precision() * self.Recall() / (self.Precision() + self.Recall())
        return f1

    def meanF1(self):
        f1 = self.F1()
        mF1 = np.nanmean(f1)
        return mF1

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        IoU = self.IntersectionOverUnion()
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        iu = [i if not np.isnan(i) else 0.0 for i in iu]
        iu = np.array(iu)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Frequency_Weighted(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)

        return freq

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def BoundaryF1(self, pred, gt, device):
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
        pred = torch.softmax(pred, dim=1)
        # print("softmax", pred)
        # pred = pred.argmax(dim=1)

        # print("debug:", pred.shape, gt.shape)
        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)
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
        gt_b = gt_b.view(n, c, -1).to(device)
        pred_b = pred_b.view(n, c, -1).to(device)
        gt_b_ext = gt_b_ext.view(n, c, -1).to(device)
        pred_b_ext = pred_b_ext.view(n, c, -1).to(device)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)
        BF1 = torch.mean(BF1)
        return BF1


def one_hot(label, n_classes):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(n_classes, device=device)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


if __name__ == '__main__':
    # imgPredict = np.array([0, 0, 0, 1, 1, 1],
    #                       [0, 0, 0, 1, 1, 1],)
    # imgLabel = np.array([0, 0, 0, 1, 1, 1],
    #                       [0, 0, 0, 1, 1, 1],)
    device = torch.device(('cuda:{}').format(0))
    imgPredict = torch.rand(1, 3, 4, 4)
    imgPredict = imgPredict.to(device)
    print(imgPredict)  #

    imgLabel = torch.randint(0, 3, (1, 4, 4))
    imgLabel = imgLabel.to(device)
    print(imgLabel)
    print(imgPredict.shape) #
    print(imgLabel.shape)
    # exit()
    metric = SegmentationMetric(2)
    # metric.addBatch(imgPredict, imgLabel)
    # acc = metric.pixelAccuracy()
    # mIoU = metric.meanIntersectionOverUnion()
    # F1 = metric.F1()
    total = torch.zeros(1, device=device)
    for i in range(5):
        Boundary_F1 = metric.BoundaryF1(imgPredict, imgLabel, device)
        total += Boundary_F1

    total = total/5
    BF1 = [0]
    BF1[0] = BF1[0] + total.cpu().detach().numpy().tolist()[0]
    print("Boundary_F1:", BF1)