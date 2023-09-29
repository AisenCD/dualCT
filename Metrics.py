import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from skimage import filters
import numpy as np


class SegmenationMetrics(object):
    def __init__(self):
        pass

    def IOU(self, all_true, all_pred):
        N = all_true.shape[0]
        iou = np.zeros([N, 4])
        for i in range(N):
            y_true = all_true[i, ...]
            y_pred = all_pred[i, ...]
            for j in range(iou.shape[1]):
                pred = (y_pred == j) * 1.0
                true = (y_true == j) * 1.0
                iou[i, j] = Jaccard_coefficient(pred.flatten(), true.flatten())

        return np.mean(iou, axis=0)

    def DICE(self, all_true, all_pred):
        N = all_true.shape[0]
        dice = np.zeros([N, 4])
        for i in range(N):
            y_true = all_true[i, ...]
            y_pred = all_pred[i, ...]
            for j in range(dice.shape[1]):
                pred = (y_pred == j) * 1.0
                true = (y_true == j) * 1.0
                # dice[i, j] = Dice_coefficient(pred.flatten(), true.flatten())
                dice[i, j] = Dice_coefficient(pred.flatten().cpu().numpy(), true.flatten().cpu().numpy())

        return np.mean(dice, axis=0)


    def ClassifyMetric(self, all_true, all_pred):
        N = all_true.shape[0]
        acc = np.zeros([N, 1])

        for i in range(N):
            y_true = all_true[i, ...]
            y_pred = all_pred[i, ...]
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            # acc[i] = np.mean(y_true == y_pred)
            acc[i] = np.mean(y_true.cpu().numpy() == y_pred.cpu().numpy())

        return acc.mean()


def Jaccard_coefficient(y_true, y_pred):
    # y_true = y_true.flatten()
    # y_pred = y_pred.flatten()
    overlap = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - overlap
    iou = (overlap + 1e-7) / (union + 1e-7)

    return iou


def Dice_coefficient(y_true, y_pred):
    # y_true = y_true.flatten()
    # y_pred = y_pred.flatten()
    overlap = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2*overlap + 1e-7) / (union + 1e-7)

    return dice