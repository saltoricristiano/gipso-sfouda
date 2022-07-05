import torch
import sklearn.metrics as metrics
import numpy as np
import pdb


def filtered_accuracy(preds, gt):

    valid_idx = torch.logical_not(gt == 0)
    gt = gt[valid_idx] - 1
    preds = preds[valid_idx]

    acc = metrics.accuracy_score(gt.numpy(), preds.numpy())

    return acc


def confusion_matrix(scores, labels, num_classes=7):
        r"""
            Compute the confusion matrix of one batch
            Parameters
            ----------
            scores: torch.FloatTensor, shape (B?, C, N)
                raw scores for each class
            labels: torch.LongTensor, shape (B?, N)
                ground truth labels
            Returns
            -------
            confusion matrix of this batch
        """

        predictions = scores.data
        labels = labels.data

        conf_m = torch.zeros((num_classes, num_classes), dtype=torch.int32)

        for label in range(num_classes):
            for pred in range(num_classes):
                conf_m[label][pred] = torch.sum(
                    torch.logical_and(labels == label, predictions == pred))
        return conf_m


def iou_from_confusion(conf_m):
    per_class_iou = [conf_m[i][i] / (conf_m[i].sum() + conf_m[:, i].sum() - conf_m[i][i]) for i in range(conf_m.shape[0])]
    return per_class_iou, torch.hstack(per_class_iou).mean()
