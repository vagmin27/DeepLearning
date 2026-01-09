"""Losses for self-supervised pretext tasks in vision."""
import numpy as np
import torch
import torch.nn.functional as F


def classification_loss(y_hat, y):
    """
    Implements standard classification loss with cross-entropy.

    :param (torch.Tensor) y_hat:
    :param (torch.Tensor) y: ground truth
    :return: (torch.Tensor) loss
    """
    loss = F.cross_entropy(y_hat, y)
    return loss

