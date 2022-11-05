import torch
import torch.nn as nn
import torch.nn.functional as F

def L1_loss(output, target):
    loss_func = nn.L1Loss()
    return loss_func(output, target)


def MSE_loss(output, target):
    loss_func = nn.MSELoss()
    return loss_func(output, target)


def BCEWithLogitsLoss(output, target):
    loss_func = nn.BCEWithLogitsLoss()
    return loss_func(output, target)


def RMSE_loss(output, target):
    loss_func = nn.MSELoss()
    return torch.sqrt(loss_func(output, target))


loss_config = {
    "l1": L1_loss,
    "mse": MSE_loss,
    "bce": BCEWithLogitsLoss,
    "rmse": RMSE_loss,
}
