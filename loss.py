import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import cv2

class pos_loss(nn.Module):
    def __init__(self):
        super(pos_loss, self).__init__()

    def forward(self, xy_pos, gt):
        loss = torch.nn.functional.mse_loss(xy_pos, gt)

        return loss
