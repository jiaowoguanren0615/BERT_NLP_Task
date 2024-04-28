import torch
import torch.nn as nn


# Define RMSE Loss
def loss_function(outputs, targets):
    """
    This is the loss function for this task
    """
    return torch.sqrt(nn.MSELoss()(outputs, targets))