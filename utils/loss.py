import torch

def _valid_idx(y_true, y_pred):
    nan_idx = torch.isnan(y_true) | torch.isnan(y_pred)
    y_true = y_true[~nan_idx]
    y_pred = y_pred[~nan_idx]
    return y_true, y_pred

def loss_NSE(y_true, y_pred):
    # y_true (lead, batch_size)
    # y_pred (lead, batch_size)

    y_true, y_pred = _valid_idx(y_true, y_pred)
    numerator = torch.sum((y_true - y_pred) ** 2, dim=0)
    denominator = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    return 1 - (numerator / denominator)

def loss_RMSE(y_true, y_pred):
    # y_true (lead, batch_size)
    # y_pred (lead, batch_size)
    
    y_true, y_pred = _valid_idx(y_true, y_pred)
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    return torch.sqrt(mse)

def loss_swi(p, pet, q, swi):
    # p, pet, q, swi: (lead, batch_size)
    # p, pet, q, swi = _valid_idx(p, pet)
    dswi = swi[1:] - swi[:-1]
    wb_residual = p[1:] - pet[1:] - q[1:] - dswi
    mse = torch.mean(wb_residual ** 2, dim=0)
    return torch.sqrt(mse)