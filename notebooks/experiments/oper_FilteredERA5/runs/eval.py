#%% Imports: Python Libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random

import os
import gc
import time
import json
import tqdm
import sys
import argparse
# from joblib import Parallel, delayed

import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns

#%% Default Paths
# import configparser
# cfg = configparser.ConfigParser()
# cfg.optionxform = str
# PATH_DEFAULTS = '/data/sarth/operational-hydrologic-emulators/assets/defaults.ini'
# cfg.read(PATH_DEFAULTS)
# cfg = {s: dict(cfg.items(s)) for s in cfg.sections()}
# PATHS = cfg['PATHS']
# del cfg

PATHS = {}
PATHS['root'] = '/data/sarth/operational-hydrologic-emulators'
PATHS['datasets'] = os.path.join(PATHS['root'], 'datasets')
PATHS['experiments'] = os.path.join(PATHS['root'], 'experiments')

#%% Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--params', type=str, required=True, help='Path to model params file')
parser.add_argument('--cuda_num', type=int, default=0, help='CUDA device number')
parser.add_argument('--dataset', type=str, default='CAMELS-US_HUCAll_lumped', help='Dataset name')
parser.add_argument('--prediction_save_path', type=str, default='predictions.pt', help='Path to save predictions')
args = parser.parse_args()
PATH_CONFIG = args.config
PATH_PARAMS = args.params
arg_cuda_num = args.cuda_num
dataset_name = args.dataset
prediction_save_path = args.prediction_save_path
# Example usage:
# python eval.py --config config.json --params model_params.json --cuda_num 0 --dataset CAMELS-US_HUCAll_lumped --prediction_save_path predictions.pt

#%% Model Parameters
# PATH_PARAMS = os.path.join(PATHS['root'], 'workdir/projects/analysis_lumped/runs/devp/model_params.json')
with open(PATH_PARAMS, 'r') as f:
    model_params = json.load(f)
# Post-compute dependent params
model_params['dim_in_encoder_lag'] = (
    model_params['dim_out_embedding_lag_dynamic'] +
    model_params['dim_out_embedding_lag_static'] + 4
)
model_params['dim_in_decoder_lead'] = (
    model_params['dim_out_embedding_lead_dynamic'] +
    model_params['dim_out_embedding_lead_static'] + 4
)

#%% Configurations
# PATH_CONFIG = os.path.join(PATHS['root'], 'workdir/projects/analysis_lumped/runs/devp/config.json')
with open(PATH_CONFIG, 'r') as f:
    cfg = json.load(f)
cfg['cuda_num'] = arg_cuda_num
#%% Setup for Experiment
PATH_DATASET = PATHS['datasets']
PATH_ANALYSIS = os.path.join(PATHS['experiments'], cfg['run_name'])
PATH_PREDICTIONS = os.path.join(PATH_ANALYSIS, 'predictions')
PATH_MODELS_BEST = os.path.join(PATH_ANALYSIS, 'weights_best', 'model_best.pt')

#%% Import Custom Modules
PATH_UTILS = os.path.join(PATHS['experiments'], 'utils')
if PATH_UTILS not in sys.path:
    sys.path.append(PATH_UTILS)

PATH_RUN = os.path.join(PATH_ANALYSIS, 'runs')
if PATH_RUN not in sys.path:
    sys.path.append(PATH_RUN)

# Reproducibility
from set_random_seeds import seed_everything
seed_everything(cfg['seed'])

# Metrics
from metrics import NSE, KGE, PBIAS, correlation, RMSE

#%% Data Loader
from dataloaders import LumpedDataLoader
loader = LumpedDataLoader(
    PATH_DATASET=os.path.join(PATH_DATASET, dataset_name),
)
loader.get_shapes()
loader._set_time_period(
    start_date=cfg['val_start_date'],
    end_date=cfg['val_end_date'],
    lag=365,
    lead=10,
)

device = torch.device(f"cuda:{cfg['cuda_num']}" if torch.cuda.is_available() else "cpu")

from models import RainfallRunoff
model = RainfallRunoff(device, model_params).to(device)

#%% Evaluation Loop
def evaluate(model, loader, device):
    model.eval()
    with torch.no_grad():
        pred_lst = []
        true_lst = []
        eval_indices = loader.available_indices
        for sample_idx in tqdm.tqdm(sorted(eval_indices), desc='eval'):
            sample, available = loader.get_sample(sample_idx, verbose=False)
            
            pred_dict = model(sample, available)
            y_pred = pred_dict['Q']  # [lead, batch_size, 1]
            y_pred = y_pred[:, :, 0]  # [lead, batch_size]

            y_true = sample['y_sim'].to(device)[:, :, 0]  # [lead, batch_size]

            pred_lst.append(y_pred.cpu())
            true_lst.append(y_true.cpu())
        
        pred_tensor = torch.stack(pred_lst, dim=0) # (samples, lead, batch_size)
        true_tensor = torch.stack(true_lst, dim=0) # (samples, lead, batch_size)
    
    # # Compute Metrics
    # def compute_metric(pred_tensor, true_tensor, metric_func):
    #     # pred_tensor: (samples, lead, batch_size)
    #     # true_tensor: (samples, lead, batch_size)
    #     metric_values = []
    #     for lead_idx in range(pred_tensor.shape[1]):
    #         pred_lead = pred_tensor[:, lead_idx, :]
    #         true_lead = true_tensor[:, lead_idx, :]
    #         metric_lead = [metric_func(true_lead[:, i], pred_lead[:, i]) for i in range(pred_lead.shape[1])]
    #         metric_lead = [round(metric_value.item(), 2) for metric_value in metric_lead]
    #         metric_values.append(metric_lead)
    #     return metric_values
    
    # NSE_values = compute_metric(pred_tensor, true_tensor, NSE)
    # NSE_nanmedian = [float(np.nanmedian(v)) for v in NSE_values]
    # print(f"NSE: {NSE_nanmedian}")

    # KGE_values = compute_metric(pred_tensor, true_tensor, KGE)
    # KGE_nanmedian = [float(np.nanmedian(v)) for v in KGE_values]
    # print(f"KGE: {KGE_nanmedian}")

    # RMSE_values = compute_metric(pred_tensor, true_tensor, RMSE)
    # RMSE_nanmedian = [float(np.nanmedian(v)) for v in RMSE_values]
    # print(f"RMSE: {RMSE_nanmedian}")

    # PBIAS_values = compute_metric(pred_tensor, true_tensor, PBIAS)
    # PBIAS_nanmedian = [float(np.nanmedian(v)) for v in PBIAS_values]
    # print(f"PBIAS: {PBIAS_nanmedian}")

    # corr_values = compute_metric(pred_tensor, true_tensor, correlation)
    # corr_nanmedian = [float(np.nanmedian(v)) for v in corr_values]
    # print(f"Correlation: {corr_nanmedian}")

    return {
        # 'NSE': NSE_values,
        # 'KGE': KGE_values,
        # 'RMSE': RMSE_values,
        # 'PBIAS': PBIAS_values,
        # 'Correlation': corr_values,
        'pred_tensor': pred_tensor,
        'true_tensor': true_tensor,
    }

#%% Execution
# model.load_state_dict(torch.load(PATH_MODELS_BEST))
weights = torch.load(PATH_MODELS_BEST, map_location=device)
model.load_state_dict(weights)

metrics = evaluate(model, loader, device)
pred_tensor = metrics['pred_tensor']
true_tensor = metrics['true_tensor']
torch.save({
    'pred_tensor': pred_tensor,
    'true_tensor': true_tensor,
}, os.path.join(PATH_PREDICTIONS, prediction_save_path))