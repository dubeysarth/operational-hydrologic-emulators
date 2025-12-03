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
import configparser
cfg = configparser.ConfigParser()
cfg.optionxform = str
try:
    PATH_DEFAULTS = '/home/iitgn/Desktop/datadir/assets/defaults.ini'
    cfg.read(PATH_DEFAULTS)
except:
    PATH_DEFAULTS = '/home/sarth/rootdir/datadir/assets/defaults.ini'
    cfg.read(PATH_DEFAULTS)
cfg = {s: dict(cfg.items(s)) for s in cfg.sections()}
PATHS = cfg['PATHS']
del cfg

#%% Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--params', type=str, required=True, help='Path to model params file')
parser.add_argument('--cuda_num', type=int, default=0, help='CUDA device number')
args = parser.parse_args()
PATH_CONFIG = args.config
PATH_PARAMS = args.params
arg_cuda_num = args.cuda_num
# Example usage:
# python train.py --config="runs/template/config.json" --params="runs/template/model_params.json"


#%% Model Parameters
# PATH_PARAMS = os.path.join(PATHS['root'], 'workdir/projects/analysis_lumped/runs/devp/model_params.json')
with open(PATH_PARAMS, 'r') as f:
    model_params = json.load(f)
# Post-compute dependent params
model_params['dim_in_encoder_lag'] = (
    model_params['dim_out_embedding_lag_dynamic']*3 +
    model_params['dim_out_embedding_lag_static'] + 4 + 3
)
model_params['dim_in_decoder_lead'] = (
    model_params['dim_out_embedding_lead_dynamic'] +
    model_params['dim_out_embedding_lead_static'] + 4 + 1
)

#%% Configurations
# PATH_CONFIG = os.path.join(PATHS['root'], 'workdir/projects/analysis_lumped/runs/devp/config.json')
with open(PATH_CONFIG, 'r') as f:
    cfg = json.load(f)
cfg['cuda_num'] = arg_cuda_num
#%% Setup for Experiment
PATH_DATASET = os.path.join(PATHS['datasets'], 'batched_catchments')

PATH_ANALYSIS = os.path.join(PATHS['projects'], cfg['analysis_name'])
PATH_EXPERIMENTS = os.path.join(PATHS['experiments'], cfg['analysis_name'])

PATH_LOGS = os.path.join(PATH_EXPERIMENTS, 'results', cfg['run_name'], 'logs')
PATH_CHECKPOINTS = os.path.join(PATH_EXPERIMENTS, 'results', cfg['run_name'], 'checkpoints')
PATH_MODELS_EPOCH = os.path.join(PATH_EXPERIMENTS, 'results', cfg['run_name'], 'models_epoch')
PATH_MODELS_BEST = os.path.join(PATH_EXPERIMENTS, 'results', cfg['run_name'], 'models_best')
PATH_PLOTS = os.path.join(PATH_EXPERIMENTS, 'results', cfg['run_name'], 'plots')
PATH_PREDICTIONS = os.path.join(PATH_EXPERIMENTS, 'results', cfg['run_name'], 'predictions')

for path in [PATH_LOGS, PATH_CHECKPOINTS, PATH_MODELS_EPOCH, PATH_MODELS_BEST, PATH_PLOTS, PATH_PREDICTIONS]:
    os.makedirs(path, exist_ok=True)

logfile_epoch = os.path.join(PATH_LOGS, 'log_epoch.txt')
if cfg['mode'] == 'train':
    with open(logfile_epoch, 'w') as f:
        f.write('')
elif cfg['mode'] == 'resume':
    if not os.path.exists(logfile_epoch):
        with open(logfile_epoch, 'w') as f:
            f.write('')

def logprint(content, logfile=None, logonly=False):
    if not logonly:
        print(content)
    if logfile is not None:
        with open(logfile, 'a') as f:
            # f.write(content + '\n')
            timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
            f.write(f"{timestamp} {content}\n")

#%% Import Custom Modules
PATH_UTILS = os.path.join(PATH_ANALYSIS, 'utils')
if PATH_UTILS not in sys.path:
    sys.path.append(PATH_UTILS)

PATH_RUN = os.path.join(PATH_ANALYSIS, 'runs', cfg['run_name'])
if PATH_RUN not in sys.path:
    sys.path.append(PATH_RUN)

# Reproducibility
from set_random_seeds import seed_everything
seed_everything(cfg['seed'])

# Metrics
from metrics import NSE, KGE, PBIAS, correlation, RMSE

# Loss
from loss import loss_NSE, loss_RMSE, loss_swi
def criterion(y_true, y_pred):
    return 0.15 * loss_NSE(y_true, y_pred) + 0.85 * loss_RMSE(y_true, y_pred)

#%% Data Loader
from dataloaders import LumpedDataLoader
loader = LumpedDataLoader(
    PATH_DATASET=os.path.join(PATH_DATASET, cfg['dataset']),
)
loader.get_shapes()
loader._set_time_period(
    start_date=cfg['train_start_date'],
    end_date=cfg['train_end_date'],
    lag=365,
    lead=10,
)
# _, _ = loader.get_sample(0, verbose=False)


device = torch.device(f"cuda:{cfg['cuda_num']}" if torch.cuda.is_available() else "cpu")

from models import RainfallRunoff
model = RainfallRunoff(device, model_params).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sample, available = loader.get_sample(0, verbose=False)

# pred_dict = model(sample, available)
# predictions = pred_dict['Q']
# print(f"Predictions shape: {predictions.shape}")

start_epoch = 0
if cfg['mode'] == 'resume':
    checkpoint_files = sorted(glob.glob(os.path.join(PATH_CHECKPOINTS, 'checkpoint_epoch_*.pt')))
    if len(checkpoint_files) > 0:
        checkpoint_latest = checkpoint_files[-1]
        checkpoint = torch.load(checkpoint_latest, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logprint(f"Resumed from checkpoint: {checkpoint_latest}", logfile=logfile_epoch)
        start_epoch = int(checkpoint_latest.split('_')[-1].split('.')[0])
        

#%% Training Loop
def train_one_epoch(model, loader, optimizer, device, cfg, epoch, logfile=None):
    loss_epoch = 0
    model.train()
    train_indices = loader.available_indices
    if cfg['shuffle']:
        random.shuffle(train_indices)

    for sample_idx in tqdm.tqdm(train_indices, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        sample, available = loader.get_sample(sample_idx, verbose=False)
        pred_dict = model(sample, available)
        y_pred = pred_dict['Q']
        y_pred = y_pred[:, :, 0] # [lead, batch_size]
        
        y_true = sample['y_sim'].to(device)
        y_true = y_true[:, :, 0] # [lead, batch_size]

        p = sample['Prcp_lead'].to(device)  # [lead, batch_size]
        pet = sample['PET_lead'].to(device)  # [lead, batch_size]
        swi_pred = pred_dict['swi']

        loss = criterion(y_true, y_pred) + 0.01 * loss_swi(p, pet, y_pred, swi_pred)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
    loss_epoch /= len(train_indices)
    logprint(f"Epoch {epoch+1}: Loss: {loss_epoch:.4f}", logfile=logfile)
    return loss_epoch

#%% Evaluation Loop
def evaluate(model, loader, device, cfg, epoch, logfile):
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
    
    # Compute Metrics
    def compute_metric(pred_tensor, true_tensor, metric_func):
        # pred_tensor: (samples, lead, batch_size)
        # true_tensor: (samples, lead, batch_size)
        metric_values = []
        for lead_idx in range(pred_tensor.shape[1]):
            pred_lead = pred_tensor[:, lead_idx, :]
            true_lead = true_tensor[:, lead_idx, :]
            metric_lead = [metric_func(true_lead[:, i], pred_lead[:, i]) for i in range(pred_lead.shape[1])]
            metric_lead = [round(metric_value.item(), 2) for metric_value in metric_lead]
            metric_values.append(metric_lead)
        return metric_values
    
    NSE_values = compute_metric(pred_tensor, true_tensor, NSE)
    NSE_nanmedian = [float(np.nanmedian(v)) for v in NSE_values]
    logprint(f"NSE: {NSE_nanmedian}", logfile=logfile)

    KGE_values = compute_metric(pred_tensor, true_tensor, KGE)
    KGE_nanmedian = [float(np.nanmedian(v)) for v in KGE_values]
    logprint(f"KGE: {KGE_nanmedian}", logfile=logfile)

    RMSE_values = compute_metric(pred_tensor, true_tensor, RMSE)
    RMSE_nanmedian = [float(np.nanmedian(v)) for v in RMSE_values]
    logprint(f"RMSE: {RMSE_nanmedian}", logfile=logfile)

    PBIAS_values = compute_metric(pred_tensor, true_tensor, PBIAS)
    PBIAS_nanmedian = [float(np.nanmedian(v)) for v in PBIAS_values]
    logprint(f"PBIAS: {PBIAS_nanmedian}", logfile=logfile)

    corr_values = compute_metric(pred_tensor, true_tensor, correlation)
    corr_nanmedian = [float(np.nanmedian(v)) for v in corr_values]
    logprint(f"Correlation: {corr_nanmedian}", logfile=logfile)

    return {
        'NSE': NSE_values,
        'KGE': KGE_values,
        'RMSE': RMSE_values,
        'PBIAS': PBIAS_values,
        'Correlation': corr_values,
        'pred_tensor': pred_tensor,
        'true_tensor': true_tensor,
    }

#%% Execution
monitor = -np.inf
for epoch in range(start_epoch, cfg['num_epochs']):
    loss_epoch = train_one_epoch(model, loader, optimizer, device, cfg, epoch, logfile=logfile_epoch)

    # Save model weights every epoch
    torch.save(model.state_dict(), os.path.join(PATH_MODELS_EPOCH, f'model_epoch_{epoch+1:03d}.pt'))

    if (epoch + 1) % cfg['eval_every'] == 0:
        metrics = evaluate(model, loader, device, cfg, epoch, logfile=logfile_epoch)

        pred_tensor = metrics['pred_tensor']
        true_tensor = metrics['true_tensor']
        torch.save({
            'pred_tensor': pred_tensor,
            'true_tensor': true_tensor,
        }, os.path.join(PATH_PREDICTIONS, f'predictions_epoch_{epoch+1:03d}.pt'))

        nse_value = np.nanmedian(metrics['NSE'][0])

        if nse_value > monitor:
            monitor = nse_value
            torch.save(model.state_dict(), os.path.join(PATH_MODELS_BEST, 'model_best.pt'))
            logprint(f"New best model saved with NSE: {monitor}", logfile=logfile_epoch)

    torch.cuda.empty_cache()
    gc.collect()

    if epoch == cfg['num_epochs'] - 1:
        # Save the predictions of the last epoch
        pred_tensor = metrics['pred_tensor']
        true_tensor = metrics['true_tensor']
        torch.save({
            'pred_tensor': pred_tensor,
            'true_tensor': true_tensor,
        }, os.path.join(PATH_PREDICTIONS, f'predictions_epoch_{epoch+1:03d}.pt'))

        # Save the checkpoint of the last epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(PATH_CHECKPOINTS, f'checkpoint_epoch_{epoch+1:03d}.pt'))

model.load_state_dict(torch.load(os.path.join(PATH_MODELS_BEST, 'model_best.pt')))
metrics = evaluate(model, loader, device, cfg, epoch, logfile=logfile_epoch)
pred_tensor = metrics['pred_tensor']
true_tensor = metrics['true_tensor']
torch.save({
    'pred_tensor': pred_tensor,
    'true_tensor': true_tensor,
}, os.path.join(PATH_PREDICTIONS, f'predictions_best.pt'))