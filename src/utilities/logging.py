import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

label_fontsize = 20
title_fontsize = 22
legend_fontsize = 12
tick_fontsize = 14
log_name = 'log.csv'
csv_dir = 'csv_logs'
fig_dir = 'figures'
npy_dir = 'test_npys'
ckpt_dir = 'checkpoints'

def initialize_experiment(save_dir, version_name):
    for dir_name in [save_dir, f'{save_dir}/{version_name}', 
                     f'{save_dir}/{version_name}/{ckpt_dir}',
                     f'{save_dir}/{version_name}/{csv_dir}', 
                     f'{save_dir}/{version_name}/{fig_dir}', 
                     f'{save_dir}/{version_name}/{npy_dir}']:
        os.makedirs(dir_name, exist_ok=True)
    row_dict = {'epoch': None, 'train_loss': None, 'val_loss': None, 
                'train_auc': None, 'val_auc': None}
    write_csv_log(save_dir, version_name, row_dict)

def write_csv_log(save_dir, version_name, row_dict):
        if row_dict['epoch'] is None:
            with open(f'{save_dir}/{version_name}/{csv_dir}/{log_name}', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                writer.writeheader()
        else:
            with open(f'{save_dir}/{version_name}/{csv_dir}/{log_name}', 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                writer.writerow(row_dict)

def freedman_diaconis_bins(data):

    n = len(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        bin_width = 2 * iqr / np.power(n, 1/3)
        data_range = np.max(data) - np.min(data)
        num_bins = int(np.ceil(data_range / bin_width))
    else:
        num_bins = 1
    
    return num_bins

def hist_ab(var_a, var_b, save_dir, version_name, fig_name, a_weight=None, b_weight=None, 
            alabel=None, blabel=None, xlabel=None, ylabel=None, title=None,
            density=True, range=None, shaded=None):

    min_index = np.argmin([len(var_a), len(var_b)])
    bins = freedman_diaconis_bins([var_a,var_b][min_index])
    bins = np.histogram_bin_edges(np.concatenate([var_a,var_b]), bins=bins, range=range)

    fig, ax = plt.subplots()
    hist_a,_,_ = ax.hist(var_a, bins=bins, alpha=0.35, weights=a_weight,
        label=alabel, density=True, range=range, color='blue');
    ax.hist(var_a, bins=bins,histtype='step', weights=a_weight,
        density=density, range=range, color='blue', alpha=0.75, linewidth=2);
    hist_b,_,_ = ax.hist(var_b, bins=bins, alpha=0.35, weights=b_weight,
        label=blabel, density=True, range=range, color='orange');
    ax.hist(var_b, bins=bins, histtype='step', weights=b_weight,
        density=True, range=range, color='orange', alpha=0.75, linewidth=2);
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_xlabel(xlabel,fontsize=label_fontsize)
    ax.set_ylabel(ylabel,fontsize=label_fontsize)
    ax.set_title(title,fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize)

    if shaded is not None:
        min_mass = np.concatenate([var_a,var_b]).min()
        max_mass = np.concatenate([var_a,var_b]).max()
        max_count = np.concatenate([hist_a,hist_b]).max()
        x_lo = np.linspace(min_mass,shaded[0])
        x_hi = np.linspace(shaded[1],max_mass)
        ax.fill_between(x_lo,0,max_count,where=(x_lo<shaded[0]),color='grey',alpha=0.3)
        ax.fill_between(x_hi,0,max_count,where=(x_hi>shaded[1]),color='grey',alpha=0.3)

    fig.savefig(f'{save_dir}/{version_name}/{fig_dir}/{fig_name}', bbox_inches='tight')
    plt.close(fig)

def plot_roc_curve(fpr, tpr, roc_auc, save_dir, version_name, fig_name):

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_title(f'Test AUC: {roc_auc:.4f}', fontsize=title_fontsize)
    ax.set_xlabel('FPR', fontsize=label_fontsize)
    ax.set_ylabel('TPR', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    fig.savefig(f'{save_dir}/{version_name}/{fig_dir}/{fig_name}', bbox_inches='tight')
    plt.close(fig)

def plot_loss_history(save_dir, version_name, fig_name, early_stop=None):

    log = pd.read_csv(f'{save_dir}/{version_name}/{csv_dir}/{log_name}')
    train_loss = log['train_loss'].values
    val_loss = log['val_loss'].values

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(train_loss)), train_loss, label='Train')
    ax.plot(np.arange(len(val_loss))-0.5, val_loss, label='Valid')
    if early_stop is not None:
        ax.axvline(x=early_stop, linestyle='--', linewidth=1, label='Early Stop')
    ax.set_title(f'Loss History', fontsize=title_fontsize)
    ax.set_xlabel('Epoch', fontsize=label_fontsize)
    ax.set_ylabel('Loss', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    fig.savefig(f'{save_dir}/{version_name}/{fig_dir}/train_val_loss.png', bbox_inches='tight')
    plt.close(fig)

def plot_auc_history(save_dir, version_name, fig_name, early_stop=None):

    log = pd.read_csv(f'{save_dir}/{version_name}/{csv_dir}/{log_name}')
    train_auc = log['train_auc'].values
    val_auc = log['val_auc'].values

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(train_auc)), train_auc, label='Train')
    ax.plot(np.arange(len(val_auc))-0.5, val_auc, label='Valid')
    if early_stop is not None:
        ax.axvline(x=early_stop, linestyle='--', linewidth=1)
    ax.set_title(f'AUC History', fontsize=title_fontsize)
    ax.set_xlabel('Epoch', fontsize=label_fontsize)
    ax.set_ylabel('AUC', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    fig.savefig(f'{save_dir}/{version_name}/{fig_dir}/train_val_auc.png', bbox_inches='tight')
    plt.close(fig)

def plot_outputs(labels, outs, save_dir, version_name, fig_name):
    
    hist_ab(outs[labels==1], outs[labels==0], save_dir, version_name, fig_name,
            alabel='Signal', blabel='Background', xlabel='Output', ylabel='Density',
            density=True)