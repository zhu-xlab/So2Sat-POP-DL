'''
Utility Module for Metric Calculations
'''

import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns

from utils.dataset import denormalize_reg_labels

import pandas as pd


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Can be used for accumulating the Loss or other Metrics
    """
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_reg_metrics(targets, preds, id_list, path, dataset_name):
    '''
    Calculates the metrics based on targets and predictions and saves them. Filename is specified by path and dataset_name
    -----------
    :Args:
        targets: ground truth torch tensor
        preds: prediction torch tensor
        path: directory where to save metrics
        dataset_name: name of dataset e.g. if it is the test dataset
    '''
    log_file = os.path.join(path, path.split(os.sep)[-1]+'_' + dataset_name + '.txt')
    y_true = targets.detach().numpy()
    y_pred = preds.detach().numpy()
    y_pred = denormalize_reg_labels(y_pred)

    all_targets_list = y_true.tolist()
    all_preds_list = y_pred.tolist()
    data = {}

    data.update({'GRD_ID': id_list})
    data.update({'GT_POP': all_targets_list})
    data.update({'PR_POP': all_preds_list})
    df = pd.DataFrame(data)

    csv_path = os.path.join(path, path.split(os.sep)[-1] + '_evaluation.csv')

    df.to_csv(csv_path, index=False)

    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true,y_pred)
    me = mean_error(y_true, y_pred)
    print(f"\n — Mean Absolute Error: {mae} — Root Mean Squared Error: {rmse}"
            f" \n — R2: {r2} \n — Bias {me}")
    ## save scatterplot
    plot_scatter(y_true, y_pred, log_file.replace('.txt', '_scatter.png'))
    with open(log_file, 'a') as f:
        f.writelines(f"\n — Mean Absolute Error: {mae} — Root Mean Squared Error: {rmse}"
            f" \n — R2: {r2} \n — Bias {me}")


def plot_scatter(targets, preds, path):
    '''
    Creates a Scatterplot
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    plt.rcParams['font.size'] = '20'
    #sns.set(font_scale=5)
    # ax.scatter(targets, preds)
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=2)
    sns.regplot(x = targets, y = preds, scatter_kws={"color": "#069AF3"}, line_kws={"color": "#DC143C"}, ci=None)
    ax.set_xlabel('Observed', fontsize=20)
    ax.set_ylabel('Predicted', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    #plt.title('Predictions vs Actual Values')
    plt.axis('equal')
    plt.savefig(path, bbox_inches="tight", dpi=600)
    plt.close()


def mean_error(targets, preds):
    return np.mean(preds-targets)
