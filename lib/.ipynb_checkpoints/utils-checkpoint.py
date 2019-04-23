"""Utility functions."""

import os
from os.path import join
import shutil
import logging

import matplotlib
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import numpy as np


LOG_FORMAT = '%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s'

def create_metric_plot(exp_dir, epochs, train, test, metric_name):
    """Plot metrics and save.

    Args:
        exp_dir (str): experiment directory.
        epochs (list): list of epochs (x-axis of metric plot).
        train (list): list with train in terms of the metric during each epoch.
        test (list): list with test in terms of the metric during each epoch.

    """
    f = plt.figure()
    plt.title(metric_name + "plot")
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.plot(epochs, train, 'b', marker='o', label='train' + metric_name)
    plt.plot(epochs, test, 'r', marker='o', label='test' + metric_name)
    plt.legend()
    plt.savefig(join(exp_dir, metric_name + '.png'))


def create_confusion_matrix(exp_dir, y_true, y_pred, class_names, session):

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    ax1 = plot_confusion_matrix(exp_dir, y_true, y_pred, class_names, session,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    ax2 = plot_confusion_matrix(exp_dir, y_true, y_pred, class_names, session, normalize=True, 
                          title='Normalized confusion matrix')

    return ax1, ax2


def setup_logging(log_path=None, log_level='DEBUG', logger=None, fmt=LOG_FORMAT):
    """Prepare logging for the provided logger.

    Args:
        log_path (str, optional): full path to the desired log file.
        debug (bool, optional): log in verbose mode or not.
        logger (logging.Logger, optional): logger to setup logging upon,
            if it's None, root logger will be used.
        fmt (str, optional): format for the logging message.

    """
    logger = logger if logger else logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info('Log file is %s', log_path)


def save_checkpoint(state, target_dir, file_name='checkpoint.pth.tar',
                    backup_as_best=False,):
    """Save checkpoint to disk.

    Args:
        state: object to save.
        target_dir (str): Full path to the directory in which the checkpoint
            will be stored.
        backup_as_best (bool): Should we backup the checkpoint as the best
            version.
        file_name (str): the name of the checkpoint.

    """
    best_model_path = os.path.join(target_dir, 'model_best.pth.tar')
    target_model_path = os.path.join(target_dir, file_name)

    os.makedirs(target_dir, exist_ok=True)
    torch.save(state, target_model_path)
    if backup_as_best:
        shutil.copyfile(target_model_path, best_model_path)



def plot_confusion_matrix(exp_dir, y_true, y_pred, classes, session,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if normalize:
        plt.savefig(join(exp_dir, session + '_conf_matrix_w_norm' + '.png'))
    else:
        plt.savefig(join(exp_dir, session + '_conf_matrix_wo_norm' + '.png'))
    return ax



def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 