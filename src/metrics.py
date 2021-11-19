from sklearn.metrics import roc_curve, top_k_accuracy_score, balanced_accuracy_score
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(history: History) -> None:
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    fig, ax = plt.subplots(2, 2, figsize=(20,8))
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        if metric == 'loss':
          pos = (0, 0)
        elif metric == 'accuracy':
          pos = (0, 1)
        elif metric == 'precision':
          pos = (1, 0)
        elif metric == 'recall':
          pos = (1, 1)
        ax[pos[0], pos[1]].plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        ax[pos[0], pos[1]].plot(history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val')
        ax[pos[0], pos[1]].set_xlabel('Epoch')
        ax[pos[0], pos[1]].set_ylabel(name)
        ax[pos[0], pos[1]].set_ylim([0,1])
    ax[pos[0], pos[1]].legend()


def plot_roc(name: str, labels: np.array, predictions: np.array, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def compute_score(labels: np.array, prediction: np.array):
    pass