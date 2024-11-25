'''
Functions to Implement:

1) weighted log loss
2) confusion matrix computation
3) Pretty Accuracy, Precision, Recall, F1
4) Per-Class Metrics
5) Save-Load Metrics

'''

#imports
import json
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np



def compute_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm

def tab_classification_metrics(y_true, y_pred, average='macro', class_names=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    overall_metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, precision, recall, f1]
    }
    overall_df = pd.DataFrame(overall_metrics)
    return overall_df


def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)

def load_metrics(filename):
    with open(filename, 'r') as f:
        return json.load(f)


