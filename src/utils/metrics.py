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
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, confusion_matrix


def weighted_log_loss(y_true, y_pred, sample_weights=None): 
    return log_loss(y_true, y_pred, sample_weight=sample_weights)

def compute_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm

def save_metrics(metrics, filename):

    with open(filename, 'w') as f:
        json.dump(metrics, f)

def load_metrics(filename):
    with open(filename, 'r') as f:
        return json.load(f)


