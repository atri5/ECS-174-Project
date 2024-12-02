'''
@brief visualization metrics for use throughout processing stage.
@author Ayush Tripathi (atripathi7783@gmail.com)

'''

import numpy as np
import matplotlib as plt
import pandas as pd
import os
import csv
import random
import seaborn as sns

def loss_visualization(training_loss, validation_loss, epochs, dir=""):
    #input training loss, validation loss through arrays, specify directory + filename for saving config
    if len(training_loss) != epochs or len(validation_loss) != epochs:
        raise ValueError("Length of loss lists must match the number of epochs.")
    
    data = {
        'Epoch': list(range(1, epochs + 1)) * 2,
        'Loss': training_loss + validation_loss,
        'Type': ['Training Loss'] * epochs + ['Validation Loss'] * epochs
    }
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Epoch', y='Loss', hue='Type', style='Type', markers=True)
    plt.title('Loss Visualization Over Epochs')
    
    cur = os.getcwd()
    file_path = os.path.join(cur, dir)
    plt.savefig(file_path)
    plt.show()




def log_metrics(metrics, epoch, logs_file = "metrics.csv"):
    '''
    purpose:
    helper function to log/store metrics when running model.

    params:
    metrics (dict): pass in error for current epoch
    epoch (int): current epoch
    logs_file (str): where logs are accumulated
    '''
    
    
    file_exists = os.path.isfile(logs_file)
    with open(logs_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch'] + list(metrics.keys()))  
        writer.writerow([epoch] + list(metrics.values()))
    print(f"Metrics logged for epoch {epoch}: {metrics}")


def visualize_samples(dataset, class_names, samples_to_display=10):
    '''
    purpose: helper function to visualize random samples within the data.


    params:
    dataset (torch.utils.data.Dataset): PyTorch dataset object
    class_names (list): list of class names
    samples_to_display (int): # of samples displayed at a time
    
    '''
    
    indices = random.sample(range(len(dataset)), samples_to_display)
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axes[i].set_title(f"Class: {class_names[label]}")
        axes[i].axis('off')
    
    plt.show()