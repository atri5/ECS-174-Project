'''
@brief visualization metrics for use throughout processing stage.
@author Ayush Tripathi (atripathi7783@gmail.com)

'''

import numpy as np
import matplotlib.pyplot as plt
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


def plot_train_metrics(metrics: dict, desc: str):
    """
    Plot training and validation loss and accuracy.

    Args:
        metrics (dict): Dictionary containing train/val loss and accuracy.
        desc (str): Description of model for saving in figs directory.
    """
    #create the save directory
    save_dir = "./src/experiments/figs"
    os.makedirs(save_dir, exist_ok=True)


    epochs = range(1, len(metrics["train_loss"]) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", marker='o')
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{desc}_train_loss.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(epochs, metrics["val_acc"], label="Validation Accuracy", marker='o')
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{desc}_train_accuracy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    metrics = metrics = {
    "train_loss": [0.8, 0.6, 0.4, 0.35],
    "train_acc": [0.6, 0.75, 0.85, 0.9],
    "val_loss": [0.9, 0.7, 0.5, 0.4],
    "val_acc": [0.55, 0.7, 0.8, 0.85]
    }
    plot_train_metrics(metrics, "unet-model")
