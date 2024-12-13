'''
@brief visualization metrics for use throughout processing stage.
@author Arjun Ashok (arjun3.ashok@gmail.com),
        Ayush Tripathi (atripathi7783@gmail.com)

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import random
import seaborn as sns
import cv2
from pathlib import Path
import json
import plotly.graph_objects as go


SAVEDIR = Path().cwd() / "report" / "visuals"
JSONDIR = Path().cwd() / "model-reports"
WEIGHTSDIR = Path().cwd() / "model-weights" 


def save_image(img: np.ndarray, name: Path | str) -> None:
    """Saves an image.

    Args:
        img (np.ndarray): array-like
        save_path (Path | str): name to save the image as
    """

    # save the image
    output_path = Path().cwd() / "report" / "visuals" / f"{name}.png"
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to \"{output_path.absolute()}\"")


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
    
    plt.savefig(SAVEDIR, dpi=400)
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
    
    plt.savefig(SAVEDIR / "sample_dataset", dpi=400)
    plt.show()


def plot_train_metrics(metrics: dict, desc: str):
    """
    Plot training and validation loss and accuracy.

    Args:
        metrics (dict): Dictionary containing train/val loss and accuracy.
        desc (str): Description of model for saving in figs directory.
    """
    #create the save directory
    os.makedirs(SAVEDIR, exist_ok=True)

    epochs = range(1, len(metrics["train_loss"]) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", marker='o')
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVEDIR / f"{desc}_train_loss.png", dpi=400)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(epochs, metrics["val_acc"], label="Validation Accuracy", marker='o')
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVEDIR / f"{desc}_train_accuracy.png", dpi=400)

def test_tabular():
    '''Purpose:
    grabs jsons of all relevant testing data of models, compiles and returns table. Stores in ./report/tables.
    
    Args: None
    '''


    test_data = {}
    for filename in os.listdir(JSONDIR):
        if filename.endswith(".json"):
            file_path = os.path.join(JSONDIR, filename)
            with open(file_path, "r") as file:
                try:
                    data = json.load(file)
                    if "test" in data:
                        model_name = filename.replace(".json", "")
                        test_data[model_name] = data["test"]
                    else:
                        print(f"'test' key missing in: {file_path}")
                        
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
    #logic for building dataframe
    rows = []

    #lambda to check if simpler name exists
    model_search = lambda name, model_name: model_name in name


    for model_name, metrics in test_data.items():
        model_types = ["ResNet", "CNN", "KAN", "VIT"]
        for model in model_types:
            if(model_search(model_name, model)): #if model type exists, simplify name for better usage in data table
                model_name = model
        
        row = {
            "Model": model_name,
            "Accuracy": metrics.get("acc", None),
            "Loss": metrics.get("loss", None),
            "Duration": metrics.get("duration", None)
        }

        rows.append(row)
    df = pd.DataFrame(rows)
    print("completed dataframe")
    print(df)
    #render table as png
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')  
    ax.axis('off')    
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    #save
    figure_save = os.path.join(SAVEDIR, "test_metrics_table.png")
    plt.savefig(figure_save)
    print(f"saved data-table to {figure_save}.")
    return df #if needed 

                

def train_time_plot():
    '''
    Purpose: grabs jsons of all relevant trial times of models, compiles and graphs. Stores in ./report/visuals.

    Args: None
    '''

    #section 1: grabbing training times for plotting use
    training_times = {}
    for filename in os.listdir(JSONDIR):
        if filename.endswith(".json"):
            file_path = os.path.join(JSONDIR, filename)
            with open(file_path, "r") as file:
                try:
                    data = json.load(file)
                    if "train" in data and "train_time" in data["train"]:
                        train_times = data["train"]["train_time"]  # Extract list of train times
                        model_name = filename.replace(".json", "")
                        training_times[model_name] = train_times
                    else:
                        print(f"'train' or 'train_time' key missing in: {file_path}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    for key in training_times.keys():
        epoch_times = training_times[key]  
        epochs = range(1, len(epoch_times) + 1)  
        plt.plot(epochs, epoch_times, label=key) 
        
    #formulate plot
    plt.xlabel("Epoch")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time Per Epoch for Models")
    plt.legend(title="Models")
    plt.grid(True)
    plt.tight_layout()

    #save plot to path
    save_image = os.path.join(SAVEDIR, "training_times_per_epoch.png")
    plt.savefig(save_image)  # Save to ./report/visuals
    print(f"Training time plot saved to: {save_image}")



if __name__ == "__main__":
    #main pipeline, run once all model training is complete. 

    # --- Generate Time Graphs --- #
    train_time_plot()

    test_tabular()
    
    
    

        
