"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile
from pathlib import Path

import requests

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def show_side_by_side_decision_boundary(model, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1) # (rows, columns, index)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2) # (rows, columns, index)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

def eval_model(
        model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader,
        start_time: float,
        end_time: float, 
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device = "cpu",
        accuracy_fn = accuracy_fn):
    """Returns a dictionary containing the results of model predictions on data_loader."""
    loss, acc = 0, 0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)

            loss += loss_fn(y_logits, y)
            acc += accuracy_fn(y_true=y, y_pred=y_logits.argmax(dim=1))

        # Scale loss and acc by number of batches
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_obj": model,
        "model_name": model.__class__.__name__,
        "loss": loss.item(),
        "acc": acc,
        "epochs": epochs,
        "device": device,
        "optimizer": optimizer,
        "train_time": end_time - start_time,
    }

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device = "cpu", 
               accuracy_fn = accuracy_fn,
               verbose: bool = True):
    """Performs a single training step (forward pass, backward pass, weights update) and returns the loss and accuracy for the batch."""

    model.train()
    model.to(device)
    train_loss, train_acc = 0.0, 0.0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        train_acc +=  accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    if verbose:
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
    }

def test_step(model: torch.nn.Module, 
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device = "cpu", 
              accuracy_fn = accuracy_fn,
              verbose: bool = True):
    """Performs a single evaluation step (forward pass) and returns the loss and accuracy for the batch."""

    model.eval()
    model.to(device)
    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_acc +=  accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            test_loss += loss.item()

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    if verbose:
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def print_eval_results_table(eval_results: List[dict]) -> None:
    """Prints a table of eval_results.

    Args:
        eval_results (List[dict]): A list of dictionaries containing model evaluation results.
    """
    # Make a table of the results, sorted by acc and time
    eval_results.sort(key=lambda x: x["acc"], reverse=True)

    # Print the table headers with spaces between each column using {:<10} for each column
    print("\n{:<20} {:<10} {:<15} {:<10} {:<20} {:<15} {:<15}".format("Model", "Device", "Accuracy (%)", "Loss", "Train Time (s)", "Optimizer", "LR"))
    print("-" * 110)


    # Print the table rows
    for result in eval_results:
        learning_rate = result["optimizer"].state_dict()["param_groups"][0]["lr"]
        optimizer_name = result["optimizer"].__class__.__name__
        print(f"{result['model_name']:<20} {str(result['device']):<10} {result['acc']:<15.2f} {result['loss']:<10.2f} {result['train_time']:<20.2f} {optimizer_name:<15} {learning_rate:<15.5f}")

def save_best_model(eval_results: List[dict], model_name: str, models_dir="models", results_dir: str = "data") -> None:
    results_file = f"{results_dir}/model_eval_results"
    results_file_acc = f"{results_dir}/model_eval_results_acc.txt"

    # Sort eval_results by accuracy
    eval_results.sort(key=lambda x: x["acc"], reverse=True)

    # Check if RESULTS_FILE_ACC exists, if not, create it
    if not os.path.exists(results_file_acc):
        with open(results_file_acc, "w") as f:
            f.write("")
        best_acc = None
    else:
        with open(results_file_acc, "r") as f:
            best_acc = f.read()
            if best_acc == "":
                best_acc = None
            else:
                best_acc = float(best_acc)

    if not best_acc:
        print("\nNo best accuracy found, saving current model...")
    else:
        print(f"\nBest accuracy so far: {best_acc:.2f}%")

    current_best = eval_results[0]["acc"]
    print(f"Current best accuracy: {current_best:.2f}%")

    # If the best model's accuracy is less than the current model's accuracy, save the current model & update the best accuracy
    if not best_acc or current_best > best_acc:
        torch.save(eval_results[0]["model_obj"].state_dict(), f"{models_dir}/{model_name}.pth")

        with open(results_file_acc, "w") as f:
            f.write(str(current_best))

        print(f"Saved model to {models_dir}/{model_name}.pth")
    else:
        print("Current model's accuracy is less than the best model's accuracy, not saving...")

    with open(f"{results_file}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt", "w") as f:
        f.write(str(eval_results))
        
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device):
    """Makes predictions on data using model.

    Args:
        model (torch.nn.Module): trained PyTorch model.
        data (list): data to make predictions on.
        device (torch.device): target device to compute on.

    Returns:
        torch.Tensor: predictions made on data.
    """
    model.to(device)
    model.eval()

    pred_probs_list = []

    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logits = model(sample)
            pred_probs = torch.softmax(pred_logits.squeeze(), dim=0)
            pred_probs_list.append(pred_probs.cpu())
        
    return torch.stack(pred_probs_list)
