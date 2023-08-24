import os
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from models import FoodVisionMini, FoodVisionMiniV2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from helper_functions import train_model, download_data, save_best_model, print_eval_results_table, plot_loss_curves
 
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================
# 1. Setup Data
# ================
train_dir = Path("data") / "pizza_steak_sushi" / "train"
test_dir = Path("data") / "pizza_steak_sushi" / "test"

download_data(source="https://github.com/JTravaux/pytorch-course/raw/main/pizza_steak_sushi_100_percent.zip", destination="pizza_steak_sushi")

# ==========================================
# 2. Prepare transformers and data loaders
# ==========================================
BATCH_SIZE = 128

train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(), 
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)
class_names = train_data.classes

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Amount of data
print(f"Length of train data: {len(train_data)}")
print(f"Length of test data: {len(test_data)}")

# ============================================
# 3. Setup Models, Loss Functions, Optimizers
# ============================================
MODEL_NAME = "04_food_vision_model_v2_best" # Indicates model save name/load from name

model_1 = FoodVisionMiniV2(input_shape=3, hidden_units=128, output_shape=len(class_names)).to(device)
if os.path.exists(f"models/{MODEL_NAME}.pth"):
    model_1.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth"))

models = [
    {
        "model": model_1,
        "model_name": "v1",
        "optimizer": torch.optim.Adam(model_1.parameters(), lr=0.001),
        "epochs": 1000,
        "results": None
    },
]

# ================================
# 4. Train and Evaluate the Models
# ================================
eval_results_list = []
plt.figure(figsize=(15, 10))

for idx, data in enumerate(models):
    data["results"] = train_model(
        model=data["model"],
        epochs=data["epochs"],
        name=data["model_name"],
        optimizer=data["optimizer"],
        test_data=test_dataloader,
        train_data=train_dataloader,
        backups_during_training=False,
        device=next(data["model"].parameters()).device,
    )

    epoch_results = data["results"]["epoch_results"]
    eval_results_list.append(data["results"]["eval_results"])

    # Plot the data
    df = pd.DataFrame(epoch_results)

    plt.subplot(2, 2, 1)
    plt.plot(range(data["epochs"]), epoch_results["train_loss"], label=data["model_name"])
    plt.title("Train Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(data["epochs"]), epoch_results["test_loss"], label=data["model_name"])
    plt.title("Test Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(data["epochs"]), epoch_results["train_acc"], label=data["model_name"])
    plt.title("Train Accuracy")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(data["epochs"]), epoch_results["test_acc"], label=data["model_name"])
    plt.title("Test Accuracy")
    plt.legend()

print_eval_results_table(eval_results_list)
save_best_model(eval_results=eval_results_list, model_name=MODEL_NAME, models_dir="models", results_dir="data/pizza_steak_sushi")
plt.show()
