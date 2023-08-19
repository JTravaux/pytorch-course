import time
import torch
import matplotlib.pyplot as plt

from torch import nn
import tqdm
from models import AlexNet, Food101V0
from torchvision import datasets, transforms
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from helper_functions import eval_model, train_step, test_step, print_eval_results_table, save_best_model, make_predictions, show_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================
# 1. Get Dataset(s)
# ===================
data_transformers = transforms.Compose([
    transforms.Resize(size=(227, 227)),
    transforms.ToTensor()
])

train_data = datasets.Food101(root="data",
                              download=True,
                              transform=data_transformers,
                              split="train")

test_data = datasets.Food101(root="data",
                                download=True,
                                transform=data_transformers,
                                split="test")

print(len(train_data), len(test_data))
print(f"Image shape: {train_data[0][0].shape}") # 3, 227, 227

random_train_idx = torch.randint(0, len(train_data), (1,)).item()
show_image(image_tensor=train_data[random_train_idx][0], label=train_data.classes[train_data[random_train_idx][1]])

classes = train_data.classes
num_classes = len(classes) # 101

print(f"Number of classes: {num_classes}")
print(f"Classes: {classes}")

# Create Dataloaders
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data shape: {next(iter(train_loader))[0].shape}") # 64, 3, 227, 227

# ===================
# 2. Create Model(s)
# ===================
models = [
    AlexNet(num_classes=num_classes).to(device),
    Food101V0(input_shape=3, hidden_units=32, output_shape=num_classes).to(device)
]

# sample = models[0](torch.randn(1, 3, 227, 227).to(device))
# print(f"Sample shape: {sample.shape}")  # torch.Size([1, 101])

loss_fn = nn.CrossEntropyLoss()

# ===================
# 3. Train Model
# ===================
EPOCHS = 3
eval_results = []
train_models = True

if train_models:
    print(f"\nStarting training of {len(models)} models...")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Images per epoch: {len(train_loader) * BATCH_SIZE}")

    for idx, model in enumerate(models):
        device = next(model.parameters()).device
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        print(f"\nTraining model {idx + 1}/{len(models)} ({model.__class__.__name__}) on {device}...")

        for epoch in range(EPOCHS):
            start_time = time.time()
            train_step(model=model, data_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, device=device, verbose=True)
            test_step(model=model, data_loader=test_loader, loss_fn=loss_fn, device=device, verbose=True)
            end_time = time.time()

        eval_results.append(eval_model(
            model=model,
            epochs=EPOCHS,
            device=device,
            loss_fn=loss_fn,
            end_time=end_time,
            optimizer=optimizer,
            start_time=start_time,
            data_loader=test_loader,
        ))

    print("\n===================")
    print("Training completed!")
    print("===================\n")
    print(f"Models trained: {len(models)}")
    print(f"Number of epochs: {EPOCHS}")
    print("=====================")

    print_eval_results_table(eval_results)
    save_best_model(eval_results=eval_results, model_name="03_food101_model_best", models_dir="models", results_dir="data/food-101")
else:
    print("Loading model...")
    model = AlexNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/03_food101_model_best.pth"))

    # Print Confusion Matrix

    # 1. Make predictions
    preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_loader, "Making predictions..."):
            X, y = X.to(device), y.to(device)

            # Logits -> Probabilities -> Class
            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

            preds.append(y_pred.cpu())
    
    preds = torch.cat(preds)

    # 2. Create Confusion Matrix
    confmat = ConfusionMatrix(num_classes=num_classes, task="multiclass")
    confmat_tensor = confmat(preds=preds, target=test_data.targets)
    print(f"Confusion matrix:\n{confmat_tensor}")

    # 3. Plot Confusion Matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=test_data.classes, # turn the row and column labels into class names
        figsize=(10, 7)
    );

    plt.show()
