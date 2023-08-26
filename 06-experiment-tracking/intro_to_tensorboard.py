import torch
from pathlib import Path
from model import get_model
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from helper_functions import set_seeds, download_data, create_dataloaders, train_model, create_writer

set_seeds(42)

BATCH_SIZE = 32
class_names = ["pizza", "steak", "sushi"]
folder_name = "pizza_steak_sushi_20_percent"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transforms = get_model(output_size=len(class_names), device="cuda")
writer = create_writer(experiment_name="SetupTest", model_name="FoodVisionMiniV2_EfficientNetB0", extra="extraTest")

test_dir = Path("data") / folder_name / "test"
train_dir = Path("data") / folder_name / "train"

download_data(source="https://github.com/JTravaux/pytorch-course/raw/main/pizza_steak_sushi_100_percent.zip", destination=folder_name)
train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=transforms, batch_size=BATCH_SIZE, num_workers=0) # TODO: look into why any num_workers > 0 causes an error

print(summary(model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(
    epochs=5,
    model=model,
    writer=writer,
    device=device,
    loss_fn=loss_fn,
    optimizer=optimizer,
    test_data=test_dataloader,
    train_data=train_dataloader,
    backups_during_training=False,
    name="FoodVisionMiniV2_EfficientNetB0",
)
