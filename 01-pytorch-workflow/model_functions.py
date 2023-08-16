import torch
from pathlib import Path

def save_model(model, model_name: str, model_path = Path("models")):
    model_path.mkdir(parents=True, exist_ok=True)
    model_save_path = model_path / model_name
    torch.save(model, model_save_path)
    print(f"Model saved to: {model_save_path}")

def load_model(model_name: str, model_path = Path("models")):
    model_save_path = model_path / model_name
    return torch.load(model_save_path)
