import os
import torch
import pathlib

from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict, List
from torch.utils.data import Dataset

# Helper function to get class names. This function should:
    # 1. Get class names using `os.scandir()` to traverse the directory (should already be in standard image classification format)
    # 2. Raise an error if the class names aren't found (if this happens, there may be something wrong with the directory structure)
    # 3. Turn class names into a dict and a list and return them
def get_class_names(dir_path: pathlib.Path) -> Tuple[Dict[str, int], List[str]]:
    class_names = {}
    class_labels = []
    for index, class_dir in enumerate(os.scandir(dir_path)):
        if class_dir.is_dir():
            class_names[class_dir.name] = index
            class_labels.append(class_dir.name)
    if len(class_names) == 0:
        raise FileNotFoundError("Class names not found. Please check the directory structure.")
    return class_names, class_labels

# Custom 'Dataset' class to replicate the functionality of `torchvision.datasets.ImageFolder()`
class ImageFolderCustom(Dataset):
    def __init__(self, dir_path: pathlib.Path, transform: transforms.Compose=None) -> None:
        self.dir_path = dir_path
        self.transform = transform
        self.image_paths = list(self.dir_path.glob("*/*.jpg"))
        self.class_to_idx, self.classes = get_class_names(dir_path=self.dir_path)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index=index)
        label = self.image_paths[index].parent.name
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
        return image, label
