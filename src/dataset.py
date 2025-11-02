import torch
from torchvision import datasets, transforms
import pathlib
import splitfolders


def prepare_data(data_dir, output_dir='brain', split_ratio=(0.8, 0.2), seed=20):
    data_dir = pathlib.Path(data_dir)
    splitfolders.ratio(data_dir, output=output_dir, seed=seed, ratio=split_ratio)
    return pathlib.Path(output_dir)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
