import torch
from torchvision import datasets, transforms
import pathlib
import splitfolders

def prepare_data(data_dir, output_dir='brain', split_ratio=(0.8, 0.2), seed=20):
    data_dir = pathlib.Path(data_dir)
    splitfolders.ratio(data_dir, output=output_dir, seed=seed, ratio=split_ratio)
    return pathlib.Path(output_dir)
