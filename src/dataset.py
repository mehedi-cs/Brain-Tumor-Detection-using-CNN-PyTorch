import torchvision.transforms as transforms
#from torchvision.datasets import ImageFolder
#from torch.utils.data import DataLoader
#import pathlib

def get_dataloaders(data_path='brain', batch_size=64):
    data_dir = pathlib.Path(data_path)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = ImageFolder(data_dir.joinpath("train"), transform=transform)
    val_set = ImageFolder(data_dir.joinpath("val"), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
  
    return train_loader, val_loader
