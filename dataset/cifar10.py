from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from lightning.pytorch.core import LightningDataModule

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transforms) -> None:
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = np.array(image)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        
        return (image, label)
        

class CIFAR10DataModule(LightningDataModule):
    def __init__(self, train_transforms, test_transforms, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def prepare_data(self):
        self.cifar10 = CIFAR10Dataset(datasets.CIFAR10('./data', train=True, download=True), self.train_transforms)
        self.test_data = CIFAR10Dataset(datasets.CIFAR10('./data', train=False, download=True), self.test_transforms)

    def setup(self, stage):
            self.train_data, self.val_data = random_split(self.cifar10, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=True, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=True, batch_size=self.batch_size)