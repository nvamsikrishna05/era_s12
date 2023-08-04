import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import lightning.pytorch as pl

class BaseNet(pl.LightningModule):
    def __init__(self, learning_rate, epochs):
        super(BaseNet, self).__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.epochs = epochs

        # Prep Layer
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Layer 1
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Layer 2
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Layer 3
        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=10, bias=False)
        )


    def forward(self, x):
        
        x = self.prep(x)
        
        X1 = self.c1(x)
        R1 = self.c2(x)
        x = X1 + R1

        x = self.c3(x)

        X2 = self.c4(x)
        R2 = self.c5(x)
        x = X2 + R2

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = OneCycleLR(optimizer,max_lr= self.lr,
                               total_steps= self.epochs * self.trainer.estimated_stepping_batches,
                               pct_start=10/self.epochs,
                               div_factor=5,
                               final_div_factor=10,
                               anneal_strategy='cos')
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = F.cross_entropy(output, labels)
        preds = torch.argmax(output, dim=1)
        acc = (labels == preds).float().mean()

        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage):
        images, labels = batch
        output = self(images)
        loss = F.cross_entropy(output, labels)
        preds = torch.argmax(output, dim=1)
        acc = (labels == preds).float().mean()

        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")