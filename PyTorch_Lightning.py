import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import pytorch_lightning as pl

import random

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 30
batch_size = 100
learning_rate = 0.001

class LitNeuralNetwork(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, datasets):
        super(LitNeuralNetwork, self).__init__()

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]

        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x
    
    # Called for every training batch during training
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return {"loss": loss} # Must return a dictionary
    
    # Called for every validation batch during training
    # Only works if you have a validation dataloader
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        # Accumulates loss during epoch and does mean reduction by default
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return {"loss": loss} # Must return a dictionary
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.learning_rate_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.learning_rate_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
        }
    
    def on_train_epoch_start(self):
        self.log("learning_rate", self.learning_rate_scheduler.get_last_lr()[0], prog_bar=True)
    
    # Returns the dataloader for the training
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=24,
                                                    persistent_workers=True,
                                                    shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=24,
                                                    persistent_workers=True,
                                                    shuffle=False)
        return val_loader
    
    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                            batch_size=batch_size,
                                            num_workers=24,
                                            persistent_workers=True,
                                            shuffle=False)
        return test_loader

    
def main():
    train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(20),
                                    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                            transform=train_transform,
                                            download=True)
    # Note that train_transform is now also applied to the validation dataset (not recommended in practice)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

    test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                            transform=test_transform)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    # Use fast_dev_run=True to test the model
    trainer = pl.Trainer(max_epochs=num_epochs, devices=1)
    model = LitNeuralNetwork(input_size, hidden_size, num_classes, datasets)
    trainer.fit(model)

    # TODO: Test


if __name__ == "__main__":
    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)

    # GPU setup
    torch.set_float32_matmul_precision('medium')

    main()