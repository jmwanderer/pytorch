"""
Siamese Net for image comparision

"""
import os
from typing_extensions import runtime
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import sklearn
import numpy as np

import matplotlib.pyplot as plt

def plot_loss(history):
  """
  Plot loss values generated by model.fit
  """
  plt.plot(history, label='train_loss')
  plt.ylim([0, 5])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.show()

class ImageDataset(Dataset):
    def __init__(self, root="data", train=True, transform=None, target_transform=None):
        data = sklearn.datasets.fetch_olivetti_faces(data_home=root)
        self.images = []
        self.targets = []

        target = np.array(data.target, dtype=np.int64)

        for index in range(len(data.images)):
            if ((train and index % 10 != 0) or
                (not train and index % 10 == 0)):
                self.images.append(data.images[index])
                self.targets.append(target[index])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Define model
class ConvNeuralNetwork(nn.Module):
    def __init__(self, image_size, label_classes):
        super().__init__()
        self.size = int(image_size / 4)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32*self.size*self.size, 64)
        self.fc2 = nn.Linear(64, label_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 32*self.size*self.size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, image_size, label_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(image_size * image_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, label_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    global pred, y
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(torch.get_default_device()), y.to(torch.get_default_device())

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss.item() * X.shape[0]

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss_v, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_v:>7f}  [{current:>5d}/{size:>5d}]")
    return running_loss / len(dataloader.dataset)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(torch.get_default_device()), y.to(torch.get_default_device())
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def get_training_data(use_faces_data, use_conv_model):
    if use_faces_data:
        training_data = ImageDataset(
            root="data",
            train=True,
            transform=ToTensor())

        test_data = ImageDataset(
            root="data",
            train=False,
            transform=ToTensor())
        image_size=64
        label_classes=40
        batch_size=10
        learning_rate=1e-2
    else:
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor())

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor())
        image_size=28
        label_classes=10
        batch_size=64
        learning_rate=1e-3
    return training_data, test_data, image_size, label_classes, batch_size, learning_rate

def train_model(use_faces_data, use_conv_model):
    training_data, test_data, image_size, label_classes, batch_size, learning_rate = get_training_data(use_faces_data, use_conv_model)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    if use_conv_model:
        model = ConvNeuralNetwork(image_size, label_classes).to(torch.get_default_device())
    else:
        model = NeuralNetwork(image_size, label_classes).to(torch.get_default_device())
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 100
    global loss_history
    loss_history = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_dataloader, model, loss_fn, optimizer)
        loss_history.append(loss)

    test(test_dataloader, model, loss_fn)    
    print("Saving model")
    save_model(use_faces_data, use_conv_model, model)
    print("Done!")
    plot_loss(loss_history)
    return model

def get_name(use_faces_data, use_conv_model):
    if use_conv_model:
        name = "cnn_"
    else:
        name = "nn_"
    if use_faces_data:
        name += "faces"
    else:
        name += "fashion"
    name += ".pt"
    return os.path.join("data", name)

def save_model(use_faces_data, use_conv_model, model):
    path = get_name(use_faces_data, use_conv_model)
    torch.save(model.state_dict(), path)

def load_cnn(use_faces_data) -> ConvNeuralNetwork:
    path = get_name(use_faces_data, use_conv_model=True)
    training_data, test_data, image_size, label_classes, batch_size, learning_rate = get_training_data(use_faces_data, use_conv_model=True)
    model = ConvNeuralNetwork(image_size, label_classes).to(torch.get_default_device())
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def load_lnn(use_faces_data) -> NeuralNetwork:
    path = get_name(use_faces_data, use_conv_model=False)
    training_data, test_data, image_size, label_classes, batch_size, learning_rate = get_training_data(use_faces_data, use_conv_model=False)
    model = NeuralNetwork(image_size, label_classes).to(torch.get_default_device())
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def load_model(use_faces_data, use_conv_model):
    if use_conv_model:
        model = load_cnn(use_faces_data)
    else:
        model = load_lnn(use_faces_data)
    print(model)
    return model

USE_CONV_MODEL=True
USE_FACES_DATA=True

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    train_model(USE_FACES_DATA, USE_CONV_MODEL)
