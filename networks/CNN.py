import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class CNN(nn.Module):

    def __init__(self, num_classes=1000):
        super(CNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            Flatten(),
            nn.Linear(64 * 112 * 150, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),

        )


    def forward(self, x):
        return self.layers(x)
