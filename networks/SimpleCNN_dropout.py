import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_dropout(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleCNN_dropout, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=8, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32 * 56 * 75, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 32 * 56 * 75)
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = self.fc2(x)
        return(x)

