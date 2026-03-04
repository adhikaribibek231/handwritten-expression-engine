import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 32×28×28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64×14×14 after pool

        self.pool = nn.MaxPool2d(2, 2)                            # halves H,W
        self.dropout = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                     # 3136 → 128
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32×14×14
        x = self.pool(F.relu(self.conv2(x)))  # 64×7×7

        x = self.flatten(x)                   # 3136
        x = F.relu(self.fc1(x))               # 128
        x = self.dropout(x)
        x = self.fc2(x)                       # logits (10)

        return x
    
if __name__=="__main__":
    model = MNISTCNN()
    x= torch.randn(32,1,28,28)
    y= model(x)
    print(y.shape)