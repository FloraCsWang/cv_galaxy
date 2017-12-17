import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.01)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nclasses = 37
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv31 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, self.nclasses)

    def forward(self, x):
        for i in x.shape[0]:
            x[i] = F.relu(F.max_pool2d(self.conv1(x[i]), 2))
            # x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x[i] = F.relu(F.max_pool2d(self.conv2(x[i]), 2))
            # x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x[i] = F.relu(F.max_pool2d(self.conv32(self.conv31(x[i])), 2))
            # x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)
