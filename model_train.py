from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import sys
from time import time 

from data import get_train_test_loaders

class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48,24)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 16*5*5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

class FullyNet(nn.Module):

    def __init__(self):
        
        super(FullyNet, self).__init__()
    
        self.fc = nn.Linear(784, 928)
        self.fc1 = nn.Linear(928, 700)
        self.fc2 = nn.Linear(700, 400)
        self.fc3 = nn.Linear(400, 120)
        self.fc4 = nn.Linear(120, 48)
        self.fc5 = nn.Linear(48, 24)

    def forward(self, x):
        
        x = x.view(-1, 28*28)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return self.fc5(x)


def main(conv, n_epochs=15):
    
    if conv:
        print(f'{"="*5} Using Convolutional Neural Network {"="*5}')
        net = ConvNet().float()
    else:
        print(f'{"="*5} Using Fully Connected Neural Network {"="*5}')
        net = FullyNet().float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainloader, _ = get_train_test_loaders()

    print(f'{"="*5} Training with {n_epochs} Epochs {"="*5}')
    start = time()
    for epoch in range(n_epochs):
        train(net, criterion, optimizer, trainloader, epoch)
        scheduler.step()
    end = time()
    total_secs = int(end-start)
    mins = total_secs//60
    secs = total_secs%60
    print(f'Time Spent Training: {mins} Minutes, {secs} Seconds')
    if conv:
        torch.save(net.state_dict(), 'conv_checkpoint.pth')
    else:
        torch.save(net.state_dict(), 'full_checkpoint.pth')

def train(net, criterion, optimizer, trainloader, epoch):

    running_loss = 0.0

    for idx, data in enumerate(trainloader, 0):
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels[:,0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx%100 == 0:
            print(f'Epoch: {epoch} with Index: {idx} --> Loss: {running_loss/(idx+1)}')

if __name__ == '__main__':
    conv = bool(int(sys.argv[1]))
    n_epochs = int(sys.argv[2])
    
    main(conv, n_epochs)