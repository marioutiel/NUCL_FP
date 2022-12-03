from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import sys

from data import get_train_test_loaders
from model_train import ConvNet, FullyNet


def evaluate(outputs: Variable, labels: Variable) -> float:
    """Evaluate neural network outputs against non-one-hotted labels."""
    Y = labels.numpy()
    Yhat = np.argmax(outputs, axis=1)
    return float(np.sum(Yhat == Y))


def batch_evaluate(
        net,
        dataloader: torch.utils.data.DataLoader) -> float:
    """Evaluate neural network in batches, if dataset is too large."""
    score = n = 0.0
    for batch in dataloader:
        n += len(batch['image'])
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])
    return score / n


def validate(conv):
    trainloader, testloader = get_train_test_loaders()
    

    if conv:
        net = ConvNet().float().eval()
        pretrained_model = torch.load("conv_checkpoint.pth")
        print('=' * 10, 'Convolutional NN', '=' * 10)
    else:
        net = FullyNet().float().eval()
        pretrained_model = torch.load("full_checkpoint.pth")
        print('=' * 10, 'Fully Connected NN', '=' * 10)
    net.load_state_dict(pretrained_model)

    
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy: %.1f' % test_acc)

if __name__ == '__main__':
    conv = bool(int(sys.argv[1]))
    validate(conv)
