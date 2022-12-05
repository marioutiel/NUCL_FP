from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
import sys
import onnx
import onnxruntime as ort

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

    trainloader, testloader = get_train_test_loaders(1)

    fname = "signlanguage.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])

    model = onnx.load(fname)
    onnx.checker.check_model(model)

    ort_session = ort.InferenceSession(fname)
    net = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]

    print('=' * 10, 'ONNX', '=' * 10)
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy: %.1f' % test_acc)


if __name__ == '__main__':
    conv = bool(int(sys.argv[1]))
    validate(conv)
