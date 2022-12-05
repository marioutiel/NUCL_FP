from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

class SignLanguageMNIST(Dataset):
    @staticmethod
    def get_label_mapping():
        
        mapping = list(range(25))
        mapping.pop(9)
        
        return mapping

    @staticmethod
    def read_label_samples_from_csv(path: str):
        
        mapping = SignLanguageMNIST.get_label_mapping()
        labels, samples = [], []

        with open(path) as f:
            _ = next(f)
            for line in csv.reader(f):
                label = int(line[0])
                labels.append(mapping.index(label))
                samples.append(list(map(int, line[1:])))
        
        return labels, samples

    def __init__(self,
            path: str='data/sign_mnist_train.csv', 
            mean: list[float]=[0.485],
            std: list[float]=[0.229]):
        
        labels, samples = SignLanguageMNIST.read_label_samples_from_csv(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1,28,28,1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1,1))

        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._samples[index]).float(),
            'label': torch.from_numpy(self._labels[index]).float()
        }

def get_train_test_loaders(batch_size=32):
    trainset = SignLanguageMNIST('data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size,
            shuffle=True)

    testset = SignLanguageMNIST('data/sign_mnist_test.csv')
    testloader = torch.utils.data.DataLoader(testset,
            batch_size=batch_size,
            shuffle=False)

    return trainloader, testloader

def plot(n_rows=3,n_cols=3):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    
    mapping = SignLanguageMNIST.get_label_mapping()
    labels, samples = SignLanguageMNIST.read_label_samples_from_csv('data/sign_mnist_train.csv')
    for i in range(n_rows):
        for j in range(n_cols):
            image = np.array(samples[i*n_rows+j])
            label = labels[i*n_rows+j]

            ax[i,j].imshow(image.reshape((28,28)), cmap='gray')
            ax[i,j].set_title(f'Label: {"ABCDEFGHIJKLMNOPQRSTUVWXY"[mapping[label]]}')
            ax[i,j].axis('off')

    fig.savefig('Example.jpg')
    fig.show()

if __name__ == '__main__':
    loader, _ = get_train_test_loaders(2)
    plot()