import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import os

from RBM import RBM

def initialize_model():
    model = torch.nn.Sequential(
            torch.nn.Linear(784, 2500),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2500, 10),
            torch.nn.Softmax(dim=1),
            )

    return model

def test(model):
    test_dataset = datasets.MNIST('dataset', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    for test_x, test_y in test_loader:
        test_x = test_x.view(-1, 784)

    criterion = torch.nn.CrossEntropyLoss()

    output_test = model(test_x)
    test_loss = criterion(output_test, test_y).item()

    output_test = torch.argmax(output_test, axis=1)
    test_acc = torch.sum(output_test == test_y).item() / len(test_dataset)

    return test_loss, test_acc

def train(model, epochs=5):
    train_dataset = datasets.MNIST('dataset', download=True, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    progress = []

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        for batch_x, target in train_loader:
            batch_x = batch_x.view(-1, 784)
            output = model(batch_x)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            output = torch.argmax(output, dim=1)
            train_acc += torch.sum(output == target).item() / target.shape[0]

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print('epoch %d train loss %5.3f train acc %5.3f' % (epoch+1, train_loss, train_acc))

        test_loss, test_acc = test(model)

        progress.append([epoch+1, test_loss, train_loss, test_acc, train_acc])

    return model, progress

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    train_dataset = datasets.MNIST('dataset', download=True, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    for train_x, train_y in train_loader:
        train_x = train_x.view(-1, 784)

    vn = train_x.shape[1]
    hn = 2500
    rbm = RBM(vn, hn, savefile='models/mnist_trained_rbm.pt')

    print('Unsupervised pre-training of RBM')
    rbm.train(train_x)

    model = initialize_model()

    print('\nTraining without pre-training')
    model, progress = train(model)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('results/RBM_without_pretraining_classifier.csv', index=False)

    model = initialize_model()

    model[0].weight = torch.nn.Parameter(rbm.W)
    model[0].bias = torch.nn.Parameter(rbm.hb)

    print('\nTraining with pre-training')
    model, progress = train(model)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('results/RBM_pretrained_classifier.csv', index=False)
