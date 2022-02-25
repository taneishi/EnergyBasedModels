import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import os

from DBN import DBN

def initialize_model():
    model = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 10),
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

def train(model, epochs=5, batch_size=64):
    train_dataset = datasets.MNIST('dataset', download=True, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    layers = [512, 128, 64, 10]

    dbn = DBN(train_x.shape[1], layers, savefile='models/mnist_trained_dbn.pt')

    print('Unsupervised pre-training of DBN')
    dbn.train_DBN(train_x)

    model = dbn.initialize_model()

    completed_model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    torch.save(completed_model, 'models/mnist_trained_dbn_classifier.pt')

    print('Without Pre-Training')
    model = initialize_model()
    model, progress = train(model)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('results/DBN_without_pretraining_classifier.csv', index=False)

    print('With Pre-Training')
    model, progress = train(completed_model)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('results/DBN_with_pretraining_classifier.csv', index=False)
