import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import os

from DBN import DBN

if __name__ == '__main__':
    os.makedirs('figure', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    test_dataset = datasets.MNIST('dataset', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    for test_x, test_y in test_loader:
        test_x = test_x.view(-1, 784)

    layers = [512, 128, 64, 10]
    dbn = DBN(device, test_x.shape[1], layers)
    dbn.layer_parameters = torch.load('models/mnist_trained_dbn.pt')
    
    plt.figure(figsize=(25, 30))

    for n in range(10):
        x = test_x[np.where(test_y==n)[0][0]]
        x = x.unsqueeze(0)
        gen_image, hidden_image = dbn.reconstructor(x)

        #print(f'MAE (mean absolute error) of an all 0 reconstructor: {torch.mean(x).item():5.3f}')
        #print(f'MAE between reconstructed and original sample: {torch.mean(torch.abs(gen_image - x)).item():5.3f}')

        gen_image = gen_image.numpy()
        hidden_image = hidden_image.numpy()
        image = x.numpy()

        # revert transforms.ToTensor() scaling
        image = (image*255)[0]
        hidden_image = (hidden_image*255)[0]
        gen_image = (gen_image*255)[0]

        image = np.reshape(image, (28, 28))
        hidden_image = np.reshape(hidden_image, (5, 2))
        gen_image = np.reshape(gen_image, (28, 28))

        image = image.astype(np.int32)
        hidden_image = hidden_image.astype(np.int32)
        gen_image = gen_image.astype(np.int32)

        plt.subplot(5, 6, 1 + n*3)
        plt.imshow(image, cmap='gray')
        plt.title('original image')

        plt.subplot(5, 6, 2 + n*3)
        plt.imshow(hidden_image, cmap='gray')
        plt.title('hidden image')

        plt.subplot(5, 6, 3 + n*3)
        plt.imshow(gen_image, cmap='gray')
        plt.title('reconstructed image')

        print(f'generated images for digit {n}')

    plt.tight_layout()
    plt.savefig('figure/DBN_digits.jpg', dpi=20)
