import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import os

from RBM import RBM

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    test_dataset = datasets.MNIST('dataset', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    for test_x, test_y in test_loader:
        test_x = test_x.view(-1, 784)

    vn = test_x.shape[1]
    hn = 2500

    rbm = RBM(device, vn, hn)
    rbm.load_rbm('models/mnist_trained_rbm.pt')

    plt.figure(figsize=(25, 30))
    
    for n in range(10):
        x = test_x[np.where(test_y==n)[0][0]]
        x = x.unsqueeze(0)
        hidden_image = []
        gen_image = []

        for k in range(rbm.k):
            _, hk = rbm.sample_h(x)
            _, vk = rbm.sample_v(hk)
            gen_image.append(vk.cpu().numpy())
            hidden_image.append(hk.cpu().numpy())

        hidden_image = np.array(hidden_image)
        hidden_image = np.mean(hidden_image, axis=0)
        gen_image = np.array(gen_image)
        gen_image = np.mean(gen_image, axis=0)
        image = x.numpy()

        # revert transforms.ToTensor() scaling
        image = (image*255)[0]
        hidden_image = (hidden_image*255)[0]
        gen_image = (gen_image*255)[0]

        image = np.reshape(image, (28, 28))
        hidden_image = np.reshape(hidden_image, (50, 50))
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

        print('generated images for digit %d' % (n))

    plt.tight_layout()
    plt.savefig('images/RBM_digits.jpg', dpi=20)
