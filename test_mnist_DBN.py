import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from PIL import Image
import cv2
import os

from DBN import DBN

def image_beautifier(names, final_name):
    image_names = sorted(names)
    images = [Image.open(x) for x in names]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(final_name)
    img = cv2.imread(final_name)
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    cv2.imwrite(final_name, img)

def gen_displayable_images():
    suffix = '_image.jpg'
    for n in range(10):
        prefix = 'images_DBN/digitwise/'+str(n)+'_'
        names = ['original', 'hidden', 'reconstructed']
        names = [prefix+name+suffix for name in names]
        image_beautifier(names, 'images_DBN/%d.jpg' % (n))

if __name__ == '__main__':
    os.makedirs('images_DBN/digitwise', exist_ok=True)

    test_dataset = datasets.MNIST('dataset', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    for test_x, test_y in test_loader:
        test_x = test_x.view(-1, 784)

    layers = [512, 128, 64, 10]
    dbn = DBN(test_x.shape[1], layers)
    dbn.layer_parameters = torch.load('models/mnist_trained_dbn.pt')
    
    for n in range(10):
        x = test_x[np.where(test_y==n)[0][0]]
        x = x.unsqueeze(0)
        gen_image, hidden_image = dbn.reconstructor(x)
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

        prefix = 'images_DBN/digitwise/%d_' % (n)
        suffix = '_image.jpg'
        
        plt.cla()
        plt.imshow(image, cmap='gray')
        plt.title('original image')
        plt.savefig(prefix+'original'+suffix)

        plt.cla()
        plt.imshow(hidden_image, cmap='gray')
        plt.title('hidden image')
        plt.savefig(prefix+'hidden'+suffix)

        plt.cla()
        plt.imshow(gen_image, cmap='gray')
        plt.title('reconstructed image')
        plt.savefig(prefix+'reconstructed'+suffix)

        print('generated images for digit %d' % (n))

    gen_displayable_images()
