import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

def plot(filenames):
    plt.figure(figsize=(12, 8))
    index = 1
    for filename in filenames:
        for metrics in ['loss', 'acc']:
            cols = ['%s %s' % (split, metrics) for split in ['test', 'train']]
            df = pd.read_csv(filename, usecols=cols)

            plt.subplot(2, 2, index)
            for col in cols:
                plt.plot(np.array(range(1, df.shape[0]+1)), df[col], label=col)
            plt.legend()
            plt.title('%s_%s' % (filename[8:-4], metrics))
            plt.grid(True)

            if metrics == 'acc':
                plt.ylim([-0.01, 1.01])

            index += 1

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    filenames = [
            'results/RBM_pretrained_classifier.csv',
            'results/RBM_without_pretraining_classifier.csv']
    
    plot(filenames)
    plt.savefig('images/RBM.jpg')

    filenames = [
            'results/DBN_without_pretraining_classifier.csv',
            'results/DBN_with_pretraining_classifier.csv']
    
    plot(filenames)
    plt.savefig('images/DBN.jpg')
