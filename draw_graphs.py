import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

def plot(filename, model, metrics, condition, split):
    df = pd.read_csv(filename)

    col = f'{split} {metrics}'

    metrics = 'Accuracy' if metrics == 'acc' else metrics
    condition = condition.replace('_', ' ')
    split = 'Training' if split == 'train' else 'Test'

    linestyle = '--' if split == 'Training' else '-'
    color = 2 if condition.startswith('without') else 1

    plt.plot(range(1, df.shape[0]+1), df[col], label=f'{split} {condition}',
            linestyle=linestyle, color=f'C{color}')
    plt.legend(fontsize=16)
    plt.xticks(range(1, df.shape[0]+1))
    plt.xlabel('Epoch', fontsize=16, weight='bold')
    plt.ylabel(metrics, fontsize=16, weight='bold')
    plt.title(model, fontsize=16, weight='bold')
    plt.grid(True)

    if metrics == 'Accuracy':
        plt.ylim([0, 1])

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    metrics = 'acc'
    conditions = ['without_pretraining', 'with_pretraining']

    for model in ['RBM', 'DBN']:
        plt.figure(figsize=(6, 4))

        for condition in conditions:
            for split in ['train', 'test']:
                plot(f'results/{model}_{condition}.csv',
                        model, metrics, condition, split)
       
        plt.tight_layout()
        plt.savefig(f'images/{model}.jpg', dpi=80)
