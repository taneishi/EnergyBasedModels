#!/bin/bash

if [ -d torch ]; then
    source torch/bin/activate
else
    python3 -m venv
    source torch/bin/activate
    pip install --upgrade pip
    pip install torch numpy pandas tqdm
fi

python MNIST_RBM_classifier_example.py
python MNIST_DBN_classifier_example.py
