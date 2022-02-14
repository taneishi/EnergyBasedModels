#!/bin/bash

if [ -d torch ]; then
    source torch/bin/activate
else
    python3 -m venv torch
    source torch/bin/activate
    pip install --upgrade pip
    pip install torch numpy pandas tqdm matplotlib Pillow opencv-python
fi

python MNIST_RBM_classifier_example.py
python test_MNIST_RBM_example.py

python MNIST_DBN_classifier_example.py
python test_MNIST_DBN_example.py

python draw_graphs.py
