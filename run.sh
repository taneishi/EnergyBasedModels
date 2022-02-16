#!/bin/bash

if [ -d torch ]; then
    source torch/bin/activate
else
    python3 -m venv torch
    source torch/bin/activate
    pip install --upgrade pip
    pip install torchvision numpy pandas matplotlib Pillow opencv-python
fi

python mnist_RBM.py
python test_mnist_RBM.py

python mnist_DBN.py
python test_mnist_DBN.py

python draw_graphs.py
