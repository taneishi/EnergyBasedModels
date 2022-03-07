#!/bin/bash

pip install -r requirements.txt

python mnist_RBM.py
python test_mnist_RBM.py

python mnist_DBN.py
python test_mnist_DBN.py

python draw_graphs.py
