#!/bin/bash

pip install -qr requirements.txt

python mnist_RBM.py
python reconstruct_RBM.py

python mnist_DBN.py
python reconstruct_DBN.py

python draw_graphs.py
