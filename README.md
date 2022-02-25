# Energy Based Models in PyTorch

The aim of this repository is to create Energy Based Models (EBM) in generalized manner, so as to allow modifications and variations.

## Restricted Boltzmann Machine

Energy-Based Models are a set of deep learning models which utilize physics concept of energy.
They determine dependencies between variables by associating a scalar value, which represents the energy to the complete system.

* It is a probabilistic, unsupervised, generative deep machine learning algorithm.
* It belongs to the energy-based model
* Restricted Boltzmann Machine (RBM) is undirected and has only two layers, Input layer, and hidden layer
* No intralayer connection exists between the visible nodes. 
* All visible nodes are connected to all the hidden nodes

In an RBM, we have a symmetric bipartite graph where no two units within the same group are connected.
Multiple RBMs can also be stacked and can be fine-tuned through the process of gradient descent and back-propagation.
Such a network is called a Deep Belief Network.

The above project allows one to train an RBM and a Deep Belief Network (DBN) in PyTorch on both CPU and GPU.
Finally let us take a look at some of the reconstructed images.

## Results of RBM

![RBM-Acc](./images/RBM_acc.jpg)
![RBM-Loss](./images/RBM_loss.jpg)

'''
Unsupervised pre-training of RBM
epoch   0 loss  0.068
epoch   1 loss  0.066
epoch   2 loss  0.068
epoch   3 loss  0.069
epoch   4 loss  0.069

Training without pre-training
epoch 1 train loss 1.948 train acc 0.512
epoch 2 train loss 1.582 train acc 0.884
epoch 3 train loss 1.516 train acc 0.949
epoch 4 train loss 1.503 train acc 0.961
epoch 5 train loss 1.495 train acc 0.968

Training with pre-training
epoch 1 train loss 1.563 train acc 0.917
epoch 2 train loss 1.499 train acc 0.966
epoch 3 train loss 1.488 train acc 0.977
epoch 4 train loss 1.482 train acc 0.982
epoch 5 train loss 1.477 train acc 0.986
'''

## Deep Belief Networks

In machine learning, a Deep Belief Network (DBN) is a generative graphical model, or alternatively a class of deep neural network, composed of multiple layers of latent variables ("hidden units"), with connections between the layers but not between units within each layer.

When trained on a set of examples without supervision, a DBN can learn to probabilistically reconstruct its inputs.
The layers then act as feature detectors. After this learning step, a DBN can be further trained with supervision to perform classification.

DBNs can be viewed as a composition of simple, unsupervised networks such as restricted Boltzmann machines (RBMs) or autoencoders, where each sub-network's hidden layer serves as the visible layer for the next.
An RBM is an undirected, generative energy-based model with a "visible" input layer and a hidden layer and connections between but not within layers. This composition leads to a fast, layer-by-layer unsupervised training procedure, where contrastive divergence is applied to each sub-network in turn, starting from the "lowest" pair of layers (the lowest visible layer is a training set).

The observation that DBNs can be trained greedily, one layer at a time, led to one of the first effective deep learning algorithms.
Overall, there are many attractive implementations and uses of DBNs in real-life applications and scenarios (e.g., electroencephalography, drug discovery).

### Results of DBN

![DBN-Acc](./images/DBN_acc.jpg)
![DBN-Loss](./images/DBN_loss.jpg)

'''
Unsupervised pre-training of DBN
epoch   0 loss  0.169
epoch  10 loss  0.109
epoch  20 loss  0.101
epoch  30 loss  0.096
epoch  40 loss  0.093
epoch  50 loss  0.091
epoch  60 loss  0.089
epoch  70 loss  0.088
epoch  80 loss  0.087
epoch  90 loss  0.086
Finished Training Layer: 0 to 1
epoch   0 loss  0.302
epoch  10 loss  0.206
epoch  20 loss  0.193
epoch  30 loss  0.185
epoch  40 loss  0.180
epoch  50 loss  0.176
epoch  60 loss  0.172
epoch  70 loss  0.170
epoch  80 loss  0.168
epoch  90 loss  0.166
Finished Training Layer: 1 to 2
epoch   0 loss  0.358
epoch  10 loss  0.249
epoch  20 loss  0.227
epoch  30 loss  0.213
epoch  40 loss  0.205
epoch  50 loss  0.200
epoch  60 loss  0.195
epoch  70 loss  0.192
epoch  80 loss  0.189
epoch  90 loss  0.187
Finished Training Layer: 2 to 3
epoch   0 loss  0.406
epoch  10 loss  0.277
epoch  20 loss  0.263
epoch  30 loss  0.260
epoch  40 loss  0.256
epoch  50 loss  0.252
epoch  60 loss  0.250
epoch  70 loss  0.244
epoch  80 loss  0.239
epoch  90 loss  0.234
Finished Training Layer: 3 to 4
Without Pre-Training
epoch 1 train loss 1.837 train acc 0.649
epoch 2 train loss 1.583 train acc 0.887
epoch 3 train loss 1.521 train acc 0.944
epoch 4 train loss 1.508 train acc 0.956
epoch 5 train loss 1.500 train acc 0.962
With Pre-Training
epoch 1 train loss 1.683 train acc 0.844
epoch 2 train loss 1.507 train acc 0.960
epoch 3 train loss 1.493 train acc 0.972
epoch 4 train loss 1.485 train acc 0.978
epoch 5 train loss 1.480 train acc 0.983
'''

## Images of RBM

<img src="images_RBM/0.jpg" alt="image 0" width="300" />
<img src="images_RBM/1.jpg" alt="image 1" width="300" />
<img src="images_RBM/2.jpg" alt="image 2" width="300" />
<img src="images_RBM/3.jpg" alt="image 3" width="300" />
<img src="images_RBM/4.jpg" alt="image 4" width="300" />
<img src="images_RBM/5.jpg" alt="image 5" width="300" />
<img src="images_RBM/6.jpg" alt="image 6" width="300" />
<img src="images_RBM/7.jpg" alt="image 7" width="300" />
<img src="images_RBM/8.jpg" alt="image 8" width="300" />
<img src="images_RBM/9.jpg" alt="image 9" width="300" />

## Images of DBN

<img src="images_DBN/0.jpg" alt="image 0" width="300" />
<img src="images_DBN/1.jpg" alt="image 1" width="300" />
<img src="images_DBN/2.jpg" alt="image 2" width="300" />
<img src="images_DBN/3.jpg" alt="image 3" width="300" />
<img src="images_DBN/4.jpg" alt="image 4" width="300" />
<img src="images_DBN/5.jpg" alt="image 5" width="300" />
<img src="images_DBN/6.jpg" alt="image 6" width="300" />
<img src="images_DBN/7.jpg" alt="image 7" width="300" />
<img src="images_DBN/8.jpg" alt="image 8" width="300" />
<img src="images_DBN/9.jpg" alt="image 9" width="300" />

