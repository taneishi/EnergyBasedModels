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

Without Pre-Training:

epochs | test loss | train loss | test acc | train acc
---|---|---|---|---
1.0 | 1.793358325958252 | 1.7901512384414673 | 0.6681372549019607 | 0.672005772005772
2.0 | 1.703145980834961 | 1.6950132846832275 | 0.7593837535014005 | 0.7689033189033189
3.0 | 1.614591121673584 | 1.60787832736969 | 0.8499299719887955 | 0.8563492063492063
4.0 | 1.548156976699829 | 1.539191484451294 | 0.9173669467787114 | 0.9269119769119769
5.0 | 1.5276743173599243 | 1.5182831287384033 | 0.9369047619047619 | 0.9461760461760462

With Pre-Training:

epochs | test loss | train loss | test acc | train acc
---|---|---|---|---
1.0 | 1.5359452962875366 | 1.5310659408569336 | 0.9349439775910364 | 0.9391053391053391
2.0 | 1.514952540397644 | 1.5070991516113281 | 0.9525210084033613 | 0.9602813852813853
3.0 | 1.5090779066085815 | 1.4990419149398804 | 0.9563025210084034 | 0.9665584415584415
4.0 | 1.5039907693862915 | 1.4926044940948486 | 0.9602941176470589 | 0.9722943722943723
5.0 | 1.4975998401641846 | 1.4844372272491455 | 0.9669467787114846 | 0.9796536796536797

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

Without Pre-Training:

epochs | test loss | train loss | test acc | train acc
---|---|---|---|---
1.0 | 1.7656803131103516 | 1.7620763778686523 | 0.7411764705882353 | 0.745021645021645
2.0 | 1.647426724433899 | 1.6424834728240967 | 0.8315826330532213 | 0.837049062049062
3.0 | 1.6079597473144531 | 1.6017967462539673 | 0.8528011204481792 | 0.8601731601731601
4.0 | 1.541589617729187 | 1.533886194229126 | 0.9266806722689076 | 0.9341269841269841
5.0 | 1.52533757686615 | 1.5153533220291138 | 0.9397759103641457 | 0.9494949494949495

With Pre-Training:

epochs | test loss | train loss | test acc | train acc
---|---|---|---|---
1.0 | 1.5659916400909424 | 1.563258409500122 | 0.9289915966386555 | 0.9310966810966811
2.0 | 1.5216224193572998 | 1.5138438940048218 | 0.9504901960784313 | 0.9579004329004329
3.0 | 1.5103492736816406 | 1.4991555213928223 | 0.9566526610644258 | 0.9685425685425686
4.0 | 1.5037704706192017 | 1.489931583404541 | 0.9618347338935574 | 0.9756132756132756
5.0 | 1.4998645782470703 | 1.4846107959747314 | 0.9647759103641457 | 0.9790764790764791

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
generated images for digit 0
generated images for digit 1
generated images for digit 2
generated images for digit 3
generated images for digit 4
generated images for digit 5
generated images for digit 6
generated images for digit 7
generated images for digit 8
generated images for digit 9
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
generated images for digit 0
generated images for digit 1
generated images for digit 2
generated images for digit 3
generated images for digit 4
generated images for digit 5
generated images for digit 6
generated images for digit 7
generated images for digit 8
generated images for digit 9

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

