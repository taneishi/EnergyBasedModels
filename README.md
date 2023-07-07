# Energy-Based Models in PyTorch

## Introduction

Energy-Based Models (EBM) are machine learning methods inspired by the concept of energy in physics.
Examples of EBMs include the Restricted Boltzmann Machine (RBM), the Deep Belief Networks (DBN) introduced by Geoffrey Hinton in 2006,
are well-known early implementations of deep learning.

EBM determines the dependencies between input and latent variables by associating them to scalar values representing the energy in the system.
It is based on the property that a low entropy state gives a better representation of the regularity of the subject.
Thus, the EBM can perform unsupervised learning.

Here I use PyTorch implementations of RBM and DBN to train and validate the accuracy of the image reconstruction task,
and then reconstruct the images from the trained model to confirm that the data has been learned well.

## Restricted Boltzmann Machine

RBM is a generative energy-based model consisting of two symmetric graphs.
The two graphs can be rephrased as a visible layer and a hidden layer.
Nodes in the visible layer correspond to input variables and nodes in the hidden layer correspond to latent variables.
Nodes in the same graph (i.e. in the same layer) have no connections, while all visible nodes and all hidden nodes have undirected connections.
This restriction that no connections in the same graph is what is meant by *restricted*.
Without this restriction, it is simply called Boltzmann Machine.
The lack of directionality in the connections allows for outputs from latent variables to input variables, thus RBM is a generative model.

The following shows the process of training handwritten numbers from 0 to 9 using RBM and the results of reconstructing the original handwritten numbers using the trained model.

<p align="center">
<img src="images/RBM.jpg" alt="RBM loss and acc" width="500px" />
<p>

```
Unsupervised pretraining of Restricted Boltzmann Machine
epoch   5/  5 train loss  0.068 12.8sec

Training without pretraining
epoch  1/ 5 train loss 1.684 train acc 0.789  6.3sec
epoch  2/ 5 train loss 1.584 train acc 0.880  6.4sec
epoch  3/ 5 train loss 1.517 train acc 0.947  6.4sec
epoch  4/ 5 train loss 1.506 train acc 0.957  6.4sec
epoch  5/ 5 train loss 1.498 train acc 0.965  6.3sec

Training with pretraining
epoch  1/ 5 train loss 1.560 train acc 0.918  6.4sec
epoch  2/ 5 train loss 1.500 train acc 0.965  6.4sec
epoch  3/ 5 train loss 1.488 train acc 0.976  6.4sec
epoch  4/ 5 train loss 1.481 train acc 0.982  6.4sec
epoch  5/ 5 train loss 1.477 train acc 0.986  6.4sec
```

<p align="center">
<img src="images/RBM_digits.jpg" alt="RBM digits" width="500px" />
</p>

## Deep Belief Networks

Multiple RBMs in the previous section can be stacked to form a network.
After unsupervised learning with multiple RBMs, supervised fine-tune can be performed using stochastic gradient descent (SGD) and back-propagation,
which are fundamental methods in deep learning.
DBN is a network with this configuration.

DBN is generative because it is derived from RBM, and it is a graphical model in which input variables are represented by connections of latent variables.
Inheriting the characteristics of RBM, there are connections between different layers, but no connections within the same layer.

In a stacked configuration of multiple RBMs, unsupervised learning can be performed for each RBM by applying contrast divergence (CD) to each RBM in order from the layer closest to the input.
In this case, the visible layer closest to the input is connected to the input data, and the hidden layer of each RBM is connected to the visible layer of the next RBM.

Like RBM, DBN learn to probabilistically reconstruct the input in unsupervised learning.
At this point, each RBM can be thought of as performing feature detection in each stage.
After unsupervised learning in each RBM, the entire DBN can be fine-tuned for supervised classification.

To add a historical note, there were known problems such as exploding gradient and vanishing gradient when learning deep neural networks,
but DBN could learn even with deep layers by pre-training in each RBM.
This is one of the reasons why DBN was used in early deep learning.

As with RBM, the learning process and reconstruction results are shown below.

<p align="center">
<img src="images/DBN.jpg" alt="DBN loss and acc" width="500px" />
</p>

```
Unsupervised pretraining of Deep Belief Network
epoch 100/100 train loss  0.085  1.9sec
Finished Training Layer: 0 to 1
epoch 100/100 train loss  0.165  0.5sec
Finished Training Layer: 1 to 2
epoch 100/100 train loss  0.187  0.4sec
Finished Training Layer: 2 to 3
epoch 100/100 train loss  0.244  0.4sec
Finished Training Layer: 3 to 4

Training without pretraining
epoch  1/ 5 train loss 1.813 train acc 0.667  4.579sec
epoch  2/ 5 train loss 1.563 train acc 0.905  4.735sec
epoch  3/ 5 train loss 1.516 train acc 0.949  4.724sec
epoch  4/ 5 train loss 1.504 train acc 0.958  4.741sec
epoch  5/ 5 train loss 1.497 train acc 0.965  4.755sec

Training with pretraining
The Last layer will not be activated. The rest are activated using the Sigmoid Function
epoch  1/ 5 train loss 1.702 train acc 0.820  4.721sec
epoch  2/ 5 train loss 1.507 train acc 0.960  4.714sec
epoch  3/ 5 train loss 1.493 train acc 0.971  4.727sec
epoch  4/ 5 train loss 1.485 train acc 0.978  4.698sec
epoch  5/ 5 train loss 1.480 train acc 0.983  4.733sec
```

<p align="center">
<img src="images/DBN_digits.jpg" alt="DBN digits" width="500px" />
</p>

## Applications

I applied DBN to a virtual screening task for drug discovery in 2014.
Virtual screening based on the Quantitative Structure-Activity Relationship (QSAR) is suitable for early deep learning because it deals with scalar vectors that represent drug candidate compounds as input.

## References

- Hinton et al., *A fast learning algorithm for deep belief nets*, Neural Computation, 2006.
- Theano Development Team, *Deep Learning Tutorials*, https://deeplearningtutorials.readthedocs.io/, 2008-2013.

