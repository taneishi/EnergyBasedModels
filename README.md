# Energy-Based Models

## Introduction

*Energy-based model*, EBM is a machine learning method based on the concept of energy in physics. Examples of EBM include *restricted Boltzmann machine*, RBM, and *deep belief network*, DBN, which was introduced by *G.E. Hinton* in 2006 and is known as one of the earliest examples of *deep learning*, DL[^Hinton].

EBM determines the dependencies between input and latent variables by associating them to real values representing the energy in the system. This is based on the physical law that a low entropy state gives a better representation of the regularity of the subject. Thus, EBM is essentially *unsupervised learning*, which can be interpreted as achieving the regularity underlying the subject, that is, *feature extraction*. This is one of the basic ideas on which DL was established, although in a different form afterwards.

Here, we take the `MNIST` dataset, which gives the problem of classifying handwritten digits from 0 to 9, as an example to introduce RBM and DBN to perform feature extraction as *pretraining*. And we reconstruct images from the trained models to visually confirm how the features underlying the dataset were extracted.

## Restricted Boltzmann Machine

RBM is one of the simplest EBM consisting of two graphs. The two graphs are called *visible layer* and *hidden layer*, respectively, from the side closer to the input. Nodes in the visible layer correspond to input variables and nodes in the hidden layer correspond to latent variables. Nodes in the same graph, i.e. in the same layer, have no connections, while all visible nodes are connected to all hidden nodes and all hidden nodes are connected to all visible nodes. Therefore, the connections between these layers are symmetric and have no direction.

Having no connections in the same graph is what is meant by *restricted*, while allowing the connections in the same graph is *Boltzmann machine*. Since the connections are not directional, the computation in RBM is reversible, i.e., it is possible to propagate from latent variables to input variables in the opposite direction of learning. As a consequence, RBM is a *generative model*.

First, the classification is performed by *supervised learning* using the smallest *multilayer perceptron*, MLP, which consists of a single hidden layer. The weights from the input layer to the hidden layer are initialized by random values. Next, we pretrain the weights from the input layer to the hidden layer in unsupervised learning using RBM, and assign the weights to the hidden layer of MLP for supervised learning. Therefore, it is necessary to align the same number of units in the hidden layers of MLP and RBM. This supervised learning after pretraining is called *fine-tuning*.

Actually, comparing the accuracy up to 5 epochs with and without pretraining, the accuracy on the test set without pretraining is 0.937, while the accuracy with pretraining is 0.967, confirming that pretraining improves the accuracy.

The reconstructed images of the handwritten digits used for input with this pretrained RBM is shown in Fig 1.

![RBM_digits](figure/RBM_digits.jpg)

**Figure 1. Reconstructed handwritten digits image with pretrained RBM.**

## Deep Belief Network

Even before DL, it was known that multilayer neural networks have the potential to learn more complex subjects because they are hierarchical and have many learning parameters. On the other hand, multilayer neural networks have issues such as *vanishing gradient* and *overfitting*, making it difficult to effectively progress the learning process.

DBN is one of the answers to these problems, where each layer of a multilayered neural network corresponds to RBM, and unsupervised pretraining is performed in each RBM to extract hierarchical features step by step. Unsupervised learning is performed in the order from the layer closest to the input, with the visible layer of the RBM closest to the input being directly connected to the input, and the visible layers of the other RBM being connected to the hidden layers of the RBM on the side closer to the input. In this pretrained multilayer neural network is less prone to vanishing gradient and overfitting due to the feature extraction at each stage. A pretrained MLP with a single RBM can be considered a minimal DBN.

DBN is a generative model because they are derived from RBM, and it is also a *graphical model* in which input variables are represented by the combination of latent variables.

Comparing the accuracies obtained by training up to 5 epochs with DBN is 0.940 without pretraining and 0.965 with pretraining, and again, the accuracy is higher with pretraining. The images of the handwritten digit for input reconstructed with the pretrained DBN model is shown in Fig 2.

![DBN_digits](figure/DBN_digits.jpg)

**Figure 2. Reconstructed handwritten digits image with pretrained DBN.**

## Application

We applied DBN to a *virtual screening* for *drug discovery* in 2014, and the paper summarizing the results was published in 2017. A virtual screening based on *quantitative structure-activity relationship*, QSAR was suitable for early DL because it deals with real vectors representing drug candidate compounds as input.

[^Hinton]: G.E. Hinton et al., *A fast learning algorithm for deep belief nets*, **Neural Computation**, 2006.
