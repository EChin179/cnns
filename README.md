# machinelearning

A collection of machine learning notebooks and deep learning models.

**Table of Contents**

**self-driving-car** -- (in progress) my current neural network to steer a self driving car. currently makes it a quarter around the track before diving into the water.

## models
1. **cat-image-recognition** -- a neural network that recognizes cat pictures from non cat pictures
2. **emotion-detection-keras** -- a neural network built through Keras that detects 'happy' vs. 'not happy' faces
3. **face-verification** -- implements face verification and recognition through building the triplet loss function and loading a pretrained Inception model for encoding
4. **mnist-numbers** -- a neural network that classifies handwritten digits
5. **SIGNS-tf-cnn** -- a convolutional neural network that classifies the numbers 0 to 5 of the SIGNS dataset
6. **traffic-signs** -- a neural network that classifies and recognizes traffic signs

## other-labs
1. **art-generation-nst** -- implements the neural style transfer algorithm to generate novel artistic images
2. **car-detection-yolo** -- uses the YOLO algorithm to detect cars and objects on the road, setting up score-thresholding and non-max suppression
3. **cnn-setup** -- sets up the fundamental functions and building blocks that are required of CNNs (convolution, forward pass, forward and backward pooling, etc)
4. **comparing-initializations** -- a comparison of zero, random, and He initialization of parameters
5. **comparing-optimizations** -- a comparison of batch gradient descent, stochastic gradient descent, and mini-batch gradient descent
6. **comparing-regularization** -- a comparison of L2 regularization, dropout, and no regularization
7. **deep-neural-network-setup** -- sets up the fundamental functions and processes that are required of neural networks (forward propagation, backward propagation, initializing parameters, etc)
8. **gradient-check** -- implementing the process of gradient check to ensure back propagation works effectively
9. **linear-regression** -- a simple implementation of linear regression, and my first introduction to machine learning
10. **resnet-setup** - sets up the identity and convolutional blocks of resnet, and builds a 50-layer ResNet through Keras
11. **tensorflow** -- an introduction to tensorflow and its usage of variables, constants, and sessions (essentially a tensorflow tutorial)