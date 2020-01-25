# machinelearning

A collection of machine learning notebooks and deep learning models.
Kaggle: https://www.kaggle.com/evelynchin

**Table of Contents**

**self-driving-car** -- (IN PROGRESS) my current neural network to steer a self driving car. currently makes it a quarter around the track before diving into the water. 

Self Driving Car Video Demo Update: https://youtu.be/UuoOCzKG2l0 

## models
1. **cat-image-recognition** -- a neural network that recognizes cat pictures from non cat pictures
2. **dinosaur-name-generator-rnn** -- a character level language model that generates new dinosaur names, as well as a shakespeare poem generator :)
3. **emojify** -- chooses an emoji to go along with an inputted sentence using a 2 layer LSTM model and embedding matrices
4. **emotion-detection-keras** -- a neural network built through Keras that detects 'happy' vs. 'not happy' faces
5. **face-verification** -- implements face verification and recognition through building the triplet loss function and loading a pretrained Inception model for encoding
6. **[KAGGLE] house-prices-advanced-regression** -- implements ensembles and stacked models to predict house prices based on tens of features
7. **jazz-improvisation-lstm** -- uses an LSTM network to generate a jazz solo improvisation through Keras and a 78-value musical encoding system
8. **mnist-numbers** -- a neural network that classifies handwritten digits
9. **SIGNS-tf-cnn** -- a convolutional neural network that classifies the numbers 0 to 5 of the SIGNS dataset
10. **traffic-signs** -- a neural network that classifies and recognizes traffic signs

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
10. **neural-machine-translation-with-attention** -- builds an NMT model that translates human-readable dates into machine-readable dates using the attention model
11. **operations-on-word-vectors** -- implementing cosine similarity, solving analogy problems, and modifying word embeddings to reduce their gender bias
12. **resnet-setup** -- sets up the identity and convolutional blocks of resnet, and builds a 50-layer ResNet through Keras
13. **rnn-setup** -- sets up individual RNN and LSTM cells, and implements the forward propagation process of a recurrent neural network and an LSTM
14. **tensorflow** -- an introduction to tensorflow and its usage of variables, constants, and sessions (essentially a tensorflow tutorial)
15. **trigger-word-detection** -- synthesizes and processes audio recordings to create train/dev datasets and trains a trigger word detection model to make predictions
