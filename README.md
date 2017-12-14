# MNIST Keras Model

## Description

This repository is implements a Convolutional Neural Network on the MNIST digits dataset. The model is further used for digit classification tasks in other projects.

### Process Dataset
The dataset is first converted from the native format to easy usable csv format. The first column corresponds to the labels and the remaining 784 (28*28) columns correspond to the pixels of the images in the dataset in both the training and test dataset csv files.


### CNN
In case of images, it is seen that a fully connected structure such as Multilayer Perceptron does not scale well. This is because, for every single neuron, we have nh x nw x nc weights, where nh is the height, nw is the width and nc is the number of channels in the image. The large number of weights in the network is not only quite wasteful but also quickly leads to overfitting.

Convolutional Neural networks take advantage of the structure of images, and constrain the architecture in a more sensible way. Unlike fully connected layers, they work on 3D volumes of neurons and the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner. This not only reduces the parameters, but also is more effective since a whole window of neighboring pixels/neurons is considered at a time, thus still maintaining some kind of a structure in the image. 

Usually 3 main types of layers exist while building a general ConvNet.

--- Convolutional Layer - Windows of filters slide over the 3D volume of neurons in the previous layer to generate the next layer using a convolution (dot) operation, usually with reduced nh, nw and (usually increased) nc which is equal to the number of filters used in this layer. Parameters associated with this layer are components of the filters. 

This operation is usually followed by an elementwise BatchNormalization operation and an activation function. Generally ReLU, leaky ReLU or swish are used as activation functions due to the speed of computation of forward pass and also gradients in backward pass.

--- Pooling Layer - Windows of MAX/AVG filters slide over the 3D volume of neurons in the previous layer, so as to downsample the spatial dimensions, keeping the most important information and reduce the number of operations in the subsequent layers. Usually either MAX or AVERAGE Pooling is used.

--- Fully Connected Layer - The 3D volume is usually converted into a FC layer, so that class scores can be computed through a SOFTMAX activation. Here, each neuron is connected to every other neuron in the previous layer, just like in Multilayer Perceptron.

### Architecture Used
[Input] -> [CONV32] -> [BN] -> [ReLU] -> [CONV32] -> [BN] -> [ReLU] -> [MAXPOOL] ->

-> [CONV64] -> [BN] -> [ReLU] -> [CONV64] -> [BN] -> [ReLU] -> [MAXPOOL] ->

--[Dropout]-[Flatten]-> [FC256] --[Dropout]-> [FC10] --[Softmax10]-> [Output]

### Model
The trained model is saved using *model.save(filepath)* into a single HDF5 file called MNIST\_keras\_CNN.h5 which contains:

-the architecture of the model, allowing to re-create the model
-the weights of the model
-the training configuration (loss, optimizer)
-the state of the optimizer, allowing to resume training exactly where you left off.

The model can be loaded from disk using *model = load_model(filepath)*

It is also saved using *model.to_yaml()* in a human-readable yaml file called MNIST\_keras\_CNN.yaml which contains the specification of the model. No weights are saved here, but they can be saved in a separate HDF5 file using the *model.save_weights(filepath)* function.

The model and the weights can then be loaded from the disk using *model = model_from_yaml(filepath)*
and *model.load_weights(filepath)*.

## Libraries used
Keras 2.1.2 with tensorflow backend
tensorflow 1.4.0
h5py
numpy
matplotlib.pyplot

## Authors
Kunal Jain
