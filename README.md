# digits-mnist-resnet

Made the python script / notebook for the beginner kaggle competition [Digits](https://www.kaggle.com/c/digit-recognizer), essentially using MNIST dataset.

Used Residual Network with ~99% accuracy. Residual Network reference I used to create my model - 
![link](https://raw.githubusercontent.com/lambdal/cifar10-fast/master/net.svg)

Also added softmax at the tail end of this model. Also, used the learning rate schedule in the one cycle policy ([link](https://medium.com/dsnet/the-1-cycle-policy-an-experiment-that-vanished-the-struggle-in-training-neural-nets-184417de23b9)). 
Implemented this using Pytorch.
