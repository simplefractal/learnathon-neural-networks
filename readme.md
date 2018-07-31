## Introduction to Neural Networks

![machine-learning-logo](https://www.recordedfuture.com/assets/machine-learning-cybersecurity-applications.png)

1. What is Machine Learning?

	Machine learning is a subset of artificial intelligence in the field of computer science that often uses statistical techniques to give computers the ability to "learn" (i.e., progressively improve performance on a specific task) with data, without being explicitly programmed.

2. What is an Artificial Neural Network aka ANN?

	Artificial Neural Networks are created as a means of doing machine learning in which a computer learns to perform some task by analyzing training examples, or datasets. It is modeled loosely on the human brain, where a neural net consists of thousands or millions of simple processing nodes that are densely interconnected.

## NEURAL NETWORK FACTS

* neural networks along with deep learning have been along for quite some time
* impacting the world now
* invented in 60s and 70s
* caught wind in the 80s
* technology back then was not up to par to facilitate neural networks
* in order to work properly u need A LOT of data and A LOT of processing power(computing power)
* in 1980 a 10MB hard drive was $3495
* exponential curve when it comes to storage
* at the rate were going we may be going into DNA storage
* 1 kg of DNA can store ALL of the worlds data
* entering the era of computers that can process things way faster than we can
* imagine, by 2045 we will have computing power thats faster than all human brains combined in the world
* Geoffrey Hinton is godfather of deeplearning
* we wanna be able to mimic the human brain(most powerful for learning)

## HOW DO NEURAL NETWORKS LEARN

* can learn using hard coded conditions
* for machine to determine between a cat and dog consider the following
- to be determined as cat, look for whiskers, look for certain shape, look for pointy ears
- to be determined as dog, look for sloppy ears, drool, barking

## LAYERS

a basic neural network consists of three or more layers

![layers](https://icdn5.digitaltrends.com/image/artificial_neural_network_1-791x388.jpg)

	* INPUT layer
	* HIDDEN layer(s)
	* OUTPUT layer

* input layers hit the hidden layers(many many nodes) which then gets sent to output
* input values are processed via hidden layers and their activation functions, then get go to output


## ARTIFICIAL NEURAL NETWORK

* 3 input values
* x1, x2, xm
* goes through neuron node
* y output value produce
* each input value has weights
* neuron has an activation function, which is applied to the weighted sum of the inputs
* then the neuron passes on the signal to the next neuron

## ACTIVATION FUNCTIONS

* each node in the hidden layer has an activation function - here are some popular functions
	- threshold
	- sigmoid
	- rectifier - one of the most used
	- hyperbolic tangent(tanh)

## EXAMPLE OF A NEURAL NETWORK

* BASIC CASE - determine price of a house
	- x1 = Area(ft^2)
	- x2 = Bedrooms
	- x3  = Distance to city(miles)
	- x4 = Age
	- PRICE =  w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4
		one particular node(neuron in hidden layer) might only care about two parameters like area or distance to city
		another node(neuron in hidden layer) might care about three other parameters


## CREATING A NEURAL NETWORK

* create a facility for program to understand what to do on its own
* code the program to train itself what a dog or cat is and it will determine on its own
* these are two fundamentally different approaches
* back propagation -
	in order for neural network to learn this must happen - when summation of y hat - y values squared is back propagated through neural network and then weights are adjusted accordingly

## LINEAR REGRESSION MODEL

* in simple linear regression an independent variable is used to predict a dependent variable

* we use this model in artificial neural networks

## GRADIENT DESCENT

* brute forcing doing millions of combinations to adjust weights in neural network is not efficient at all and cannot be done with more complex networks so in neural networks the gradient descent algorithm is used

* is just one way to learn the weight coefficients of a linear regression model

* optimization algorithm where weights are updated incrementally after each epoch(pass over training dataset)

* best way to figure out weights while minimizing cost function

* object - the cost function(something like the sum of squared errors or SSE) must be minimized which will then increase accuracy of prediction

* go from solving a problem in 10^57 years to minutes or hours

* two types of gradient descent
	- batch gradient descent - pass entire data set through network and adjust weight, deterministic algorithm
	- stochastic gradient descent - pass each row of data set through network and adjust weight, helps produce better output, actually faster as well because it does not have to load up all data into memory(lighter algo), random algorithm vs batch

## STEPS TO TRAINING A.N.N. WITH STOCHASTIC GRADIENT

	1. Randomly initialize the weights to small numbers close to 0(but not 0)

	2. Input the first observation of your dataset in the input layer, each feature in one input node

	3. Forward-Propogation: from left to right, the neurons are activated in a way that the impact of each neuron’s activation is limited by the weights. Propagate the activations until getting the predicted result y.

	4. Compare the predicted result to the actual result.
	Measure the generated error.

	5. Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.

	6. Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning). OR: Repeat Steps 1 to 5 but update the weights only after a batch of observations (Batch Learning).

	7. When the whole training set passed through the ANN, that makes an epoch. Redo more epochs so that neural network can adjust and minimize error and cost function

## LINEAR REGRESSION

* basic formula for linear regression is y = b0 + b1 * x
* find best fit line for distribution of data
* predicted y values are the same for the expected linear regression so that the learning rate decides by how much we update the weights

## PRACTICAL EXPERIMENT

For this experiment I used Spyder which is a great IDE for machine learning. Also needed the following packages:
* pandas
* numpy
* matplotlib
* keras
* scikit

Objective - to build an Artificial Neural Network

Problem - the bank has given us 10,000 lines of data that trains on bank customer data which includes the features below as well as whether or not the customer has left the bank. In order to to best serve the bank we will build an ANN to predict probability of a customer leaving the bank based on the features given:
* credit score
* geography
* age
* tenure
* balance
* number of products
* has credit card
* active
* estimated salary

![data](/screenshots/data.png)

This data must first be encoded, since our program will only understand numbers and scaled or normalized to comparable numbers (between -1 and 1) so that their magnitudes are relatively similar.

![scaled](/screenshots/scaled.png)



```
# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps theano

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# adding the input layer and the first hidden layer
# choosing the optimal number of hidden layer nodes is purely preference
# a good rule of thumb is to create an average of input layers and output layer to decide on #
# this customization is also known as paramater tuning
# we will create six nodes
# relu is the activation function we chose(rectifier) because it is one of the most commonly used
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))

# adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))

# adding the output layer
# we must change activation function to sigmoid function - helps to get probabilities
# we only have binary output so sigmoid works here otherwise we will need to use softmax
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# compiling the ANN
# applying stochastic gradient descent
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# fit ANN to training set
# pass in arguments - matrix of features, then dependent variable vector
# no rule of thumb to choose batch_size and # of epochs
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Part 3 - Making predictions and evaluating the model

# predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

## RESULT
After feeding 8000 lines of training data to the network, we used 2000 lines to test. The program hit an accuracy of 86% in predicting whether a customer will exit the bank or not! Very cool!
