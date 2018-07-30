## Introduction to Neural Networks

1. What is Machine Learning?
  * Machine learning is a subset of artificial intelligence in the field of 	  
  computer science that often uses statistical techniques to give computers the ability to "learn" (i.e., progressively improve performance on a specific task) with data, without being explicitly programmed.
2. What is an Artificial Neural Network aka ANN?
  * Artificial Neural Networks are created as a means of doing machine Learning
	in which a computer learns to perform some task by analyzing training examples, or datasets. It is modeled loosely on the human brain, where a neural net consists of thousands or millions of simple processing nodes that are densely interconnected.

## DEEP LEARNING

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

## LAYERS

		INPUT
		HIDDEN
		OUTPUT
		input layers hit the hidden layers(billions of nodes) which then gets sent to output
		input values are processed via hidden layers then get to output

		ARTIFICIAL NEURAL NETWORK

		3 input values
		x1, x2, xm
		goes through neuron node
		y output value produce
		each input value has weights
		neuron has an activation function, which is applied to the weighted sum of the inputs
		then the neuron passes on the signal to the next neuron

		ACTIVATION FUNCTIONS

		popular functions
		threshold
		sigmoid
		rectifier - one of the most used
		hyperbolic tangent(tanh)
		QUESTION - if your output will be binary, which activation function would you use?
		a threshold activation function
		a sigmoid activation function

		HOW DO NEURAL NETWORKS WORK

		BASIC CASE - determine price of a house
		x1 = Area(ft^2)
		x2 = Bedrooms
		x3  = Distance to city(miles)
		x4 = Age
		Price = w1 * x1 + w2 * x2 + w3 *x3 + w4 *x4
		one particular node(neuron in hidden layer) might only care about two parameters like area or distance to city
		another node(neuron in hidden layer) might care about three other parameters

		HOW DO NEURAL NETWORKS LEARN

		hard coded conditions
		for input to be determine of cat, look out for whiskers, look out for certain shape, look for pointy ears
		for input to be determined as dog, look for sloppy ears, drool, bark etc
		create a facility for program to understand what to do on its own
		code the program to train itself what a dog or cat is and it will determine on its own
		these are two fundamentally different approaches
		back propagation***
		in order for neural network to learn this must happen - when summation of y hat - y values squared is back propagated through neural network and then weights are adjusted accordingly

		GRADIENT DESCENT

		cost function -> C = 1/2(y^ - y)^2
		brute forcing into doing millions of combinations to adjust weights in neural network is not efficient at all and cannot be done with more complex networks so…we have gradient descent method
		best way to figure out weights while minimizing cost function
		go from solving a problem in 10^57 years to minutes or hours
		TWO types of gradient descent
		batch gradient descent - pass entire data set through network and adjust weight, deterministic algorithm
		stochastic gradient descent - pass each row of data set through network and adjust weight, helps produce better output, actually faster as well because it does not have to load up all data into memory(lighter algo), random algorithm vs batch

		STEPS TO TRAINING A.N.N. WITH STOCHASTIC GRADIENT

		Randomly initialize the weights to small numbers close to 0(but not 0)
		Input the first observation of your dataset in the input layer, each feature in one input node
		Forward-Propogation: from left to right, the neurons are activated in a way that the impact of each neuron’s activation is limited by the weights. Propagate the activations until getting the predicted result y.
		Compare the predicted result to the actual result. Measure the generated error.
		Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.
		Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning). OR: Repeat Steps 1 to 5 but update the weights only after a batch of observations (Batch Learning).
		When the whole training set passed through the ANN, that makes an epoch. Redo more epochs so that neural network can adjust and minimize error and cost function

		LINEAR REGRESSION

		y = b0 + b1*x - this is the basic formula
		find best fit line for distribution of data
		predicted y values are the same for the expected linear regression so that the learning rate decides by how much we update the weights rep updates
