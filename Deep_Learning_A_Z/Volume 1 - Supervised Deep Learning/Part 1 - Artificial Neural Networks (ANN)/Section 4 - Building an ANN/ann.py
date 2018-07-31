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


