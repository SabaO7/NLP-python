#defintion for keras is a high level neural network API which is capable of running on top of tensorflow, theano, or CNTK

import numpy as np
from sklearn.datasets import load_iris

#load the iris dataset
iris = load_iris()
print(iris.DESCR)

X = iris.data #numpy arrary of features
y = iris.target #numpy array of labels (species of iris flower)

#hot encode the labels, which is a way of representing categorical data
## for example, if we have 3 species of flowers, we can represent them as 0, 1, and 2
## class 0 --> [1, 0, 0]; class 1 --> [0, 1, 0]; class 2 --> [0, 0, 1]

from keras.utils import to_categorical
y = to_categorical(y)

#split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

#for NN, its often a good idea to scale or standardize the data
from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler() # this basically scales the data between 0 and 1 (normalization), i.e. if i had an array of 12, 25, 10, it would divide each number by 25 (the highest number) to get 0.48, 1, 0.4
scaler_object.fit(X_train) #you should always fit it to the training data
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test) 

#STEP 1: now we can create the neural network with keras
from keras.models import Sequential
from keras.layers import Dense #dense means fully connected

##this is the part where we define the architecture of the model
model = Sequential() #this is the most common model, which is a feed forward model
model.add(Dense(8, input_dim=4, activation='relu')) #this is the first hidden layer, 8 neurons (its a multiply of the features, and its domain base on what that multiply number should be), 4 inputs (because we have 4 features), relu activation function (rectified linear unit)
model.add(Dense(8, input_dim=4, activation='relu')) #this is the second hidden layer, 8 neurons, 4 inputs, relu activation function
model.add(Dense(3, activation='softmax')) #this is the output layer, 3 neurons (because we have 3 classes), softmax activation function (because we are doing classification, since there are 3 classes, we want to know the probability of each class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #this is the part where we compile the model, we have to specify the loss function, optimizer, and metrics, we are using categorical_crossentropy because we are doing classification, adam is a type of optimizer, and we are using accuracy as our metric
model.summary() #this prints out a summary of the model
model.fit(scaled_X_train, y_train, epochs=150, verbose=2) #this is the part where we train the model, we have to specify the training data, the labels, and the number of epochs (how many times we want to go through the training data), verbose is just how much information we want to see, 0 is nothing, 1 is a progress bar, and 2 is just the epochs

#why did we pick 2 hidden layer in this case? because we have 4 features, so we want to have a number of neurons that is a multiple of 4, and we want to have 2 hidden layers because we want to have a number of hidden layers that is between 1 and 2 times the number of input layers, so we picked 2 hidden layers; in simple terms, we want to have a number of hidden layers that is between 1 and 2 times the number of input layers, and we want to have a number of neurons that is a multiple of the number of input layers

#STEP 2: how to predeict in unseen data and evaluate it with keras
# Use the predict method to get the probabilities
predictions = model.predict(scaled_X_test)

# Convert the probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Now you can print the predictions and compare them with the actual labels
print(predicted_classes)
print(y_test.argmax(axis=1))

from sklearn.metrics import confusion_matrix, classification_report

# Continue with the confusion matrix and classification report
print(confusion_matrix(y_test.argmax(axis=1), predicted_classes))
print(classification_report(y_test.argmax(axis=1), predicted_classes))

#STEP 3: how to save and load a model with keras
model.save('myfirstmodel.h5') #this saves the model as a .h5 file, the naming needs to be done properly
from keras.models import load_model
newmodel = load_model('myfirstmodel.h5') #this loads the model from the .h5 file 

