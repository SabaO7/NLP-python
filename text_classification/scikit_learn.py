#scikit learn is a machine learning library for python
#estimator parameters are passed as arguments to the constructor of the estimator class which in simple terms mean that the parameters are passed as arguments to the class
#scikit has a function to split the data into training and testing data - very simple
#model fitting is the process of training the model on the training data and then using the test data to test the model

import numpy as np
import pandas as pd #pandas help with data manipulation and analysis, for reading csv files

df = pd.read_csv('smsspamcollection.tsv', sep='\t') #read the clipboard and create a dataframe; the \t is the tab separator
print(df.head()) #check the first 5 rows of the dataframe

#testing if there is any missing data 
print(df.isnull().sum()) #check if there is any missing data in the dataframe (in this case everything is 0 so its not null)

#check the number of rows 
print(len(df)) #check the number of rows in the dataframe

#accessing the columns from a dataframe
print(df['label'].unique()) #access the label column and print the unique values in the column

#counting the number of spam and ham messages
print(df['label'].value_counts()) #count the number of spam and ham messages

#visualizing the data - 
import matplotlib.pyplot as plt #import the matplotlib library for visualization

##if we want to look at the length of the messages
plt.xscale('log') #scale the x axis to log scale
bins = 1.15**(np.arange(0,50)) #create the bins for the histogram
plt.hist(df[df['label']=='ham']['length'], bins=bins, alpha=0.8, color='green') #plot the histogram for ham messages
plt.hist(df[df['label']=='spam']['length'], bins=bins, alpha=0.8, color='yellow') #plot the histogram for spam messages
plt.legend(('ham', 'spam')) #add a legend to the plot
plt.show() #show the plot

##if we want to look at the punctuations in the messages
plt.xscale('log') #scale the x axis to log scale
bins = 1.5**(np.arange(0,15)) #create the bins for the histogram
plt.hist(df[df['label']=='ham']['punct'], bins=bins, alpha=0.8, color='green') #plot the histogram for ham messages
plt.hist(df[df['label']=='spam']['punct'], bins=bins, alpha=0.8, color='yellow') #plot the histogram for spam messages
plt.legend(('ham', 'spam')) #add a legend to the plot
plt.show() #show the plot

#splitting the data into training and testing data
from sklearn.model_selection import train_test_split #import the train_test_split function from sklearn
##x is the feature data 
X = df[['length', 'punct']] #create a dataframe with the length and punct columns, the reason we are using double square brackets is because we are passing a list of columns
##y is the label data 
y = df['label'] #create a dataframe with the label column

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42) #split the data into training and testing data, the test_size is the percentage of the data that will be used for testing, the random_state is the seed for the random number generator, the number is not that important, just keep it the same
print(X_train.shape) #check the shape of the training data
print(X_test.shape) #check the shape of the testing data

#training the ML model
from sklearn.linear_model import LogisticRegression #import the LogisticRegression class from sklearn
lr_model = LogisticRegression(solver='lbfgs') #create an instance of the LogisticRegression class (basically your model), the solver is the algorithm to use in the optimization problem, the lbfgs is the default solver, this can be customizable! 
lr_model.fit(X_train, y_train) #fit the model to the training data

#testing the accuracy of the model
from sklearn import metrics #import the metrics module from sklearn
predictions = lr_model.predict(X_test)  #predict the labels for the test data (new data) the lr_model.predict() function takes in the features of the test data and predicts the labels for the test data
print(predictions)

#we know the true values of the test data, so we can compare the predicted values with the true values to see how accurate the model is
# y_test
print(metrics.confusion_matrix(y_test, predictions)) #create a confusion matrix to see how many true positives, true negatives, false positives and false negatives there are

#since the accuracy is not great, we are going to add more features to the model
##creating the confusion metrix 
df2 = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['ham', 'spam'], columns=['ham', 'spam']) #create a dataframe from the confusion matrix
print(df2) #print the dataframe

##creating a classification report which gives you the precision, recall, f1-score and support!!!!
print(metrics.classification_report(y_test, predictions)) #create a classification report
print(metrics.accuracy_score(y_test, predictions)) #check the accuracy of the model for Logistic regression

#using a different model, note that everything is the same except for the model! easy, peasy lemon squeezy! :D
## although this model sucks!  
from sklearn.naive_bayes import MultinomialNB #import the MultinomialNB class from sklearn
nb_model = MultinomialNB() #create an instance of the MultinomialNB class (basically your model)
nb_model.fit(X_train, y_train) #fit the model to the training data
predictions = nb_model.predict(X_test) #predict the labels for the test data (new data) the nb_model.predict() function takes in the features of the test data and predicts the labels for the test data
print(metrics.confusion_matrix(y_test, predictions)) #create a confusion matrix to see how many true positives, true negatives, false positives and false negatives there are
print(metrics.classification_report(y_test, predictions)) #create a classification report

#another model: SVC is a support vector classifier
from sklearn.svm import SVC #import the SVC class from sklearn
svc_model = SVC(gamma='auto') #create an instance of the SVC class (basically your model), the gamma is the kernel coefficient for the rbf, poly and sigmoid kernels, the auto means that the gamma is calculated automatically
svc_model.fit(X_train, y_train) #fit the model to the training data
predictions = svc_model.predict(X_test) #predict the labels for the test data (new data) the svc_model.predict() function takes in the features of the test data and predicts the labels for the test data
print(metrics.confusion_matrix(y_test, predictions)) #create a confusion matrix to see how many true positives, true negatives, false positives and false negatives there are
print(metrics.classification_report(y_test, predictions)) #create a classification report
