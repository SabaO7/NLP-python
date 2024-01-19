import numpy as np
import pandas as pd

df = pd.read_csv('moviereviews.tsv', sep='\t')
print(df.head())
print(len(df))
print(df.isnull().sum()) #note that this time we are missing reviews!

df.dropna(inplace=True) #drop the missing reviews
print(df.isnull().sum()) #check again to make sure they are gone

#STEP 1) Cleaning up the dataset by removing blanks and nulls

##one way to remove blank reviews that are just spaces and not NA or null is to do this:
blanks = []  # start with an empty list
##(index, label, review text)
for i, lb, rv in df.itertuples():  # iterate over the DataFrame
    if rv.isspace():
        blanks.append(i)
print(blanks) #printing the index of the blank reviews

##next step is to remove the blanks
df.drop(blanks, inplace=True)
print(len(df)) #check the length of the dataframe; note that its reduced 

#STEP 2) splitting the data to train and test
from sklearn.model_selection import train_test_split
X = df['review'] #note that this is the review column
y = df['label'] #note that this is the label column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #random_state is the seed for the random number generator

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

#STEP 3) Creating a pipeline to vectorize the data and then perform classification
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())]) #note that the pipeline is a list of tuples

#STEP 4) Fitting the pipeline to the training data
text_clf.fit(X_train, y_train)

#STEP 5) Predicting the test data
predictions = text_clf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

#STEP 6) Printing the accuracy of the model
print(accuracy_score(y_test, predictions))

#STEP 7) Testing the model on a new review
print(text_clf.predict(['This movie is horrible!']))
print(text_clf.predict(['I really enjoyed the movie!']))