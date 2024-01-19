#feature extraction is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data
#vectorization is the general process of turning a collection of text documents into numerical feature vectors

#types of vectorization methods include count vectorization, tf-idf vectorization, and word embedding
##count vectorizer is a bag of words approach, it counts the number of times a word appears in a document and uses this value as its weight, counts the frequency of the words in the document

##TF_IDF Vectorizer is a combination of two different algorithms, the TF (Term Frequency) and the IDF (Inverse Document Frequency); the TF is the same as the count vectorizer, the IDF is a measure of how significant a term is in the entire corpus (collection of documents), the IDF is the log of the number of documents divided by the number of documents that contain the word w
###basically, the TF-IDF vectorizer is a measure of how important a word is to a document in a collection of documents
### TF-IDF allows us to understand the context of the words across an entire corpus of documents, instead of just looking at the frequency of the words in a single document


#at the base of any NLP, we need to first create a vocabulary of all the unique words in the corpus, this is called tokenization

import numpy as np
import pandas as pd #pandas help with data manipulation and analysis, for reading csv files
df = pd.read_csv('smsspamcollection.tsv', sep='\t') #read the clipboard and create a dataframe; the \t is the tab separator
print(df.head()) #check the first 5 rows of the dataframe

#we are going to take the raw text and vectorize it using the count vectorizer, before doing that its a good practice to check for missing values
print(df.isnull().sum()) #check if there is any missing data in the dataframe (in this case everything is 0 so its not null)
## you can also check for a string of missing space and not just a missing value 

#count the number of spam and ham messages
print(df['label'].value_counts())

#split the data into training and testing sets
from sklearn.model_selection import train_test_split 
X = df['message'] #create a series of the message column; X is capital because it represents a matrix
y = df['label'] #create a series of the label column; y is lowercase because it represents a vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #split the data into training and testing sets, the test size is 33% of the data

#performing count vectorization
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer() #create an instance of the count vectorizer

#FIT: fit the vectorizer to the data (build a vocabulary, count the number of words, and assign an ID to each word)
#TRANSFORM: transform the original text message to vector

##you can do it either this way as seperate steps Fit and Transfer:
count_vect.fit(X_train) #fit the vectorizer to the training data
X_train_counts1 = count_vect.transform(X_train) #transform the training data into a document term matrix (DTM)
print(X_train_counts1.shape) #check the shape of the DTM

##or all in one step - this is more efficient 
X_train_counts2 = count_vect.fit_transform(X_train) #fit and transform the training data all in one step 
print(X_train_counts2.shape) #check the shape of the DTM

#TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_vect = TfidfTransformer() #create an instance of the tfidf vectorizer
X_train_tfidf = tfidf_vect.fit_transform(X_train_counts2) #fit and transform the DTM from the count vectorizer into a tfidf vectorizer
print(X_train_tfidf.shape) #check the shape of the tfidf vectorizer

#combine the count vectorizer and the tfidf vectorizer into one step
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer() #create an instance of the tfidf vectorizer
X_train_tfidf2 = vectorizer.fit_transform(X_train) #fit and transform the DTM from the count vectorizer into a tfidf vectorizer
print(X_train_tfidf2.shape) #check the shape of the tfidf vectorizer

#train the classifier
from sklearn.svm import LinearSVC
clf = LinearSVC() #create an instance of the classifier
clf.fit(X_train_tfidf2, y_train) #fit the classifier to the tfidf vectorizer
print(clf.predict(vectorizer.transform(['Hi how are you doing today?']))) #predict the label of a new message

#build a pipeline which simplifies the entire process -- BEST PRACTICE
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())]) #create a pipeline with the tfidf vectorizer and the classifier
text_clf.fit(X_train, y_train) #fit the pipeline to the training data
predictions = text_clf.predict(X_test) #predict the labels of the test data
print(predictions) #print the predictions
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions)) #print the confusion matrix
print(classification_report(y_test, predictions)) #print the classification report


