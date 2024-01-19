import numpy as np
import pandas as pd

#Step 1) import the data
df = pd.read_csv('moviereviews.tsv', sep='\t')
print(df.head())

#Step 2) Clean the data
df.dropna(inplace=True) #drop the null values

##index, label, review
blanks = [] #start with an empty list
for i, lb, rv in df.itertuples(): #iterate over the dataframe
    if type(rv) == str: #if the review is in string format
        if rv.isspace(): #if the review is a space; meaning if the entire review is a space
            blanks.append(i) #add the index to the list

df.drop(blanks, inplace=True) #drop the reviews that are spaces

#Step 3) review through VADER - sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review)) #apply the polarity score to each review
print(df.head()) #note that the column scores is added to the dataframe

df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound']) #apply the compound score to each review
print(df.head()) #note that the column compound is added to the dataframe

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg') #if the compound score is greater than or equal to 0, then the review is positive, else it is negative
print(df.head())

#Step 4) Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(df['label'], df['comp_score'])) #the accuracy is 70% which is not bad
print(confusion_matrix(df['label'], df['comp_score'])) #the confusion matrix shows that we have 177 negative reviews that were predicted to be positive and 55 positive reviews that were predicted to be negative
print(classification_report(df['label'], df['comp_score'])) #the classification report shows that the model is better at predicting positive reviews than negative reviews


