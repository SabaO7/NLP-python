#VADER stands for Valence Aware Dictionary and sEntiment Reasoner. it is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. It is available in the nltk package.
##VADER is smart enough to understand basic onctext of the text, it also understand capitalization and punctuation

#sentiment score is a measure of how positive or negative the text is. It is a float which lies in the range of -1 to 1, where -1 is highly negative and +1 is highly positive.
##the sentiment score of a text can be obtained by summing up the intensity of each word in the text, and that is called the document sentiment score.

#sentiment analysis on raw text its very difficult because of two reasons 
##1) having both positive and negative sentiment in the same text data 
##2) sarcasim
## for example, the word "sick" can be used to describe something that is cool, or something that is not well.

#sentiment analysis VADER in python NLTK

##there might some firewall for downloading the vader lexicon, so we need to do some workaround
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')

#the VADAR Sentiment Intensity Analyzer returns a dictionary of scores in each of four categories: negative, neutral, positive, and compound (computed by normalizing the scores above)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Simple Examples:
a = "This is a good movie"
print(sid.polarity_scores(a)) # note that the min and max values are -1 and 1, respectively

b = "This was the best, most awesome movie EVER MADE!!!"
print(sid.polarity_scores(b))

c = "This was the WORST movie that has ever disgraced the screen."
print(sid.polarity_scores(c))

#Real life Example with amazon reviews (note that this has the lable of the review as well)
#Step 1) import the data
import pandas as pd
df = pd.read_csv('amazonreviews.tsv', sep='\t')
print(df.head())
print(df['label'].value_counts()) #we have more negative reviews

#Step 2) Clean the data 
df.dropna(inplace=True) #drop the null values

##index, label, review
for i, lb, rv in df.itertuples(): #iterate over the dataframe
    if type(rv) == str: #if the review is in string format
        if rv.isspace(): #if the review is a space; meaning if the entire review is a space
            df.drop(i, inplace=True) #drop the review

#Step 3) review through VADER
print(sid.polarity_scores(df.iloc[0]['review'])) #polairty score of the first review

#Step 4) add the scores to the dataframe
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review)) #apply the polarity score to each review
print(df.head()) #note that the column scores is added to the dataframe

##getting the compound score on its own as thats the one we need
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound']) #apply the compound score to each review
print(df.head()) #note that the column compound is added to the dataframe

#Step 5) convert the compound score to a binary score 
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg') #if the compound score is greater than or equal to 0, then the review is positive, else it is negative
print(df.head())

#Step 6) check the accuracy of the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(df['label'], df['comp_score'])) #the accuracy is 70% which is not bad
print(classification_report(df['label'], df['comp_score'])) #the precision and recall are not bad either
print(confusion_matrix(df['label'], df['comp_score'])) #the confusion matrix is not bad either
