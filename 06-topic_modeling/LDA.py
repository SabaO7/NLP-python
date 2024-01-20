import pandas as pd
npr = pd.read_csv('npr.csv')
print(npr.head()) #note that there is no lables (topics), its just a bunch of text

#STEP 1: Preprocessing (CountVectorizer)
    ##in this step, the countvectorizer, first, tokenizes the words (meaning, breaks down big chunks of text to each word), second, it builds a vocabulary and then counts the frequency of each word 
from sklearn.feature_extraction.text import CountVectorizer
##max_df is the max document frequency (getting rid of the terms that are really across a lot of the documents), min_df is the min document frequency; it can be any number between 0 and 1, meaning 0.2 is 20% of the documents
##min_df is the min document frequency (getting rid of the terms that are really across a lot of the documents), min_df is the min document frequency; it can be any number between 0 and 1, meaning 0.2 is 20% of the documents, however, if you use integers, it means the number of documents
### for both min_df and max_df, you can use integers or any number between 0 and 1
## stop_words='english' means that you are removing the stop words
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')

#STEP 2: Fit transform; note that the fit_transform is a combination of fit and transform
    ##fitting is indexing each word from the CountVectorizer (which topkenized the words)
    ## transform does the vectorization 

##note that fit_transform uses the a lot of the work that CountVectorizer did  
##note that we are NOT using test_train_split because this is unsupervised learning!

dtm = cv.fit_transform(npr['Article']) #dtm is document term matrix, npr['Article'] is the column of the text
print(dtm) #note that this is a sparse matrix, meaning that there are a lot of zeros

#STEP 3: perform LDA
from sklearn.decomposition import LatentDirichletAllocation

## if you wanted to have more subtopics you would increase the number of components
        ### Understanding LDA.fit(dtm):
        ### 1) LDA (Latent Dirichlet Allocation):
            ### LDA is a statistical model used for topic modeling. It aims to discover abstract topics within a collection of documents.
        
        ### 2) The DTM (Document-Term Matrix):
            ### dtm stands for Document-Term Matrix, which you previously created using CountVectorizer. In this matrix, each row represents a document, and each column represents a unique word in your vocabulary. The values in the matrix are the counts of how often each word appears in each document.
        
        ### 3)The fit Method:
            ### fit is a method used in machine learning to train the model on the provided data. In the context of LDA.fit(dtm), it means you are training the LDA model on your Document-Term Matrix.
            ### During this training process, LDA tries to:
                ### Identify patterns of word distributions that form topics.
                ### Determine which topics are present in each document and in what proportion.
LDA = LatentDirichletAllocation(n_components=7, random_state=42) #n_components is the number of topics, random_state is the randomly selected seed
print(LDA.fit(dtm)) #note that this takes a while to run

#STEP 4: Grab the vocabulary of words
print(len(cv.get_feature_names_out())) #note that there are 54777 words in the vocabulary
print(cv.get_feature_names_out()[5000]) #note that this is the 5000th word in the vocabulary

##randomly grab words from the vocabulary
    ###cv.get_feature_names_out()[random_word_id] fetches the word from the vocabulary at the randomly selected index, and then it's printed out.

import random
random_word_id = random.randint(0,54777)
print(cv.get_feature_names_out()[random_word_id])

#STEP 5: Grab the topics
print(len(LDA.components_)) #note that there are 7 topics
len(LDA.components_[0]) #note that there are 54777 words in the first topic

## grabing a single topic
    ###single_topic = LDA.components_[0] assigns the first topic's word-importance array to single_topic.
single_topic = LDA.components_[0]

### argsort() is a numpy array method that returns the index positions sorted from least to greatest
    ###single_topic.argsort() sorts the indices of the words in single_topic based on their importance (or frequency). The least important words are first, and the most important are last
print(single_topic.argsort()) #note that this is the index position of the words sorted from least to greatest 

### if the top 10 words dont give you a clear picture, increase the number of words
top_ten_words = single_topic.argsort()[-10:] #note that this is the index position of the top 10 words sorted from least to greatest
for index in top_ten_words:
    print(cv.get_feature_names_out()[index]) #note that this is the top 10 words

#STEP 6: Grab the highest probability words per topic
for index, topic in enumerate(LDA.components_): #enumerate() is a python function that returns the index position and the object itself
    print(f"The top 15 words for topic #{index}")
    print([cv.get_feature_names_out()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')

#STEP 7: Attaching Discovered Topic Labels to Original Articles
topic_results = LDA.transform(dtm)
print(topic_results[0].round(2)) #note that this is the probability of each topic for the first article
