#non-negative Matrix Factorization is an unsupervised algorithm that simultaneously performs dimensionality reductio and clustering
#it can be used in conjunction with TF-IDF scheme to model topics across documents
#it is similar to LDA but faster and more interpretable
#in an analogy: NMF takes a complex set of data (like recipes) and breaks it down into simpler, meaningful components (flavor profiles and their amounts in each recipe). This is useful for understanding the underlying structure or themes in the data, like figuring out what basic flavors make up a collection of recipes.

#the NMF follows these steps 
##1) construct vector space model for documents (after stopword filtering) resulting in a term-document matrix A
##2) apply TF-IDF weight normalization to A
##3) normalize TF-IDF vectors to unit length
##4) initilize factors using NNDSVD on the rows of A (NNDSVD is a method for non-negative matrix factorization)
##5) Apply projected gradient NMF to the matrix A (this is the actual NMF algorithm)

#in return we get:
## basis vectors: the topics (clusters) in the data 
## coefficients: the membership weights for documents in each topic (cluster)

#note that just like LDA, we have to select the number of topics and interpret the results

import pandas as pd
npr = pd.read_csv('npr.csv')
print(npr.head()) #note that there is no lables (topics), its just a bunch of text


#STEP 1: Preprocessing (CountVectorizer)
    ##we need to change this part to ____
    ### because LDA's dependece on per-word-count probabilities with Dirichlet priors, we can only use count vectorizers for LDAs
    ### however, we can use TF-IDF vectorizers for NMFs because they are not probabilistic models

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(npr['Article'])
print(dtm) #this is the document term matrix (DTM) that we will be using for NMF

#STEP 2: NMF
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=7, random_state=42)
nmf_model.fit(dtm)

#STEP 3: Displaying Topics
for index, topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print('\n')

#STEP 4: Attaching Discovered Topic Labels to Original Articles
topic_results = nmf_model.transform(dtm)
print(topic_results.shape) #this is the shape of the topic results
print(topic_results[0]) #this is the first article and its topic results
print(topic_results.argmax(axis=1)) #this is the index of the highest value in the topic results, the axis=1 means that we are looking at the columns

#making a dictionary of the topics
topic_dict = {0:'health', 1:'election', 2:'legislation', 3:'politics', 4:'election', 5:'music', 6:'legislation'}
npr['Topic'] = topic_results.argmax(axis=1)
print(npr.head()) #this is the original dataframe with the topic results attached

#mapping the topic dictionary to the topic results
npr['Topic Label'] = npr['Topic'].map(topic_dict) #this is the topic dictionary mapped to the topic results
print(npr.head())