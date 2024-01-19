#after tokenization which is turning a sentence into a list of words, you can then conver the text to vectors either through TF-IDF vectors (which looks at the frequency of the text and the importance of the word; it does not account for sentiment or the context of the sentence, just looks at words) or word embedding (looks at the context of the sentence and the sentiment of the sentence) 

# word2vec is a model that is used to create word embeddings, its a two-layer neural net that processes text; its purpose is to group the vectors of similar words together in vectorspace
# when words become vectors, in spacy, each of these vectors are 300 dimensional vectors!!! you can build your own and the dimentions can range 100-1000 but its time consuming and computationally expensive
# once you have the vectors, you can then compare them to see how similar they are to each other by using cosine similarity

import spacy 
nlp = spacy.load('en_core_web_lg') #load the model, this model has word vectors!
print(nlp(u'lion').vector) #this is the vector for the word lion
print(nlp(u'Lion').vector.shape) #this is the shape of the vector (note that its 300)

#document or longer sentences use the average of the vectors of each words in the sentence; therefore formating may be different but the dimensions are the same 
print(nlp(u'The quick brown fox jumped over the lazy dog').vector.shape) #this is the shape of the vector (note that its 300)

#this is the similarity between the words using cosine similarity
##note that the similarity value is between 0 and 1, 1 being the most similar and 0 being the least similar
tokens = nlp(u'lion cat pet')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2)) #this is the similarity between the words

#note that words that are often used in the same context but have an opposite meaning may have a similar vectors
tokens = nlp(u'like love hate')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2)) #this is the similarity between the words

print(len(nlp.vocab.vectors)) #this is the number of unique vectors in the model (514157)
tokens = nlp(u'dog cat saby') #saby is not in the model
for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov) #this is the similarity between the words
    ##token.text is the word, token.has_vector is a boolean that tells you if the word has a vector, token.vector_norm is the norm of the vector, token.is_oov is a boolean that tells you if the word is out of vocabulary
    ###token.vector_norm is the square root of the sum of the squares of the values in the vector, its used to normalize the vector so that the vector is between 0 and 1
    ####normalizing is not mandatory but recommended, when using cosine similarity is highly recommended to normalize the vectors

#vector arithmetics - creating a new vector by adding or subtracting known vectors 
from scipy import spatial #we do this because we need to calculate cosine similarity ourselves

cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2) #this is the formula for cosine similarity
##lambda is a keyword in Python used to create small anonymous functions. These functions are called "anonymous" because they don't need to be named. They are useful for creating quick, short functions that are not too complex.
## Cosine similarity is the complement of cosine distance. It is calculated as 1 - cosine distance.
##By subtracting the cosine distance from 1, you get a measure of similarity instead of difference.


king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

#now we are going to ask for the king - man + woman --> new_vector
new_vector = king - man + woman 
computed_similarities = []
for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha: #this is to make sure that the word is not a number or a punctuation
                similarity = cosine_similarity(new_vector, word.vector) #the new_vector is the vector that we created above, and the word.vector is the vector of the word that we are comparing to which is every word in the model
                computed_similarities.append((word, similarity)) #append is a function that adds an element to the end of the list

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1]) #this is to sort the list by the second element in the tuple (the similarity value)
print([t[0].text for t in computed_similarities[:10]]) #this is to print the first 10 words in the list