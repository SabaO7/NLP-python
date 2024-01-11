import spacy


### Great resource: https://spacy.io/usage/spacy-101#annotations
### Great resource: https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf

##Token text is the original word text; lemma_ is the base form of the word (i.e. running -> run); 
# pos_ is the simple part-of-speech tag (i.e. ADJ for adjective) [if you do .pos it will give you the number of the part of speech];
# tag_ is the detailed part-of-speech tag (i.e. will know what part of the speech it is, adjective, noune etc.);
# dep_ is syntactic dependency (i.e. amod for the adjective modifier);
# shape_ is the word shape – capitalization, punctuation, digits (i.e. Xxxx for a four-letter word);
# is_alpha is True if the token consists of alphabetic characters only (i.e. True for “run”, False for “200m”);
# is_stop is True for stop words (i.e. True for “the”, False for “butter”

nlp = spacy.load('en_core_web_sm') #loading the model

doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion') #creating a doc object
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        token.shape_, token.is_alpha, token.is_stop)
    

nlp.pipeline #checking the pipeline
print (nlp.pipe_names) #checking the pipeline names [['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']]

##tokenization
doc2 = nlp(u"Tesla isn't   looking into startups anymore.") #note that space will have its own token 
for token in doc2:
    print(token.text, token.pos_, token.dep_)

##grabbing the first token
print(doc2[0].text)

##for longer sentences, paragraphs, or documents, we can use the .sents method
doc3 = nlp(u'After tokenization, spaCy can parse and tag a given Doc. This is where the trained pipeline and its statistical models come in, which enable spaCy to make predictions of which tag or label most likely applies in this context. A trained component includes binary data that is produced by showing a system enough examples for it to make predictions that generalize across the language – for example, a word following “the” in English is most likely a noun.')
quote = doc3[12:43] #the 12th token to the 43rd token
print(quote)

#Spacy will know this is only part of the entire document and will provide the output as Span. Span just means a slice of the document.
type(quote)
print(type(quote))

#whereas if we want to get the entire thing, it will be a Doc
type(doc3)
print(type(doc3))

#sends will give us each of the sentences
doc4 = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
for sentence in doc4.sents:
    print(sentence)

#we can also check if a token is the start of a sentence
doc4[6].is_sent_start #checking if the 6th token is the start of a sentence
print(doc4[6].is_sent_start)



