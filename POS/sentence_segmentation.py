import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.language import Language
#from spacy.pipeline import SentenceSegmenter - old version

doc1 = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
#you cant print sentences individually using sents[0] 
for sent in doc1.sents: #sents is a generator meaning, it generates and doesnt hold in memory; This is also why doc.sents[0]
    print(sent)
    print('\n')

#in order to get the first sentence, you need to use indexing and list
print(list(doc1.sents)[0])

doc2 = nlp(u'"Management is doing the right thing; leadership is doing the right things." -Peter Drucker')
#default setting of sentence segmentation is to split on punctuation
for sent in doc2.sents:
    print(sent)
    print('\n')

#add a segmentation rule or replace a rule
##approach 1) add a segmentation rule
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc2):
    for token in doc2[:-1]: #exclude the last token
        if token.text == ';':
            doc2[token.i+1].is_sent_start = True #token.i is the index of the token, meaning the next token is the start of a sentence
    return doc2

nlp.add_pipe("set_custom_boundaries", before="parser") #before='parser' means before the dependency parser

nlp.pipe_names #check the pipeline

#note that the new pipeline is saved and the sentences are split after ; as well 
doc3 = nlp(u'"Management is doing the right thing; leadership is doing the right things." -Peter Drucker')
for sent in doc3.sents:
    print(sent)
    print('\n')


##approach 2) change the segmentation rule to split on new line characters
###spliting a sentence on new lines rather than the . 
nlp = spacy.load('en_core_web_sm') #reseting the pipeline
mystring = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."
print(mystring)
doc4 = nlp(mystring)
for sent in doc4.sents:
    print(sent)

###note that you needed to import sentence segmenter from spacy.pipeline
@Language.component("split_on_newlines")
def split_on_newlines(doc4):
    start = 0 #start at the beginning of the document
    seen_newline = False #we havent seen any new lines yet
    for word in doc4: #for loop to iterate over each word in the document
        if seen_newline: #if we have seen a new line
            yield doc4[start:word.i] #this is a generator #return the text from start to the word before this one 
            start = word.i #update the start index to the current word
            seen_newline = False #reset the newline flag
        elif word.text.startswith('\n'): #if the current word starts with a new line character then we dont do anything - this is for the end of the sentence
            seen_newline = True
    yield doc4[start:]

nlp.add_pipe("split_on_newlines", before="parser") #add the sentence segmenter to the pipeline

print(nlp.pipe_names) #check the pipeline
