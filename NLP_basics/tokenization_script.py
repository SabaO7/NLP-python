# prefix is the character(s) at the beginning (i.e. $, (, “)
# suffix is the character(s) at the end (i.e. km, !)
# infix is the character(s) in between (i.e. -, --, /)
# exception is the character(s) that should not be split (i.e. U.K.)
# note that tokens cannot be re-assigned 
# take a look at this https://spacy.io/usage/visualizers for the visualization of the dependecy tree and the entity recognizer

import spacy
from spacy import displacy


nlp = spacy.load('en_core_web_sm')

mystring = '"We\'re moving to L.A.!"' # the \ is an escape character
print(mystring)

doc = nlp(mystring)
for token in doc:
    print(token.text, end=' | ') #isolating punctuation by itself is useful for text cleaning

#another example of tokenization with complex punctuation
doc2 = nlp(u"We're here to help! Send snail-mail, email support@email.com or visit at http://www.support.com!")
for t in doc2:
    print(t)

#another example of tokenization with complex punctuation
doc3 = nlp(u"A 5km NYC cab ride costs $10.30")
for t in doc3: 
    print(t)

#another example of tokenization with complex punctuation
doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")
for t in doc4:
    print(t)


len(doc4) #checking the length of the doc
print(len(doc4))

doc5 = nlp(u'It is better to give than to receive.')
doc5[0] #grabbing the first token
print(doc5[0])

#spacy recognizes name entities
doc8 = nlp(u'Apple to build a Hong Kong factory for $6 million')
for token in doc8:
    print(token.text, end=' | ')

for entity in doc8.ents:
    print(entity)
    print(entity.label_) #checking the label of the entity
    print(str(spacy.explain(entity.label_))) #checking the explanation of the label
    print('\n') #adding a space

#noun chunks
doc9 = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")
for chunk in doc9.noun_chunks: #the answer would be autonomous cars, insurance liability, manufacturers 
    print(chunk)

#displaying the dependency tree, the style='dep' is for dependency tree
doc10 = nlp(u"Apple is going to build a U.K. factory for $6 million.")
displacy.render(doc10, style='dep', jupyter=True, options={'distance': 300}) #distance is the distance between the words, this will be better visualized if you use Jupyter notebook
print(doc10)

#displaying the entity recognizer, the style='ent' is for entity recognizer
#again this will be visualized much better if you use Jupyter notebook
doc11 = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.")
displacy.render(doc11, style='ent', jupyter=True)
print(doc11)

#for displaying the visualization outside of jupyter notebook, you can use the following code
doc12 = nlp(u"Apple is going to build a U.K. factory for $6 million")
displacy.serve(doc12, style='dep', port=5001, options={'distance': 300}) #this will open a new tab in your browser and display the dependency tree
print(doc12)

doc13 = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.")
displacy.serve(doc13, style='ent', port=5002) #this will open a new tab in your browser and display the entity recognizer
print(doc13)


