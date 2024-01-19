import spacy 
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
doc1 = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million."
          u"By contrast, Sony only sold 8 thousand Walkman music players.")

#note that this part is for jupyter notebook
#printing the sentences by each sentence 
#for sent in doc1.sents:
#    displacy.serve(nlp(sent.text), style='ent', port=50001)

#filtering the entities and changing the color. Prettyyyyy!
colors = {'ORG': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)'} #this is for changing the color of the entity

#choosing to filter based on specific entities 
options = {'ents': ['PRODUCT', 'ORG'], 'colors': colors} #this is for filtering the entities
displacy.serve(doc1, style='ent', options=options, port=50001) #the style=ent is for entity recognizer