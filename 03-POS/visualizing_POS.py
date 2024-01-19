import spacy
nlps = spacy.load('en_core_web_sm')
doc1 = nlps(u"The quick brown fox jumped over the lazy dog's back")

from spacy import displacy

#displacy.render is for using jupyter notebook, wheras displacy.serve is for viewing in the browser

#changing the styl of the visualization
options = {'distance': 110, 'compact': 'True', 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'} 
displacy.serve(doc1, style='dep', port=5001, options=options) #the style=dpe is for dependency parsing

#displacy.serve is meant to accept a single document or a list of document. Since large text are difficult to view online, you can pass a list of spans instead 
doc2 = nlps(u"This is a sentence. This is another sentence, possibly longer than the other.")
spans = list(doc2.sents) #list of spans, spans are like sentences
displacy.serve(spans, style='dep', port=5002, options=options) #the style=dpe is for dependency parsing