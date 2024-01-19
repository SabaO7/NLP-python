#lemmatization is the process of converting a word to its base form, its better than stemming because it converts the word to its meaningful base form, rather than just stripping the word of its suffixes

import spacy

nlp = spacy.load('en_core_web_sm') #loading the model

doc1 = nlp(u'I am a runner running in a race because I love to run since I ran today')

for token in doc1:
  print(token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_) #note that lemma is the number of the lemma, whereas lemma_ is the actual lemma

#create a function to show the lemma
def show_lemma(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}') #note that the f string is a new way of formatting strings, it is called literal string interpolation, it is a way to format strings that allows you to embed python expressions inside of them, it is a new way of formatting strings that is more readable, more concise, and less prone to error, it is also faster than the other methods of formatting strings


# Call the function with doc1
print("Lemmas for doc1:")
show_lemma(doc1)

doc2 = nlp(u'I saw ten mice today!') #note that saw is a verb, and mice is a noun, but the lemma of saw is not see, and the lemma of mice is not mouse, this is because spacy is not a dictionary, it is a statistical model, so it is not always correct, but it is correct most of the time

show_lemma(doc2)