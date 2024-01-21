#text generation with Keras 
#First half includes process the text, celan it, and tokenize and create sequences with keras
def read_file(filepath):
    with open(filepath) as f:
        str_text = f.read()
    return str_text

read_file('moby_dick_four_chapters.txt')

#Step 1: clean and tokinzation
import spacy 
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
nlp.max_length = 1198623 

def seperate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in ['']]
