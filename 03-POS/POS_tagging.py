import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u"The quick brown fox jumped over the lazy dog's back")

# POS tagging, the spacy.explain() function gives the meaning of the tag
#VERY IMPORTANT: token.pos_ gives the coarse grain tag and token.tag_ gives the fine grain tag
for token in doc:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")

#example of fine grain tagging
doc1 = nlp(u"I read books on NLP.")
word = doc1[1]
word.text
token = word
print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")

#example of fine grain tagging - past tense, it understand its because it looks at the context!
doc2 = nlp(u"I read a book on NLP.")
word = doc2[1]
word.text
token = word
print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")

#spacy.attrs is a dictionary of all the POS tags and you can use it to count the number of POS tags in a document
doc3 = nlp(u"The quick brown fox jumped over the lazy dog's back")
POS_counts = doc3.count_by(spacy.attrs.POS)
POS_counts #dictionary of POS tags and their counts
doc.vocab[83].text

#dictionary comprehension
for k,v in sorted(POS_counts.items()):
    print(f"{k}. {doc3.vocab[k].text:{5}}: {v}")

#spacy.attrs.tag is a dictionary of all the dependency tags and you can use it to count the number of dependency tags in a document 
TAG_counts = doc3.count_by(spacy.attrs.TAG)
for k,v in sorted(TAG_counts.items()):
    print(f"{k}. {doc3.vocab[k].text:{5}}: {v}")

#spacy.attrs.dep is a dictionary of all the dependency tags and you can use it to count the number of dependency tags in a document
DEP_counts = doc3.count_by(spacy.attrs.DEP) #spacy.attrs.DEP is the dependency tag
for k,v in sorted(DEP_counts.items()):
    print(f"{k}. {doc3.vocab[k].text:{5}}: {v}")

