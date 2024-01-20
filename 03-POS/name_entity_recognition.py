#named entities are used for a lot of things, such as: 
#1. information extraction
#2. question answering
#3. chatbots
#4. text classification
#5. sentiment analysis
#6. etc.

#named entity recognition is a type of entity recognition that groups entities into categories such as person, organization, location, etc.
#spacy does a very good job with named entity recognition but you can also do a custom named entity recognition
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

#def is a function that takes a document and returns the named entities in the document
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')

#example with no named entities
doc1 = nlp(u'Hi how are you?')
print(show_ents(doc1))

doc2 = nlp(u'May I go to Washington, DC next May to see the Washington Monument?')
print(show_ents(doc2))

doc3 = nlp(u'Can I please have 500 dollars of Microsoft stock?')
print(show_ents(doc3))

#normally, you would want spacy to build a library of named entities by training it on a corpus of text (several samples of text) but for this example, we are only adding one value
doc4 = nlp(u'Tesla to build a U.K. factory for $6 million')
print(show_ents(doc4))

#adding Tesla as an entity
from spacy.tokens import Span

#doc4: This is a SpaCy document object created by processing a text string with SpaCy's NLP model. The document (doc4) contains processed information about the text, such as tokens, their linguistic features, and any entities that were recognized.

#.vocab: This refers to the vocabulary associated with doc4. The vocabulary is a comprehensive data structure that SpaCy uses to store information about the language, including details about words, their meanings, linguistic annotations, and entity types. It's a collection of all the types of data that SpaCy knows about.

#.strings: This is a specific part of the vocabulary. The strings attribute in SpaCy is a table that works as a bidirectional lookup system: it maps string texts to unique integer IDs (hashes) and vice versa. This system allows SpaCy to handle text data efficiently. Every unique word, entity label, part-of-speech tag, etc., that SpaCy encounters is stored in this table with a unique hash.

#[u'ORG']: This part is a lookup operation in the strings table. It's retrieving the unique hash value associated with the string 'ORG'. In SpaCy, entity types like 'ORG' for organizations are represented internally by these hash values. By doing this lookup, you're fetching the internal representation of the entity type 'Organization'.

ORG = doc4.vocab.strings[u'ORG'] #ORG is the entity label that is predefined in spacy, we are just assigning it to a variable
new_ent = Span(doc4, 0, 1, label=ORG) #0 is the start index, 1 is the end index, and label is the entity label
doc4.ents = list(doc4.ents) + [new_ent] #adding the new entity to the list of entities
print(show_ents(doc4))

#adding multiple entities (vacuum cleaner and vacuum-cleaner)
doc5 = nlp(u'Our company created a brand new vacuum cleaner.'
        u'This new vacuum-cleaner is the best in show.')
show_ents(doc5)

#creating a new entity
##go over this, its a bit confusing
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab) #nlp.vocab is the vocabulary of the nlp object
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]
matcher.add('newproduct', None, *phrase_patterns) #None is a callback function, *phrase_patterns is a list of patterns
found_matches = matcher(doc5)
print(found_matches)

from spacy.tokens import Span
PROD = doc5.vocab.strings[u'PRODUCT'] #PRODUCT is the entity label that is predefined in spacy, we are just assigning it to a variable
new_ents = [Span(doc5, match[1], match[2], label=PROD) for match in found_matches] #match[1] is the start index, match[2] is the end index, and label is the entity label
doc5.ents = list(doc5.ents) + new_ents #adding the new entity to the list of entities
show_ents(doc5)

#how many times money was mentioned here
doc6 = nlp(u'Originally I paid $29.95 for this car toy, but now it is marked down by 10 dollars.')
[ent for ent in doc6.ents if ent.label_ == 'MONEY']
show_ents(doc6)






