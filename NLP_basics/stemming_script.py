#steming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form
#spacy doesnt have steming, but nltk does
#stemming is not always perfect, but it is fast
#stemming is a crude heuristic that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes
# there are 5 phases of reduction algorithm in nltk stemmer
# what you should know and care about is that snowball stemmer is better than porter stemmer

import spacy

#porter stemmer
from nltk.stem.porter import PorterStemmer #importing the stemmer
p_stemmer = PorterStemmer() #creating an instance of the stemmer
words = ['run', 'runner', 'ran', 'runs', 'easily', 'fairly'] #creating a list of words
for word in words:
    print(word + '------>' + p_stemmer.stem(word)) #stemming each word in the list

#snowball stemmer is better than porter stemmer
from nltk.stem.snowball import SnowballStemmer #importing the stemmer
s_stemmer = SnowballStemmer(language='english') #creating an instance of the stemmer, note that this requires you to identify which language you are using
words = ['run', 'runner', 'ran', 'runs', 'easily', 'fairly'] #creating a list of words
for word in words:
    print(word + '------>' + s_stemmer.stem(word)) #stemming each word in the list