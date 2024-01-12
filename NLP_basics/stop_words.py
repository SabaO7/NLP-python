#stop words are words that are so common that they are not useful for the task in hand, examples include "the", "a", "is" etc.

import spacy 

nlp = spacy.load('en_core_web_sm')

#checking the stop words and printing them out
print(nlp.Defaults.stop_words) 

#checking the number of the stop words
print(len(nlp.Defaults.stop_words))

#checking if the word is a stop word
print (nlp.vocab['is'].is_stop)

# making a word a stop word
nlp.Defaults.stop_words.add('btw')
nlp.vocab['btw'].is_stop = True

len(nlp.Defaults.stop_words) #note that the number of stop words increased by 1
print(nlp.vocab['btw'].is_stop) #note that the word btw is now a stop word

#removing a stop word
nlp.Defaults.stop_words.remove('beyond')
nlp.vocab['beyond'].is_stop = False

len(nlp.Defaults.stop_words) #note that the number of stop words decreased by 1
print(nlp.vocab['beyond'].is_stop) #note that the word beyond is now not a stop word

