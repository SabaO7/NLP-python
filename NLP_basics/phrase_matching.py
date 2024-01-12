#phrase matching and vocabulary 
#its similar to regular expressions, but for tokens instead of strings and its much more powerful

import spacy
nlp = spacy.load('en_core_web_sm')

#token pattern for rule based matching 

#matcher is a tool that allows you to build a library of tokens, patterns, and rules and you can match those to a doc object to return a list of found matches
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# SolarPower example
## SolarPower (note the key value pair)
pattern1 = [{'LOWER': 'solarpower'}] #note that the pattern is a list of dictionaries, each dictionary is a token, and the keys are the attributes of the token, in this case, the attribute is the lower case of the token

## Solar-power (note the key value pair)
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}] #note that the pattern is a list of dictionaries, each dictionary is a token, and the keys are the attributes of the token, in this case, the attribute is the lower case of the token

## Solar power (note the key value pair)
pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}] #note that the pattern is a list of dictionaries, each dictionary is a token, and the keys are the attributes of the token, in this case, the attribute is the lower case of the token

#adding the patterns to the matcher
matcher.add('SolarPower',[pattern1, pattern2, pattern3]) #note that the first argument is the name of the pattern, the second argument is an optional callback, and the third argument is the patterns

doc = nlp(u'The Solar Power industry continues to grow as solarpower increases. Solar-power is amazing.')

found_matches = matcher(doc) #note that the matcher returns a list of tuples, each tuple is a match, the first element of the tuple is the match id, and the second element of the tuple is the start and end token index of the match
print(found_matches) 

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id] #note that nlp.vocab.strings is a dictionary that contains all of the string values of the match ids
    span = doc[start:end] #note that doc[start:end] is the matched span
    print(match_id, string_id, start, end, span.text) #note that span.text is the matched span


#removing a pattern from the matcher
matcher.remove('SolarPower') #note that the argument is the name of the pattern

# Let's create a pattern that will match the phrase "solar power plant"
##solarpower SolarPower Solar-power
pattern1 = [{'LOWER': 'solarpower'}]
##solar.power
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': '*'}, {'LOWER': 'power'}] #note that the OP key is the operator, and the value is the operator, the operator is the number of times the token can appear, in this case, the operator is *, which means that the token can appear 0 or more times
matcher.add('SolarPower', [pattern1, pattern2])


doc2 = nlp(u'Solar--power is solarpower yay!')
found_matches = matcher(doc2)
print(found_matches)

#Matching on lemmatization and phrase matching 
#mathc on a terminology list is a lot more efficient than matching on a list of patterns, therefore we need to use a phrase matcher in order to create a document object from a list of phrases and then pass that into the matcher instead 

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
with open('sample_text.txt') as f:
    doc3 = nlp(f.read())

phrase_list = ['life philosophy', 'theory', 'existential crisis']
phrase_patterns = [nlp(text) for text in phrase_list] #note that this is a list of document objects, each document object is a phrase
matcher.add('Philosophy', None, *phrase_patterns) #note that the first argument is the name of the pattern, the second argument is an optional callback, and the third argument is the patterns, the * is used to unpack the list of patterns
found_matches = matcher(doc3)
print(found_matches)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id] 
    span = doc3[start-5:end+5] 
    print(match_id, string_id, start, end, span.text) 