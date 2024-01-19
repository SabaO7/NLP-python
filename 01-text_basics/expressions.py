##REGEX is important, there are many more things to learn about it
import re 

text1 = "The film Titanic was released in 1997"

# Find all the years mentioned in the text
years = re.findall(r"\d{4}", text1)
print(years)

#finding the pattern for a string "phone number" and finding the span of it 
text2 = "The phone number of the agent is 408-555-1234. Call soon!"
pattern = "phone number"
print(re.search(pattern, text2))
my_match = re.search(pattern, text2)
print(my_match.span())

# Finding the pattern "phone" in the text
text3 = "The phone number of the agent is 408-555-1234. Call soon! where is your phone?"
pattern2 = "phone"

# Find all occurrences of the pattern
my_match2 = re.findall(pattern2, text3)
print(my_match2)

# Print the length of the list containing the matches
print(len(my_match2))

# Using finditer to get match objects and printing their spans
for match in re.finditer(pattern2, text3):
    print(match.span())

## finding a general pattern for phone number
text4 = "The phone number of the agent is 408-555-1234. Call soon!"
pattern3 = r"\d\d\d-\d\d\d-\d\d\d\d"
my_match3 = re.search(pattern3, text4)
print(my_match3.group())

#another way to find this pattern 
pattern4 = r"\d{3}-\d{3}-\d{4}"
my_match4 = re.search(pattern4, text4)
print(my_match4.group())

#finding the postal code in a text
text5 = "My Postal Code is A0B 1C2"
pattern5 = r"[A-Z]\d[A-Z] \d[A-Z]\d"
my_match5 = re.search(pattern5, text5)
print(my_match5.group())

#finding the first three digit of the postal code could be done by using parenthesis
pattern6 = r"([A-Z]\d[A-Z]) \d[A-Z]\d"
my_match6 = re.search(pattern6, text5)
print(my_match6.group(1))

#getting all the numbers from the text
text6 = "There are 2 numbers 34 inside 5 this sentence."
pattern7 = r"\d+"
print(re.findall(pattern7, text6))

#ignoring all the numbers from the text
pattern8 = r"[^\d]+"
print(re.findall(pattern8, text6))

##this part is important for NLP!

#removing a punctuation from the text
text7 = "This is a string! But it has punctuation. How can we remove it?"
pattern9 = r"[^!.?]+"
print(re.findall(pattern9, text7))

#joining the list back together
print(" ".join(re.findall(pattern9, text7)))

#finding a hyphenated word in the text
text8 = "Only find the hypen-words in this sentence. But you do not know how long-ish they are"
pattern10 = r"[\w]+-[\w]+"
print(re.findall(pattern10, text8))








