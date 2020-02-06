import sys
from nltk.tokenize import sent_tokenize
import nltk
tokenizer = nltk.data.load("tokenizers/punkt/polish.pickle")
f = open("wiki_50.txt","r")
text = f.read()
sent_tokenize_list = tokenizer.tokenize(text)
for line in sent_tokenize_list:
    print(line)
