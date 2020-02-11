#maxmatch

#import sample text
#f = open("japanese_sample_text.txt","r")
#text = f.read()
#text = text.replace("\n","")
text = "abd"
dict = ["a","b","abd"]
#start at first character in string

#chooses longest word in the dictionary that matches the word
#if no word matches, advance one character
#iterate
i = 0
while i < len(text):
    if text[i:i+1] in dict:
        print(text[i:i+1])
    elif text[i] not in dict:
        if text[i:i+2] in dict:
    i+=1
