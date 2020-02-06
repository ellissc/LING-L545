import csv
def maxmatch(text, dictionary):
    if len(text) == 0: #if space, return nothing
        return ""
    for i in range(len(text),-1, -1): #reverse for loop of sentence
        firstword = text[:i] #first part of the sentence
        remainder = text[i:] # remainder
        if firstword in dictionary: #reverse shrinks scope of sentence till it finds the longest word in the sentence that is in the dictionary
            return firstword + ' ' + maxmatch(remainder, dictionary)#returns found word and iterates to remainder
    firstword = text[0] #no word was found on the first try, so returns one-character word
    remainder = text[1:]
    return firstword + ' ' + maxmatch(remainder, dictionary)

f = open("nonum_dict.txt","r")
text = f.read()
dict_array = []
word = ''
for char in text:
    if char == '\n':
        dict_array.append(word)
        word = ''
    else:
        word = word + char

g = open("japanese_sample_text.txt","r")
sample = g.read()

print(maxmatch(sample, dict_array))

#with open("japanese_dict.csv","w",newline="") as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerows(dict_array)