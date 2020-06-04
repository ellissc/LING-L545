# K-Means clustering of Classical Chinese Authors

Text file location:  "/mnt/c/Users/Ellis/Desktop/L445/LING-L545/final_project/ctexts/"
  under the variable "dir"

Conllu file location: "/mnt/c/Users/Ellis/Desktop/L445/LING-L545/final_project/ctexts/conllu_files/"
  under the variable dir_conllu

Add the udpipe model to path: export PATH=$PATH:/mnt/c/Users/Ellis/Desktop/L445/LING-L545/final_project/udpipe/src
  
## Function usage:

### txt2conllu(dir, dir_conllu)

Convert text files (from dir) into conllu files (to dir_conllu) using a trained udpipe model

dir should contain the texts, and clzh.model should be in the same file as this script

Clzh.model is the classical chinese udpipe model


### conllu2arr(dir_conllu)

reads conllu files (in dir_conllu) into python arrays (arrays kongzi, mengzi, liuxiang, dongzhongshu, zhuangzi, zhuangzi_test)


### ngramify(array)

extracts n-grams from a given array, n in range 3-8


### call_overlap(kongzi, kongzi_ngrams, mengzi,mengzi_ngrams, dongzhongshu,dongzhongshu_ngrams, liuxiang,liuxiang_ngrams, zhuangzi,zhuangzi_ngrams, zhuangzi_test,zhuangzi_test_ngrams)

calculates basic vocabulary overlap between the different authors, prints to command line


### conllu2tfidf(dir_conllu, n_low=2, n_high=10)

reads conllu files (from dir_conllu) and returns tfidf vectors and author labels. 
n_low and n_high designate the range for n-grams.
  

### k_means(tfidf_x,labels,show=False, normalization=True)

Performs k-means clustering on the tfidf_x vector, compares it with the labels

Show is a boolean, True prints verbose version of the function

Normalization is a boolean, True normalizes the tf-idf vector for quicker performance (false skips normalization, processes slower but garners higher metric scores)
