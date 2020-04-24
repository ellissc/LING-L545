import os
import conllu #https://pypi.org/project/conllu/
from io import open
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import numpy as np
from sklearn.metrics import silhouette_score
kongzi = []
liuxiang = []
dongzhongshu = []
mengzi = []
zhuangzi = []
zhuangzi_test = []


punct = ["：","，","？","」","「","。","『","』","；","！","《","》","、"]
#export PATH=$PATH:/mnt/c/Users/Ellis/Desktop/L445/LING-L545/final_project/udpipe/src

def txt2conllu(direc, conllu_direc):
    #creates conllu file using clzh.model, draws txt files from direc and writes conllu files to conllu_direc
    dir = direc
    for root, dirs, files in os.walk(dir):
        for file in files:
            if ".txt" in file:
                txtfile = os.path.join(root, file)
                conllufile = file.replace(".txt",".conllu")
                os.system("udpipe --tokenize --parse clzh.model "+txtfile+ " > "+ dir+"conllu_files/"+conllufile)

def ngramify(array):
    #Takes an array of tokenized text and creates ngrams (3<=n<=8)
    ngrams = []
    inner_array = []
    for internal in array:
        for i in internal:
            inner_array.append(i)
    for n in range(3,9):
        for i in range(len(inner_array)-n+1):
            ngram_array = inner_array[i:i+n]
            strings = ""
            for i in ngram_array:
                strings = strings+i
            ngrams.append(strings)
    return ngrams

def conllu2arr(direc):
    #Draws conllu files from direc and sorts them into the respective arrays
    dir = direc
    for root, dirs, files in os.walk(dir):
        for file in files:
            if ".conllu" in file:
                doc = []
                conllufile = file
                #print(conllufile)
                filepath = os.path.join(root, conllufile)
                datafile = open(filepath,"r",encoding="utf-8")
                for tokenlist in conllu.parse_incr(datafile):
                    for token in tokenlist:
                        if token["form"] not in punct:
                            doc.append(token["form"])
                #print(doc)
            if "kongzi" in file:
                kongzi.append(doc)
            elif "mengzi" in file:
                mengzi.append(doc)
            elif "liuxiang" in file:
                liuxiang.append(doc)
            elif "dongzhongshu" in file:
                dongzhongshu.append(doc)
            elif "zhuangzi" in file and "outer" not in file:
                zhuangzi.append(doc)
            elif "outer" in file:
                zhuangzi_test.append(doc)

def overlap(arr1, arr2):
    #calculates shared overlap of 2 given arrays
    compiled_arr1 = []
    compiled_arr2 = []
    #compresses down, removes duplicates
    for internal in arr1:
        for i in internal:
            if i not in compiled_arr1:
                compiled_arr1.append(i)
    #print(len(compiled_arr1))
    for internal in arr2:
        for i in internal:
            if i not in compiled_arr2:
                compiled_arr2.append(i)
    #print(len(compiled_arr2))
    #overlap_arr = [value for value in compiled_arr1 if value in compiled_arr2]
    overlap_arr = set(compiled_arr1) & set(compiled_arr2)
    overlap_num = len(overlap_arr)
    #print(overlap_num)
    return float(overlap_num/(overlap_num+(len(compiled_arr1)-overlap_num)+(len(compiled_arr2)-overlap_num))) * 100

def call_overlap(kongzi, kongzi_ngrams, mengzi,mengzi_ngrams, dongzhongshu,dongzhongshu_ngrams, liuxiang,liuxiang_ngrams, zhuangzi,zhuangzi_ngrams, zhuangzi_test,zhuangzi_test_ngrams):
    #executes percent overlap for the authors and a test file
    #kongzi and liuxiang
    print("Kongzi and Liuxiang percent overlap: ")
    print(overlap(kongzi,liuxiang))
    print(overlap(kongzi_ngrams,liuxiang_ngrams))
    print()

    #kongzi and dongzhongshu
    print("Kongzi and Dongzhongshu percent overlap: ")
    print(overlap(kongzi,dongzhongshu))
    print(overlap(kongzi_ngrams,dongzhongshu_ngrams))
    print()

    #kongzi and mengzi
    print("Kongzi and Mengzi percent overlap: ")
    print(overlap(kongzi,mengzi))
    print(overlap(kongzi_ngrams,mengzi_ngrams))
    print()

    #liuxiang and dongzhongshu - liuxiang problem?
    print("Liuxiang and Dongzhongshu percent overlap: ")
    print(overlap(liuxiang,dongzhongshu))
    print(overlap(liuxiang_ngrams,dongzhongshu))
    print()

    #liuxiang and mengzi - liuxiang problem?
    print("Liuxiang and Mengzi percent overlap: ")
    print(overlap(liuxiang,mengzi))
    print(overlap(liuxiang_ngrams,mengzi_ngrams))
    print()

    #dongzhongshu and mengzi
    print("Dongzhongshu and Mengzi percent overlap: ")
    print(overlap(dongzhongshu,mengzi))
    print(overlap(dongzhongshu_ngrams,mengzi_ngrams))
    print()

    print("Zhuangzi (inner) and Zhuangzi (outer) percent overlap: ")
    print(overlap(zhuangzi,zhuangzi_test))
    zhuang = overlap(zhuangzi_ngrams,zhuangzi_test_ngrams)
    print(zhuang)
    print()

def conllu2tfidf(direc):
    dir = direc
    corpus = []
    authors = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if ".conllu" in file:
                conllufile = file
                filepath = os.path.join(root,conllufile)
                datafile = open(filepath, "r", encoding = "utf-8")
                sentence = ""
                for tokenlist in conllu.parse_incr(datafile):
                    for token in tokenlist:
                        if token["form"] not in punct:
                            sentence = sentence+token["form"]
                corpus.append(sentence)
            if "kongzi" in file:
                authors.append(0)
            elif "mengzi" in file:
                authors.append(4)
            elif "liuxiang" in file:
                authors.append(1)
            elif "dongzhongshu" in file:
                authors.append(2)
            elif "zhuangzi" in file:
                authors.append(3)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2,10))
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)).toarray() #calculates tfidf
    #pickle.dump(tfidf, open("tfidf.pickle","wb"))
    #with open("pickle_companion.csv","w",newline="") as f:
    #    writer = csv.writer(f)
    #    writer.writerow(authors)
    print(tfidf.shape)
    return tfidf, authors


def k_means(tfidf_x,labels, show=False):
    true_k = 5
    if show:
        print("n_samples: %d, n_features: %d" % tfidf_x.shape)
        print()
    svd = TruncatedSVD()
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tfidf_x)
    km = KMeans(n_clusters=true_k, init="k-means++", n_init = 15)
    if show:
        print("clustering data with %s" % km)
    km.fit(X)
    output_labels = km.labels_.tolist()
    if show:
        print()
        print(labels)
        print(output_labels)
        print()
        homogeneity = metrics.homogeneity_score(labels, km.labels_)
        print("Homogeneity: %0.3f" % homogeneity)
        completeness = metrics.completeness_score(labels, km.labels_)
        print("Completeness: %0.3f" % completeness)
        vmeasure = metrics.v_measure_score(labels, km.labels_)
        print("V-measure: %0.3f" % vmeasure)
        adjustedrand = metrics.adjusted_rand_score(labels, km.labels_)
        print("Adjusted Rand-Index: %.3f" % adjustedrand )
        silhou = metrics.silhouette_score(X, km.labels_, sample_size=1000)
        print("Silhouette Coefficient: %0.3f"% silhou)
        print()
        print()

    #F1 score:
    truth = labels
    predictions = output_labels
    weighted = metrics.f1_score(truth, predictions,average="weighted")
    if show:
        print("weighted f1 score")
        print(weighted)
        print("micro f1 score")
        print(metrics.f1_score(truth, predictions,average="micro"))
        print("macro f1 score")
        print(metrics.f1_score(truth, predictions,average="macro"))
    return weighted

def number_clusters(X, range_n_clusters):
    avg_sil_score = []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        avg_sil_score.append(silhouette_avg)
    max_sil_score = np.max(avg_sil_score)
    n_cls = avg_sil_score.index(max_sil_score)
    return range_n_clusters[n_cls]


def k_means_silhoutte(tfidf_x,labels, show=True):
    start = int(len(tfidf_x)/2)
    stop = len(tfidf_x)-5
    step = 5
    cls_range = range(start,stop,step)
    num_cls = number_clusters(tfidf_x, cls_range)
    #true_k = 5
    if show:
        print("n_samples: %d, n_features: %d" % tfidf_x.shape)
        print()
    svd = TruncatedSVD()
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tfidf_x)

    km = KMeans(n_clusters=num_cls, init="random",n_init = 15)
    if show:
        print("clustering data with %s" % km)
    km.fit(X)
    output_labels = km.labels_.tolist()
    if show:
        print()
        print(labels)
        print(output_labels)
        print()
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
        print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, km.labels_, sample_size=1000))
        print()

    #F1 score:
    truth = labels
    predictions = output_labels
    weighted = metrics.f1_score(truth, predictions,average="weighted")
    if show:
        print("weighted f1 score")
        print(weighted)
        print("micro f1 score")
        print(metrics.f1_score(truth, predictions,average="micro"))
        print("macro f1 score")
        print(metrics.f1_score(truth, predictions,average="macro"))
    return weighted

#file directory:
dir = "/mnt/c/Users/Ellis/Desktop/L445/LING-L545/final_project/ctexts/"
#example n-gram directory:
dir_gram = "/mnt/c/Users/Ellis/Desktop/L445/LING-L545/final_project/ctexts/3_gram/"
dir_conllu = "/mnt/c/Users/Ellis/Desktop/L445/LING-L545/final_project/ctexts/conllu_files/"

#Convert text files into conllu files using a trained udpipe model
#dir should contain the texts, and clzh.model should be whatever the title of the udpipe model is, in the same file as this script

#txt2conllu(dir, dir_conllu)
#conllu2arr(dir_conllu)

kongzi_ngrams = ngramify(kongzi)
liuxiang_ngrams = ngramify(liuxiang)
dongzhongshu_ngrams = ngramify(dongzhongshu)
mengzi_ngrams = ngramify(mengzi)
zhuangzi_ngrams = ngramify(zhuangzi)
zhuangzi_test_ngrams = ngramify(zhuangzi_test)

#percent overlap section:
#call_overlap(kongzi, kongzi_ngrams, mengzi,mengzi_ngrams, dongzhongshu,dongzhongshu_ngrams, liuxiang,liuxiang_ngrams, zhuangzi,zhuangzi_ngrams, zhuangzi_test,zhuangzi_test_ngrams)


#tf-idf part
#conllu2tfidf(dir_conllu)

#kmeans clusering part
show = True
#k_means(dir_conllu,show)
tfidf_x, labels = conllu2tfidf(dir_conllu)
#k_means(tfidf_x,labels,show) # best f1 score so far is 0.64
#k_means_silhoutte(tfidf_x, labels)
print()

total = 75
f1_outputs = []
homogeneity = []
completeness = []
vmeasure = []
adjustedrand = []
silhou = []
for i in range(total):
    f1_outputs.append(k_means(tfidf_x,labels))
average = np.mean(f1_outputs)
maxx = max(f1_outputs)
minn = min(f1_outputs)
print("F1 score average value: %.3f" %average)
print("Max F1 score %.3f" %maxx)
print("Min F1 score %.3f" %minn)
#Mean Average Precision
