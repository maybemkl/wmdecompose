from wmdecompose.documents import Document
from wmdecompose.gale_shapeley import Matcher
from wmdecompose.models import LC_RWMD, WMD, WMDPairs
from wmdecompose.utils import *

from collections import defaultdict
from datetime import datetime
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from random import shuffle
from scipy.spatial.distance import is_valid_dm, cdist
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn import cluster

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import seaborn as sns
import sys
import umap

from pylab import savefig

random.seed(42)

vecs = sys.argv[1]
pairing = sys.argv[2]
reduced = sys.argv[3]
timestamp = f'wmdecomp_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
outpath = f'experiments/{timestamp}_{vecs}_{pairing}_reduced-{reduced}_/'

os.mkdir(outpath)

print(f"Beginning WMD pipeline with {vecs} vectors and {pairing} pairing.")
print(f"Vector reduction: {reduced}")

PATH = "data/"
print("Loading and preparing data.")
sample = pd.read_pickle(f"{PATH}yelp_sample.pkl")
tokenizer = ToktokTokenizer()

pos = sample[sample.sentiment == "positive"].reset_index(drop=True)
neg = sample[sample.sentiment == "negative"].reset_index(drop=True)

pos = pos.review_clean.tolist()
neg = neg.review_clean.tolist()

pos_tok = list(map(lambda x: tokenize(x, tokenizer), pos))
neg_tok = list(map(lambda x: tokenize(x, tokenizer), neg))

pos_sample = [" ".join(doc) for doc in pos_tok]
neg_sample = [" ".join(doc) for doc in neg_tok]

print(f"Positive samples: {len(pos_sample)}")
print(f"Negative samples: {len(neg_sample)}")

finetuned = True

if not finetuned:
    print("Loading GoogleNews Vectors")
    model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
else:
    print("Loading GoogleNews Vectors finetuned using Yelp review data.")
    model = KeyedVectors.load_word2vec_format('embeddings/yelp_w2v.txt', binary=False)

print("Preparing vectorizer and getting oov words.")
corpus = pos_sample + neg_sample
vectorizer = TfidfVectorizer(use_idf=False, tokenizer=tfidf_tokenize, norm='l1')
vectorizer.fit(corpus)

oov = [word for word in vectorizer.get_feature_names() if word not in model.key_to_index.keys()]

print(f"There are {len(oov)} oov words.")
print(f"Example oov words:")
print(oov[:50])

print("Tokenizing samples.")
pos_sample = list(map(lambda x: remove_oov(x, tokenizer, oov), pos_sample))
neg_sample = list(map(lambda x: remove_oov(x, tokenizer, oov), neg_sample))

print("Example of tokenized positive sample:")
print(pos_sample[5])

print("Vectorizing corpus.")
corpus = pos_sample + neg_sample
vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tfidf_tokenize,norm='l1')
vectorizer.fit(corpus)

pos_nbow = vectorizer.transform(pos_sample)
neg_nbow = vectorizer.transform(neg_sample)

pos_tok = list(map(lambda x: tokenize(x, tokenizer), pos_sample))
neg_tok =list(map(lambda x: tokenize(x, tokenizer), neg_sample))

print("Example of vectorized sample:")
print(pos_tok[5][:20])

print("Removing oov.")
oov_ = [word for word in vectorizer.get_feature_names() if word not in model.key_to_index.keys()]
print(f"There are {len(oov_)} oov words left.")

features = vectorizer.get_feature_names()
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
idx2word = {idx: word for idx, word in enumerate(vectorizer.get_feature_names())}
E = model[features]

pos_docs, neg_docs = [], []

for idx, doc in enumerate(pos_tok):
    pos_docs.append(Document(doc, pos_nbow[idx], word2idx, E))
    
for idx, doc in enumerate(neg_tok):
    neg_docs.append(Document(doc, neg_nbow[idx], word2idx, E))

## Create Kmeans for WMD
k = 100

if vecs == 'tsne':
    print("Getting T-SNE vectors.")
    method='barnes_hut'
    n_components = 2
    verbose = 1
    E_tsne = TSNE(n_components=n_components, method=method, verbose=verbose).fit_transform(E)
    plt.scatter(E_tsne[:, 0], E_tsne[:, 1], s=1);
    plt.savefig(f'{outpath}tsne_yelp.png')
    if reduced == True:
        E = E_tsne
    
if vecs == 'umap':
    print("Getting distance matrix and determining UMAP hyperparameters.")
    metric = 'cosine'
    dm = cdist(E, E, metric)
    np.fill_diagonal(dm, 0)
    print("Checking for validity of distance matrix.")
    print(f"Is valid dm: {is_valid_dm(dm)}")
    mean, std = np.mean(dm), np.std(dm)
    print(mean, std)
    min_dist=mean - 2*std
    n_neighbors = int(0.001*len(E))
    n_components=2
    print(f"Min distance: {min_dist}")
    print(f"N. neighbors: {n_neighbors}")
    print(f"N. compontents: {n_components}")
    print("Getting UMAP vectors.")
    verbose = 1
    E_umap = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        verbose=verbose
    ).fit_transform(E)
    plt.scatter(E_umap[:, 0], E_umap[:, 1], s=1);
    plt.savefig(f'{outpath}umap_yelp.png')
    if reduced == True:
        E = E_umap

km = cluster.KMeans(n_clusters=k,max_iter=300).fit(E)
labels = km.labels_      
word2cluster = {features[idx]: cl for idx, cl in enumerate(labels)}
cluster2words = defaultdict(list)
for key, value in word2cluster.items():
    cluster2words[value].append(key)
    
pairs = get_pairs(pairing, pos_docs, neg_docs)

print(f"Prepared {len(pairs)} pairs.")
print("Initializing WMD.")
wmd_pairs_flow = WMDPairs(pos_docs,neg_docs,pairs,E,idx2word)

print("Getting WMD distances.")
wmd_pairs_flow.get_distances(decompose = True, 
                             sum_clusters = True, 
                             w2c = word2cluster, 
                             c2w = cluster2words,
                             thread = False,
                             relax = True)

print("Getting differences in flow.")
wmd_pairs_flow.get_differences()

print("Saving model.")
with open(f'{outpath}WMDmodel.pkl', 'wb') as handle:
    pickle.dump(wmd_pairs_flow, handle, protocol=pickle.HIGHEST_PROTOCOL)

top_n = 100
top_words_source_df = get_top_words(wmd_pairs_flow, top_n, True) 
top_words_sink_df = get_top_words(wmd_pairs_flow, top_n, False) 

print(f"Saving top {top_n} words both ways.")
top_words_source_df.to_csv(f"{outpath}source_to_sink.csv", index=False)
with open(f'{outpath}pos_to_neg_diff.pkl', 'wb') as handle:
    pickle.dump(source_to_sink, handle, protocol=pickle.HIGHEST_PROTOCOL)

top_words_sink_df.to_csv(f"{outpath}sink_to_source.csv", index=False)
with open(f'{outpath}neg_to_pos_diff.pkl', 'wb') as handle:
    pickle.dump(sink_to_source, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
n_clusters = 100
n_words = 100

print(f"Getting {n_clusters} with {n_words} each.")
c1 = output_clusters(wd=wmd_pairs_flow.wd_source_diff.items(), 
                     cd=wmd_pairs_flow.cd_source.items(), 
                     c2w=cluster2words, 
                     n_clusters=n_clusters, 
                     n_words=n_words)
c2 = output_clusters(wd=wmd_pairs_flow.wd_sink_diff.items(), 
                     cd=wmd_pairs_flow.cd_sink.items(), 
                     c2w=cluster2words, 
                     n_clusters=n_clusters, 
                     n_words=n_words)

print("Saving clusters.")
c1.to_csv(f'{outpath}pos_to_neg_clusters.csv', index=False)
with open(f'{outpath}pos_to_neg_clusters.pkl', 'wb') as handle:
    pickle.dump(c1, handle, protocol=pickle.HIGHEST_PROTOCOL)

c2.to_csv(f'{outpath}neg_to_pos_clusters.csv', index=False)
with open(f'{outpath}neg_to_pos_clusters.pkl', 'wb') as handle:
    pickle.dump(c2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
if pairing != 'full':
    print("Preparing and saving boxplots.")
    plot_box(wmd_pairs_flow, sample, c1, 500,1000, "city", "distance", True, f'{outpath}pos_to_neg_boxplots.png', True, False)
    plot_box(wmd_pairs_flow, sample, c2, 500,1000, "city", "distance", False, f'{outpath}pos_to_neg_boxplots.png', True, False)
