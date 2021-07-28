from collections import defaultdict
from flow_wmd.documents import Document
from flow_wmd.gale_shapeley import Matcher
from flow_wmd.models import LC_RWMD, WMD, WMDManyToMany, WMDPairs
from flow_wmd.utils import *

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

if vecs == 'w2v':
    km = cluster.KMeans(n_clusters=k,max_iter=300).fit(E)
    labels = km.labels_

if vecs == 'tsne':
    print("Getting T-SNE vectors.")
    method='barnes_hut'
    n_components = 2
    verbose = 1
    E_tsne = TSNE(n_components=n_components, method=method, verbose=verbose).fit_transform(E)
    plt.scatter(E_tsne[:, 0], E_tsne[:, 1], s=1);
    plt.savefig('img/tsne_yelp.png')
    if reduced:
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
    plt.savefig('img/umap_yelp.png')
    if reduced:
        E = E_umap
        
word2cluster = {features[idx]: cl for idx, cl in enumerate(labels)}
cluster2words = defaultdict(list)
for key, value in word2cluster.items():
    cluster2words[value].append(key)
    
if pairing == 'gs':
    lc_rwmd = LC_RWMD(pos_docs, neg_docs,pos_nbow,neg_nbow,E)
    lc_rwmd.get_D()
    print("Running Gale-Shapeley pairing.")
    matcher = Matcher(lc_rwmd.D)
    engaged = matcher.matchmaker()
    matcher.check()
    pairs = engaged
if pairing == 'random':
    print("Running random pairing.")
    pos_idx = list(range(0,len(pos_docs)))
    neg_idx = list(range(0,len(neg_docs)))
    shuffle(pos_idx)
    shuffle(neg_idx)
    pairs = dict(zip(pos_idx, neg_idx))
if pairing == 'full':
    print("Running full pairing.")
    pos_idx = list(range(0,len(pos_docs)))
    neg_idx = list(range(0,len(neg_docs)))
    pairs = [(i,j) for i in pos_idx for j in neg_idx]

print("Initializing WMD.")
wmd_pairs_flow = WMDPairs(pos_docs,neg_docs,pairs,E,idx2word)

print("Getting WMD distances.")
wmd_pairs_flow.get_distances(return_flow = True, 
                             sum_clusters = True, 
                             w2c = word2cluster, 
                             c2w = cluster2words,
                             thread = True)

print("Getting differences in flow.")
wmd_pairs_flow.get_differences()

print("Saving model.")
with open(f'experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_WMDmodel.pkl', 'wb') as handle:
    pickle.dump(wmd_pairs_flow, handle, protocol=pickle.HIGHEST_PROTOCOL)

top_n = 100
print(f"Getting top {top_n} words both ways.")
x1_to_x2 = {k: v for k, v in sorted(wmd_pairs_flow.wc_X1_diff.items(), key=lambda item: item[1], reverse=True)[:top_n]}
top_words_x1_df = pd.DataFrame.from_dict(x1_to_x2, orient='index', columns = ["cost"])
top_words_x1_df['word'] = top_words_x1_df.index

x2_to_x1 = {k: v for k, v in sorted(wmd_pairs_flow.wc_X2_diff.items(), key=lambda item: item[1], reverse=True)[:top_n]}
top_words_x2_df = pd.DataFrame.from_dict(x2_to_x1, orient='index', columns = ["cost"])
top_words_x2_df['word'] = top_words_x2_df.index

print(f"Saving top {top_n} words both ways.")
top_words_x1_df.to_csv(f"experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_x1_to_x2.csv", index=False)
with open(f'experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_pos_to_neg_diff.pkl', 'wb') as handle:
    pickle.dump(x1_to_x2, handle, protocol=pickle.HIGHEST_PROTOCOL)

top_words_x2_df.to_csv(f"experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_x2_to_x1.csv", index=False)
with open(f'experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_neg_to_pos_diff.pkl', 'wb') as handle:
    pickle.dump(x2_to_x1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
n_clusters = 100
n_words = 100

print(f"Getting {n_clusters} with {n_words} each.")
c1 = output_clusters(wc=wmd_pairs_flow.wc_X1_diff.items(), 
                     cc=wmd_pairs_flow.cc_X1.items(), 
                     c2w=cluster2words, 
                     n_clusters=n_clusters, 
                     n_words=n_words)
c2 = output_clusters(wc=wmd_pairs_flow.wc_X2_diff.items(), 
                     cc=wmd_pairs_flow.cc_X2.items(), 
                     c2w=cluster2words, 
                     n_clusters=n_clusters, 
                     n_words=n_words)

print("Saving clusters.")
c1.to_csv(f'experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_pos_to_neg_clusters.csv', index=False)
with open(f'experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_pos_to_neg_clusters.pkl', 'wb') as handle:
    pickle.dump(c1, handle, protocol=pickle.HIGHEST_PROTOCOL)

c2.to_csv(f'experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_neg_to_pos_clusters.csv', index=False)
with open(f'experiments/{vecs}_{pairing}_reduced-{reduced}_yelp_neg_to_pos_clusters.pkl', 'wb') as handle:
    pickle.dump(c2, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Preparing and saving boxplots.")
x1_costs = pd.DataFrame(wmd_pairs_flow.X1_feat)
x1_costs.index = list(pairs.keys())
x1_costs = x1_costs.sort_index()
x1_costs = x1_costs[c1.columns]
x1_costs['city'] = sample[:500].city
x1_costs_long = pd.melt(x1_costs, id_vars=['city']).rename(columns={"variable":"cluster"})
x1_costs_long = x1_costs_long[x1_costs_long.value != 0]

g = sns.catplot(x="city", 
                y="value", 
                col="cluster", 
                data=x1_costs_long, 
                kind="box",
                height=5, 
                aspect=.7,
                col_wrap=5,
                margin_titles=True);
g.map_dataframe(sns.stripplot, 
                x="city", 
                y="value", 
                palette=["#404040"], 
                alpha=0.2, dodge=True)
g.set_axis_labels("City", "Cost")
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)

g.savefig(f'img/{vecs}_{pairing}_reduced-{reduced}_yelp_pos_to_neg_boxplots.png', dpi=400)

x2_costs = pd.DataFrame(wmd_pairs_flow.X1_feat)
x2_costs.index = list(pairs.values())
x2_costs = x2_costs.sort_index()
x2_costs = x2_costs[c2.columns]
x2_costs['city'] = sample[500:1000].city.tolist()

x2_costs_long = pd.melt(x2_costs, id_vars=['city']).rename(columns={"variable":"cluster"})
x2_costs_long = x2_costs_long[x2_costs_long.value != 0]

g = sns.catplot(x="city", 
                y="value", 
                col="cluster", 
                data=x2_costs_long, 
                kind="box",
                height=5, 
                aspect=.7,
                col_wrap=5,
                margin_titles=True);
g.map_dataframe(sns.stripplot, 
                x="city", 
                y="value", 
                palette=["#404040"], 
                alpha=0.2, dodge=True)
g.set_axis_labels("City", "Cost")
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)

g.savefig(f'img/{vecs}_{pairing}_reduced-{reduced}_yelp_neg_to_pos_boxplots.png', dpi=400)