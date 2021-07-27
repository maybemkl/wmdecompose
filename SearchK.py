from collections import defaultdict
from flow_wmd.utils import *

from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from scipy.spatial.distance import is_valid_dm, cdist
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn import cluster

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import umap

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
E = model[features]

K = range(10,210, 10)

## W2V
search_w2v = True
if search_w2v:
    print("Searching K for full W2V vectors.")
    w2v_ssd, w2v_silhouette = kmeans_search(E, K)
    plot_kmeans(K,w2v_ssd,"elbow","img/w2v_ssd")
    plot_kmeans(K,w2v_silhouette,"img/w2v_silhouette",)

## T-SNE
search_tsne = True
if search_tsne:
    print("Getting T-SNE vectors.")
    method='barnes_hut'
    n_components = 2
    verbose = 1
    E_tsne = TSNE(n_components=n_components, method=method, verbose=verbose).fit_transform(E)
    plt.scatter(E_tsne[:, 0], E_tsne[:, 1], s=1);
    plt.savefig('img/tsne_yelp.png')
    print("Searching K for T-SNE vectors.")
    tsne_ssd, tsne_silhouette = kmeans_search(E_tsne, K)
    plot_kmeans(K,tsne_ssd,"elbow","img/tsne_ssd")
    plot_kmeans(K,tsne_silhouette,"silhouette","img/tsne_silhouette")
    
## PCA
search_pca = True
if search_pca:
    print("Getting PCA vectors.")
    n_components = 0.9
    verbose = 1
    pca_fit = PCA(n_components = n_components).fit(E)
    print(len(pca_fit.explained_variance_ratio_))
    print(pca_fit.explained_variance_ratio_)
    print(np.sum(pca_fit.explained_variance_ratio_))
    E_pca = pca_fit.transform(E)
    plt.scatter(E_pca[:, 0], E_pca[:, 1], s=1);
    plt.savefig('img/pca_yelp.png')
    print("Searching K for PCA vectors.")
    pca_ssd, pca_silhouette = kmeans_search(E_pca, K)
    plot_kmeans(K,pca_ssd,"elbow","img/tsne_ssd")
    plot_kmeans(K,pca_silhouette,"silhouette","img/tsne_silhouette")
    
## UMAP
search_umap = True
if search_umap:
    print("Determining UMAP hyperparameters.")
    metric = 'cosine'
    dm = cdist(E, E, metric)
    np.fill_diagonal(dm, 0)
    is_valid_dm(dm)
    mean, std = np.mean(dm), np.std(dm)
    print(mean, std)
    min_dist=mean - 5*std
    n_neighbors = int(0.001*len(E))
    n_components=2
    print(f"Min distance: {min_dist}")
    print(f"N. neighbors: {n_neighbors}")
    print(f"N. compontents: {n_components}")
    print("Getting UMAP vectors.")
    E_umap = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        verbose=verbose
    ).fit_transform(E)
    plt.scatter(E_umap[:, 0], E_umap[:, 1], s=1);
    plt.savefig('img/umap_yelp.png')
    print("Searching K for UMAP vectors.")
    umap_ssd, umap_silhouette = kmeans_search(E_umap, K)
    plot_kmeans(K,umap_ssd,"elbow","img/umap_ssd")
    plot_kmeans(K,umap_silhouette,"silhouette","img/umap_silhouette")


