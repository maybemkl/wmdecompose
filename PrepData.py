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
import re
import seaborn as sns
import umap
import umap.plot

t = time.process_time()

PATH = "../data/yelp_dataset/"
yelp_data = []
r_dtypes = {"review_id":str,
            "user_id":str,
            "business_id":str,
            "stars": np.int32, 
            "date":str,
            "text":str,
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32}
drop = ['review_id', 'user_id', 'useful', 'funny', 'cool']
query = "date >= '2017-12-01' and (stars==1 or stars ==5)"

with open(f"{PATH}yelp_academic_dataset_review.json", "r") as f:
    reader = pd.read_json(f, orient="records", lines=True, dtype=r_dtypes, chunksize=1000)
    for chunk in reader:
        reduced_chunk = chunk.drop(columns=drop).query(query)
        yelp_data.append(reduced_chunk)
    
yelp_data = pd.concat(yelp_data, ignore_index=True)

stopword_list=stopwords.words('english')

elapsed = time.process_time() - t
print(f"Review data loaded. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")
print(f"Review data shape: {yelp_data.shape}")

print("Loading business data.")
yelp_business = pd.read_json(f"{PATH}yelp_academic_dataset_business.json", 
                             orient="records", 
                             lines=True)
yelp_business = yelp_business[yelp_business.city.isin(["Portland", "Atlanta"])]
print(f"Businesses in Atl and Ptl shape:{yelp_business.shape}")

yelp_merged = yelp_data.merge(yelp_business, on='business_id')

print(f"Merged data shape: {yelp_merged.shape}")

s_size = 250
rs = 42
print(f"Sampling data with sample size {s_size} and random seed {rs}.")
sample = yelp_merged.groupby(["city", "stars"]).sample(n=s_size, random_state=rs).reset_index()
print(f"Merged data shape: {sample.shape}")

tokenizer = ToktokTokenizer()

print("Removing stopwords.")
sample['review_clean']=[remove_stopwords(r, stopword_list, tokenizer) for r in sample['text']]
elapsed = time.process_time() - t
print(f"Stopwords removed. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

print("Cleaning sentences.")
sample['review_clean']=sample['review_clean'].apply(denoise_text)
sample['review_clean']=sample['review_clean'].apply(remove_special_characters)
sample['review_clean']=sample['review_clean'].apply(simple_lemmatizer)
elapsed = time.process_time() - t
print(f"Sentences cleaned. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

print("Removing stopwords again.")
sample['review_clean']=[remove_stopwords(r, stopword_list, tokenizer) for r in sample['review_clean']]
elapsed = time.process_time() - t
print(f"Stopwords removed again. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

PHRASING = True
MIN = 500
THRESHOLD = 200

if PHRASING:
    print("Starting phrasing.")
    sample['review_clean']= get_phrases([tokenizer.tokenize(i) for i in sample['review_clean']],
                                        min_count = MIN, 
                                        threshold = THRESHOLD)
    elapsed = time.process_time() - t
    print(f"Phrasing done. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

print("Example sentence:")
print(sentences_clean[0])