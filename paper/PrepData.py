from wmdecompose.utils import *
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer

import numpy as np
import pandas as pd
import re
import time

t = time.process_time()

PATH = "data/yelp_dataset/"
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

print("Loading review data.")
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
yelp_merged = yelp_merged.rename(columns={"stars_x":"stars"})

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
                                        threshold = THRESHOLD,
                                        save=False,
                                        load=True)
    elapsed = time.process_time() - t
    print(f"Phrasing done. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

print("Example sentence before preprocessing:")
print(sample['text'][6])

print("Example sentence after preprocessing:")
print(" ".join(sample['review_clean'][6]))

sample["sentiment"] = ['positive' if s == 5 else 'negative' for s in sample['stars']]
sample = sample.sort_values("sentiment").reset_index()

print("Saving data.")
sample.to_pickle('data/yelp_sample_categories.pkl')