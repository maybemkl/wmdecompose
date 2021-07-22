from bs4 import BeautifulSoup
from collections import Counter
from itertools import islice
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer

import pandas as pd
import re

def get_phrases(sentences:list, min_count:int=5, threshold:int=100) -> list:
    bigram = Phrases(sentences, min_count=min_count, threshold = threshold) # higher threshold fewer phrases.
    trigram = Phrases(bigram[sentences])  

    # 'Phraser' is a wrapper that makes 'Phrases' run faster
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)

    phrased_bi = [b for b in bigram[sentences]]
    phrased_tri = [t for t in trigram[[b for b in bigram[sentences]]]]
    
    return phrased_tri

def output_clusters(wc:list, cc:list, c2w:dict, n_clusters:int = 10, n_words:int = 10, average:bool=False,labels:list=[]) -> pd.DataFrame:
    #if average:
    #    lbl_counts = Counter(list(labels))
    #    cc = [lbl_counts]
    
    top_clusters = [k for k, v in sorted(cc, key=lambda item: item[1], reverse=True)[:n_clusters]]
    word_rankings = {k: v for k, v in sorted(wc, key=lambda item: item[1], reverse=True)}
    keywords = []
    for c in top_clusters:
        cluster_words = {w: word_rankings[w] for w in c2w[c] 
                         if w in word_rankings.keys()}
        top_c_words = [f"{k} ({round(v, 2)})" for k, v in sorted(cluster_words.items(), 
                                               key=lambda item: item[1], 
                                               reverse=True)[:n_words]]
        keywords.append(top_c_words)
    keywords_df = pd.DataFrame(keywords).transpose()
    keywords_df.columns = top_clusters
    return keywords_df

def read_1w_corpus(name, sep="\t"):
    for line in open(name):
        yield line.split(sep)

#removing the oov words
def remove_oov(text, tokenizer, oov):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in oov]
    #filtered_tokens = filter(lambda token: token not in oov, tokens)
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def take(n:int, iterable:iter) -> list:
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def tokenize(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return tokens

def tfidf_tokenize(text):
    tokenizer=ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

# Custom preprocessing functions
# Partly self-authored, partly from https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square <brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = re.sub('<br / ><br / >', ' ', text)
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z\s]'
    text=re.sub(pattern,'',text)
    return text

#Lemmatizing the text
def simple_lemmatizer(text):
    lemmatizer=WordNetLemmatizer() 
    text= ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

#removing the stopwords
def remove_stopwords(text, stopword_list, tokenizer, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text