from bs4 import BeautifulSoup
from collections import Counter
from gensim.models.phrases import Phrases, Phraser
from itertools import islice
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Callable, DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def kmeans_search(E:np.array, K:List[int]) -> Tuple[List[float], List[float]]:
    """Grid search for Kmeans models.
    
    Args:
      E: An array with an embedding matrix of float values.
      K: A list of integers for all the K values that should be searched.
    
    Return:
      sum_of_squared_distances: A list of float values with the 'intertia_' variable from Kmeans. 
      silhouette: A list of float values with the silhouette scores for the Kmeans models at each K.
    
    """
    
    sum_of_squared_distances = []
    silhouette = []
    for k in K:
        km = KMeans(n_clusters=k,max_iter=300)
        km = km.fit(E)
        sum_of_squared_distances.append(km.inertia_)
        cluster_labels = km.fit_predict(E)
        silhouette_avg = silhouette_score(E, cluster_labels)
        silhouette.append(silhouette_avg)
        if k % 5 == 0:
            print("For n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg)
    return sum_of_squared_distances, silhouette

def plot_kmeans(K:List[int], data:List[float], metric:str, fname:str = "") -> None:
    """Plot silhouette or elbow scores for Kmeans model.
    
    Args:
      K: A list of integers for all the K values that should be searched.
      data: A list with a float (or int) number for each Kmeans model.
      metric: A string with the metric to plot. Must be be 'elbow' or 'silhouette'.
      fname: String with filename for saving figure. Optional.  
    
    """
    
    plt.plot(K, data, 'bx-')
    plt.xlabel('k')
    if metric == "elbow":
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
    if metric == "silhouette":
        plt.ylabel('Silhouette score')
        plt.title('Silhouette Score for Optimal k')
    elif metric not in ["elbow", "silhouette"]:
        print("Please define 'metric' as either 'elbow' or 'silhouette'.")
    if len(fname) > 0:
        plt.savefig(fname)
    else:
        plt.show()

def get_phrases(sentences:List[List[str]], 
                min_count:int=5, 
                threshold:int=100, 
                save:bool=False,
                load:bool=False,
                PATH:str="embeddings/") -> List[List[str]]:
    """Function for generating, saving and loading Phrases using Gensim.
    For details, see https://radimrehurek.com/gensim/models/phrases.html
    
    Args:
      sentences: A list of strings.
      min_count: An integer for the 'min_count' argument in the Gensim Phraser.
      threshold:  An integer for the 'threshold' argument in the Gensim Phraser.
      save: Boolean indicating whether phrases should be saved when generating new phrases.
      load: Boolean indicating that whether phrases should be loaded, instead of saved.
      PATH: String for path to which save or from which to load phrases.
    
    Return:
      phrased_tri: List of lists with the phrased versions of the input sentences.
    
    """
    
    if load:
        bigram=Phrases.load(f"{PATH}bigram_phrases.pkl")
        trigram=Phrases.load(f"{PATH}trigram_phrases.pkl")
        
    else:
        print("Initializing bigram Phrases.")
        bigram = Phrases(sentences, min_count=min_count, threshold = threshold) # higher threshold fewer phrases.
        print("Initializing trigram Phrases.")
        trigram = Phrases(bigram[sentences]) 
    
    if save:
        print("Saving bigram Phrases.")
        bigram.save(f"{PATH}bigram_phrases.pkl")
        print("Saving trigram Phrases.")
        trigram.save(f"{PATH}trigram_phrases.pkl")

    print("Finding bigrams in data.")
    phrased_bi = [b for b in bigram[sentences]]
    print("Finding trigrams in data.")
    phrased_tri = [t for t in trigram[[b for b in bigram[sentences]]]]
    return phrased_tri

def output_clusters(wd:List[Tuple[str, float]], 
                    cd:List[Tuple[int, float]], 
                    c2w:DefaultDict[list, Dict[int, List[str]]], 
                    n_clusters:int = 10, 
                    n_words:int = 10) -> pd.DataFrame:
    """Get clusters with highest accumulated distance and with within cluster words organized by distance contribution.   
    
    Args:
      wd: List of tuples with words and their accumulated distance contributions in each tuple.
      cd: List of tuples with clusters and their accumulated distance contributions in each tuple.
      c2w: Default dictionary with the cluster number as key and the list of the words in said cluster as value.
      n_clusters: Integer with the number of clusters.
      n_words: Integer with the number of words per cluster.
    
    Return:
      keywords_df: Pandas dataframe with clusters with distances as column headers and words with distances as row values.
    
    """

    top_clusters = [k for k, v in sorted(cd, key=lambda item: item[1], reverse=True)[:n_clusters]]
    word_rankings = {k: v for k, v in sorted(wd, key=lambda item: item[1], reverse=True)}
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

def remove_oov(text:str, tokenizer:Callable[[List[str]], List[str]], oov:List[str]) -> str:
    """Function for removing out-of-vocabulary (oov) words.
        
    Args:
      text: String to be analyzed for oov words.
      tokenizer: Any tokenizer that returns input sentence as a list of strings.
      oov: List of oov words.
    
    Return:
      filtered_text: String with oov words removed.
    
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in oov]
    #filtered_tokens = filter(lambda token: token not in oov, tokens)
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def take(n:int, iterable:iter) -> list:
    """Return first n items of the iterable as a list.
        
    Args:
      n: An integer of the number of items to keep from iterable.
      iterable: Any iterable.
    
    Return:
      list: List of items contained in input iterable.
    """
    return list(islice(iterable, n))

def tokenize(text:str, tokenizer:Callable[[str], List[str]]) -> List[str]:
    """Callable to use with the Sklearn TfIdfVectorizer.
 
    Args:
      text: String to tokenize.
      tokenizer: Any callable that takes a string as input and returns a list of strings.
    
    Return:
      tokens: List of strings.
      
    """
    tokens = tokenizer.tokenize(text)
    return tokens

def tfidf_tokenize(text:str) -> List[str]:
    """Callable to use with the Sklearn TfIdfVectorizer with the tokenizer predetermined as the nltk ToktokTokenizer.
        
    Args:
      text: String to tokenize.
    
    Return:
      tokens: List of strings.
    
    """
    tokenizer=ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

# Custom preprocessing functions
# Partly self-authored, partly from https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

def strip_html(text:str) -> str:
    """Removing the html strips.
        
    Args:
      text: String to have HTML removed.
    
    Return:
      text: String with HTML removed.
    
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text:str) -> str:
    """Removing the square <brackets
        
    Args:
      text: String to have square brackets removed.
    
    Return:
      text: String with square brackets removed.
    """
    
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text:str) -> str:
    """Removing the noisy text
        
    Args:
      text: String to denoise for HTML.
    
    Return:
      text: String with HTML denoised.
      
    """
    
    text = re.sub('<br / ><br / >', ' ', text)
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text:str) -> str:
    """Define function for removing special characters
        
    Args:
      text: String to filter for special characters.
    
    Return:
      text: String with special characters removed.
    
    """

    pattern=r'[^a-zA-z\s]'
    text=re.sub(pattern,'',text)
    return text

def simple_lemmatizer(text:str) -> str:
    """Lemmatizing the text.
        
    Args:
      text: String to lemmatize.
    
    Return:
      text: String that has been lemmatized.
    
    """
    
    lemmatizer=WordNetLemmatizer() 
    text= ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def remove_stopwords(text:str, 
                     stopword_list:List[str], 
                     tokenizer:Callable[[str], List[str]], 
                     is_lower_case:bool=False) -> str:
    """Removing the stopwords.
            
    Args:
      text: String to filter for stopwords.
      stopword_list: List of strings with stopwords.
      tokenizer: Any callable that takes a string as input and returns a list of strings.
      is_lower_case: Boolean indicating whether input is alread lower case.
    
    Return:
      filtered_text: String with stopwords removed.
    
    """
    
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text