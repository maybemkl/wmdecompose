import gensim
import numpy as np
import pandas as pd
import pickle
import time

from wmdecompose.utils import *
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch} starting.")

    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch} ended.")
        self.epoch += 1
        
class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch != 1:
            previous_loss = self.losses[self.epoch-2]
        else:
            previous_loss = 0
        self.losses.append(loss)
        difference = loss-previous_loss
        print(f'  Loss: {loss}  Difference: {difference}')
        self.epoch += 1

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
drop = ['review_id', 'user_id','useful', 'funny', 'cool']
#query = "date >= '2017-12-01' and (stars==1 or stars ==5)"

print("Loading review data.")
with open(f"{PATH}yelp_academic_dataset_review.json", "r") as f:
    reader = pd.read_json(f, orient="records", lines=True, dtype=r_dtypes, chunksize=1000)
    for chunk in reader:
        reduced_chunk = chunk.drop(columns=drop)
        #.query(query)
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
print(f"Businesses in Atl and Ptl shape: {yelp_business.shape}")

yelp_merged = yelp_data.merge(yelp_business, on='business_id')

print(f"Merged data shape: {yelp_merged.shape}")

sentences = yelp_merged.text.astype('str').tolist()
tokenizer = ToktokTokenizer()

print("Removing stopwords.")
sentences_clean=[remove_stopwords(r, stopword_list, tokenizer) for r in sentences]
elapsed = time.process_time() - t
print(f"Stopwords removed. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

print("Cleaning sentences.")
sentences_clean=pd.Series(sentences_clean).apply(denoise_text)
sentences_clean=sentences_clean.apply(remove_special_characters)
sentences_clean=sentences_clean.apply(simple_lemmatizer)
elapsed = time.process_time() - t
print(f"Sentences cleaned. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

print("Removing stopwords again.")
sentences_clean=[remove_stopwords(r, stopword_list, tokenizer) for r in sentences_clean]
elapsed = time.process_time() - t
print(f"Stopwords removed again. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

print("Example sentence:")
print(sentences_clean[0])

print("Tokenizing sentences")
sentences_tokenized = [w.lower() for w in sentences_clean]
sentences_tokenized = [tokenizer.tokenize(i) for i in sentences_tokenized]
elapsed = time.process_time() - t
print(f"Tokenization finished. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

with open('data/yelp_w2v_tokenized.pkl', 'wb') as handle:
    pickle.dump(sentences_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Loading GoogleNews Vectors")
model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
elapsed = time.process_time() - t
print(f"News vectors loaded. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

PHRASING = True
MIN = 500
THRESHOLD = 200
SAVE = True

if PHRASING:
    print("Starting phrasing.")
    sentences_phrased = get_phrases(sentences_tokenized, 
                                    min_count = MIN, 
                                    threshold = THRESHOLD,
                                    save = SAVE)
    sentences_training = sentences_phrased
    elapsed = time.process_time() - t
    print(f"Phrasing done. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")
    print("Phrased sentence examples:")
    print(sentences_training[0])
    print(sentences_training[1])
    print(sentences_training[2])
    with open('data/yelp_w2v_phrased.pkl', 'wb') as handle:
        pickle.dump(sentences_training, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    sentences_training = sentences_tokenized
    

epoch_logger = EpochLogger()
loss_logger = LossLogger()

SIZE = model.vector_size
WINDOW = 10
EPOCHS = 4
MIN_COUNT = 2
SG = 1
HS = 0
SEED = 42
LOSS = True
ALPHA = 0.01

print("Initializing Word2Vec Gensim model with News vectors.")
model_ft = Word2Vec(vector_size= SIZE, 
                    window = WINDOW,
                    min_count= MIN_COUNT,
                    epochs=EPOCHS,
                    sg = SG,
                    hs = HS,
                    seed = SEED)
model_ft.build_vocab(sentences_training)
total_examples = model_ft.corpus_count
model_ft.build_vocab([list(model.key_to_index.keys())], update=True)

outfile = "embeddings/yelp_w2v"
print("Finetuning Word2Vec Gensim model with Yelp data.")
model_ft.train(sentences_training, 
               total_examples=total_examples,
               epochs=model_ft.epochs,
               callbacks=[loss_logger],
               compute_loss=LOSS,
               start_alpha = ALPHA)
elapsed = time.process_time() - t
print(f"Model finetuning finished. {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.") 
print("Saving model.")
model_ft.wv.save_word2vec_format(f"{outfile}.txt", binary=False)
print("Model saved.")