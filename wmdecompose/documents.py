from scipy.sparse.csr import csr_matrix
from typing import Dict, List

import numpy as np

class Document():
    def __init__(self, 
                 words:List[str], 
                 nbow:csr_matrix, 
                 word2idx:Dict[str,int], 
                 E:np.ndarray, 
                 doc_idx:int):
        self.words = words
        #self.nbow = vectorizer.transform([" ".join(words)])
        self.nbow = nbow.toarray()
        self.weights_sum = np.sum(self.nbow)
        self.idxs = list(set([word2idx[word] for word in words]))
        self.vecs = E[self.idxs,]
        self.doc_idx = doc_idx
