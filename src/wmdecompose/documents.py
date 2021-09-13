from scipy.sparse.csr import csr_matrix
from typing import Dict, List

import numpy as np

class Document():
    """Class to create the input for wmdecompose.
    
    Attributes:
      words: A list of the words in the document.
      nbow: Nbow or other vectorized (such as Tf-Idf) matrix representation of the document.
      word2idx: A dictionary mapping words to their representation in the embedding matrix E.
      E: Embedding matrix for the words in the vocabulary.
    """
    
    def __init__(self, 
                 words:List[str], 
                 nbow:csr_matrix, 
                 word2idx:Dict[str,int], 
                 E:np.ndarray):
        self.words = words
        self.nbow = nbow.toarray()
        self.weights_sum = np.sum(self.nbow)
        self.idxs = list(set([word2idx[word] for word in words]))
        self.vecs = E[self.idxs,]
