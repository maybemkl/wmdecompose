from collections import Counter, namedtuple
from gensim.models import KeyedVectors
from pyemd import emd_with_flow
from sklearn.metrics import euclidean_distances

import numpy as np

class Document():
    def __init__(self, words):
        self.words = words
        self.vecs = []
        self.weights = np.zeros(len(words))
        self.weights_sum = 0
    
    def instantiate(self, model):
        total, counts = self._getcounts(self.words)
        for i, w in enumerate(self.words):
            self.vecs.append(model[w])
            self.weights[i] = counts[w] / total
        self.weights_sum = np.sum(self.weights)
    
    def getweight(self, word):
        if word in self.words:
            return self.weights[self.words.index(word)]
        else:
            return None
        
    def getvec(self, word):
        if word in self.words:
            return self.vecs[self.words.index(word)]
        else:
            return None
    
    def data(self):
        return list(zip(self.words, self.weights, self.vecs))
    
    def _getcounts(self, words):
        counts = Counter(words)
        total = sum(counts.values())
        return total, counts
    
class DocPair():
    def __init__(self, source, sink):
        self.source = source
        self.sink = sink
        self.vecs = []
        self.vocab = []
    
    def _getvocab(self):
        self.vocab = list(set(self.source.words + self.sink.words))
        return len(self.vocab), self.vocab
    
    def getsignature(self):
        self._getvocab()
        #self.sig1, vec1 = map(list,zip(*self._getWeightsAndVecs(self.doc1, vocab)))
        #self.sig2, vec2 = map(list,zip(*self._getWeightsAndVecs(self.doc2, vocab)))
        #self.vecs = 
        self.sig1 = [self.source.getweight(w) if w in self.source.words else 0.0 for w in self.vocab]
        self.sig2 = [self.sink.getweight(w) if w in self.sink.words else 0.0 for w in self.vocab]
        self.vecs = [self.source.getvec(w) if w in self.source.words else self.sink.getvec(w) for w in self.vocab]
     
    def emd(self):
        self.w2v_distances = euclidean_distances(self.vecs, self.vecs)
        w2v_emd, w2v_flow = emd_with_flow(np.array(self.sig1, dtype=np.double), 
                                          np.array(self.sig2, dtype=np.double), 
                                          np.array(self.w2v_distances, dtype=np.double))
        return w2v_emd, w2v_flow
        
  #  def getpile(self):
        
   # def _getWeightsAndVecs(self, doc, vocab):
   #     weights, vecs = [(doc.getweight(w), doc.getvec(w)) if w in doc.words else 0.0 for w in vocab]
   #     print(weights)
   #     print(vecs)
   #     return weights, vecs
