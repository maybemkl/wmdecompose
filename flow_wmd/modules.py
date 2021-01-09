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
        self.vocab = self.source.words + self.sink.words
        
        ## SECOND APPROACH:
        # self.vocab = list(set(self.source.words + self.sink.words))
        return len(self.vocab), self.vocab
    
    def getsignature(self):
        self._getvocab()

        self.sig1 = np.append(self.source.weights, np.zeros(len(self.vocab)-len(self.source.weights)))
        self.sig2 = np.append(np.zeros(len(self.vocab)-len(self.sink.weights)), self.sink.weights)
        self.vecs = self.source.vecs + self.sink.vecs
        self.idx1 = [idx for idx, s in enumerate(self.sig1) if s>0]
        self.idx2 = [idx for idx, s in enumerate(self.sig2) if s>0]
        
        ## SECOND APPROACH
        #self.sig1 = [self.source.getweight(w) if w in self.source.words else 0.0 for w in self.vocab]
        #self.sig2 = [self.sink.getweight(w) if w in self.sink.words else 0.0 for w in self.vocab]
        #self.vecs = [self.source.getvec(w) if w in self.source.words else self.sink.getvec(w) for w in self.vocab]
        #self.idx1 = [idx for idx, s in enumerate(self.sig1) if s>0]
        #self.idx2 = [idx for idx, s in enumerate(self.sig2) if s>0]

    def emd(self):
        self.w2v_distances = euclidean_distances(self.vecs, self.vecs)
        self.w2v_emd, self.w2v_flow = emd_with_flow(np.array(self.sig1, dtype=np.double), 
                                          np.array(self.sig2, dtype=np.double), 
                                          np.array(self.w2v_distances, dtype=np.double))
        return self.w2v_emd, self.w2v_flow
    
    def getCost(self):
        cost_m = self.w2v_flow*self.w2v_distances
        cost_m = cost_m[np.ix_(self.idx1,self.idx2)]
        source_cost = np.sum(cost_m, axis=1).round(5)
        sink_cost = np.sum(cost_m, axis=0).round(5)
        self.source_cost = dict(zip(self.source.words, list(source_cost)))
        self.sink_cost = dict(zip(self.sink.words, list(sink_cost)))
