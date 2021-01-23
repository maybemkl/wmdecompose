from collections import Counter, namedtuple
from pyemd import emd, emd_with_flow
from sklearn.metrics import euclidean_distances

import bottleneck as bn
import numpy as np

class Document():
    def __init__(self, words, nbow, word2idx, E):
        self.words = words
        #self.nbow = vectorizer.transform([" ".join(words)])
        self.nbow = nbow.toarray()
        self.weights_sum = np.sum(self.nbow)
        self.idxs = list(set([word2idx[word] for word in words]))
        self.vecs = E[self.idxs,]
    
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

    
class WMD():
    def __init__(self, X1, X2, E):
        self.X1 = X1
        self.X2 = X2
        self.T = E[X1.idxs + X2.idxs,]
        self.C = euclidean_distances(self.T, self.T)
        self.X1_sig = np.concatenate((X1.nbow[0,X1.idxs], 
                                      np.zeros(len(X2.idxs))))
        self.X2_sig = np.concatenate((np.zeros(len(X1.idxs)), 
                                      X2.nbow[0,X2.idxs]))
        
    def get_distance(self, idx2word = None, return_flow = False):
        if not return_flow:
            wmd = emd(np.array(self.X1_sig, dtype=np.double), 
                      np.array(self.X2_sig, dtype=np.double), 
                      np.array(self.C, dtype=np.double))
            
            return wmd
        
        elif return_flow:
            if idx2word == None:
                print("idx2word argument is missing.")
            else:
                wmd, flow = emd_with_flow(np.array(self.X1_sig, dtype=np.double), 
                                          np.array(self.X2_sig, dtype=np.double), 
                                          np.array(self.C, dtype=np.double))
                #m = np.array(self.wmd_wf[1])[:len(self.X1.idxs),len(self.X1.idxs):]
                w1 = [idx2word[idx] for idx in self.X1.idxs]
                w2 = [idx2word[idx] for idx in self.X2.idxs]
                cost_m = flow*self.C
                cost_m = cost_m[:len(self.X1.idxs),len(self.X1.idxs):].round(5)
                return (wmd,flow,cost_m, w1, w2)

            
class WMDCluster():
    def __init__(self,X1,X2,E,idx2word):
        self.flows = []
        #self.distances = []
        self.distances = np.zeros((len(X1), len(X2)))
        self.X1 = X1
        self.X2 = X2
        self.E = E
        self.idx2word = idx2word
        
    def get_distances(self, return_flow = False):
        if not return_flow:
            for idx1, doc1 in enumerate(self.X1):
                for idx2, doc2 in enumerate(self.X2):
                    wmd = WMD(doc1, doc2, self.E).get_distance()
                    #self.distances.append({'idx1':idx1, 
                    #                       'idx2':idx2, 
                    #                       'wmd':wmd})
                    self.distances[idx1, idx2] = wmd
            print(self.distances)
                    
        elif return_flow:
            for idx1, doc1 in enumerate(self.X1):
                for idx2, doc2 in enumerate(self.X2):
                    if not return_flow:
                        self.flows.append(WMD(doc1, doc2, self.E).get_distance(self.idx2word, return_flow = True))
            return self.flows

class LC_RWMD():
    def __init__(self,X1, X2,X1_nbow,X2_nbow,E):
        self.D1, self.D2 = [], []
        self.X1 = X1
        self.X2 = X2
        self.X1_nbow = X1_nbow
        self.X2_nbow = X2_nbow
        self.E = E
        
    def get_D(self):
        # Atasu et al LC-RWMD: One-to-many
        for idx2, doc2 in enumerate(self.X2):
            Z = euclidean_distances(self.E, doc2.vecs).min(axis=1)
            lc_rwmd = np.dot(self.X1_nbow.toarray(), Z)
            self.D1.append(lc_rwmd)

        for idx1, doc1 in enumerate(self.X1):
            Z = euclidean_distances(self.E, doc1.vecs).min(axis=1)
            lc_rwmd = np.dot(self.X2_nbow.toarray(), Z)
            self.D2.append(lc_rwmd)

        self.D = np.maximum(np.vstack(self.D1), np.vstack(np.transpose(self.D2)))
        
    def get_L(self, n):
        self.Ls = []
        for idx1, doc1 in enumerate(self.X1):
            values = bn.partition(self.D[idx1], self.D[idx1].size-n)[:-n]
            indeces = bn.argpartition(self.D[idx1], self.D[idx1].size-n)[:-n]
            WMDs = []
            for idx2 in indeces:
                doc2 = self.X2[idx2]
                wmd = WMD(doc1, 
                          doc2, 
                          self.E).get_distance()
                WMDs.append(wmd)
            L = max(WMDs)
            self.Ls.append((idx1, L))
            
            
    def get_rwmd(self):
        self.wmd_s = []
        for L in self.Ls:
            for idx2, row in enumerate(self.D[L[0]]):
                if row < L[1]:
                    wmd = WMD(self.X1[L[0]], 
                              self.X2[idx2],
                              self.E).get_distance()
                    self.wmd_s.append(wmd)
                else:
                    pass

