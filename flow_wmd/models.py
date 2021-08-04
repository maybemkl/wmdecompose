from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor
from .documents import Document
from pyemd import emd, emd_with_flow
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

import bottleneck as bn
import itertools
import numpy as np
import time

"""
TODO
- Add Threading to WMDPairs.get_distances()
- Cluster input and output
- Typing
- Docstrings
- Remove redundant classes and functions
"""

class WMD():
    def __init__(self, X1:Document, X2:Document, E:np.array, metric:str='cosine')->None:
        self.X1 = X1
        self.X2 = X2
        self.T = E[X1.idxs + X2.idxs,]
        if metric == 'cosine':
            self.C = cosine_distances(self.T, self.T)
        if metric == 'euclidean':
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

class RWMD(WMD):
    def get_distance(self, idx2word:dict = None, return_flow:bool = False):
        if not return_flow:
            rwmd, _, _, _, _ = self._rwmd()
            return rwmd
        
        elif return_flow:
            if idx2word == None:
                print("idx2word argument is missing.")
            else:
                rwmd, flow_X1, flow_X2, cost_X1, cost_X2 = self._rwmd()
                w1 = [idx2word[idx] for idx in self.X1.idxs]
                w2 = [idx2word[idx] for idx in self.X2.idxs]
            return rwmd, flow_X1, flow_X2, cost_X1, cost_X2, w1, w2

    def _rwmd(self):
        potential_flow_X1 = list(j for j, dj in enumerate(self.X2_sig) if dj > 0)
        potential_flow_X2 = list(i for i, di in enumerate(self.X1_sig) if di > 0)
        flow_X1 = list(min(self.C[i, potential_flow_X1]) for i in range(len(self.X1_sig)))
        flow_X2 = list(min(self.C[j, potential_flow_X2]) for j in range(len(self.X2_sig)))
        cost_X1 = np.multiply(flow_X1, self.X1_sig) 
        cost_X2 = np.multiply(flow_X2, self.X2_sig)
        rwmd = max(np.sum(cost_X1), np.sum(cost_X2))
        return rwmd, flow_X1, flow_X2, cost_X1, cost_X2
            
class WMDPairs():
    def __init__(self,
                 X1:Document,
                 X2:Document,
                 pairs:dict,
                 E:np.ndarray,
                 idx2word:dict,
                 metric:str='cosine') -> None:
        self.flows = []
        self.wc_X1 = self._word_dict(X1)
        self.wc_X2 = self._word_dict(X2)
        self.distances = np.zeros((len(X1), len(X2)))
        self.X1 = X1
        self.X2 = X2
        self.E = E
        self.idx2word = idx2word
        self.pairs = pairs
        self.metric = metric
        
    def _word_dict(self, docs) -> dict:
        vocab = list(set(list(itertools.chain.from_iterable(doc.words for doc in docs))))
        word_dict = {word: 0 for word in vocab}
        return word_dict
        
    def get_distances(self, 
                      return_flow: bool = False, 
                      sum_clusters: bool = False, 
                      w2c: list = [], 
                      c2w: dict = {},
                      thread = False,
                      relax = False) -> None:
        self.return_flow = return_flow
        self.sum_clusters = sum_clusters
        self.X1_feat = np.zeros((len(self.pairs),len(c2w)))
        self.X2_feat = np.zeros((len(self.pairs),len(c2w)))
        self.relax = relax
        
        if sum_clusters:
            self.cc_X1 = {k: 0 for k in c2w.keys()}
            self.cc_X2 = {k: 0 for k in c2w.keys()}
            self.w2c = w2c
        
        if thread:
            futures = []
            with ThreadPoolExecutor(max_workers=15) as executor:
                if not relax:
                    for idx, pair in enumerate(self.pairs):
                        future = executor.submit(self._get_wmd, pair, idx)
                        futures.append(future) 
                        if idx % 1000 == 0:
                            print(f"Calculated distances between approximately {idx} documents.")
                else:
                    t = time.process_time()
                    for idx, pair in enumerate(self.pairs):
                        future = executor.submit(self._get_rwmd, pair, idx)
                        futures.append(future)
                        if idx % 1000 == 0:
                            elapsed = time.process_time() - t
                            print(f"Calculated distances between approximately {idx} documents."
                                  f"{time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")
        
        else:
            t = time.process_time()
            for idx, pair in enumerate(self.pairs):
                if not relax:
                    self._get_wmd(pair, idx)
                if relax:
                    self._get_rwmd(pair, idx)
                if idx % 1000 == 0:
                    elapsed = time.process_time() - t
                    print(f"Calculated distances between approximately {idx} documents."
                          f"{time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

    def _get_wmd(self, pair:tuple, doc_idx:int)->None:
        doc1 = self.X1[pair[0]]
        doc2 = self.X2[pair[1]]
        if self.return_flow:
            wmd, _, cost_m, w1, w2 = WMD(doc1, doc2, self.E,metric=self.metric).get_distance(self.idx2word, 
                                                                                             return_flow = True)
            print(wmd)
            self._add_word_costs(w1, w2, cost_m, doc_idx)
        else:
            wmd = WMD(doc1, doc2, self.E,metric=self.metric).get_distance()
        self.distances[pair[0], pair[1]] = wmd 
    
    def _get_rwmd(self, pair:tuple, doc_idx:int)->None:
        doc1 = self.X1[pair[0]]
        doc2 = self.X2[pair[1]]
        if self.return_flow:
            rwmd, _, _, cost_X1, cost_X2, w1, w2 = RWMD(doc1, 
                                                        doc2, 
                                                        self.E,
                                                        metric=self.metric).get_distance(self.idx2word,
                                                                                         return_flow = True)
            self._add_rwmd_costs(w1, w2, cost_X1, cost_X2, doc_idx)
        else:
            rwmd = RWMD(doc1, doc2, self.E,metric=self.metric).get_distance()
        self.distances[pair[0], pair[1]] = rwmd 
    
    def _add_word_costs(self, w1: list, w2: list, cost_m, doc_idx:int)->None:
        for idx,w in enumerate(w1):
            cost = np.sum(cost_m[idx,:])
            self.wc_X1[w] += cost
            if self.sum_clusters:
                self.cc_X1[self.w2c[w]] += cost
                self.X1_feat[doc_idx,self.w2c[w]] = cost

        for idx,w in enumerate(w2):
            cost = np.sum(cost_m[:,idx])
            self.wc_X2[w] += cost
            if self.sum_clusters:
                self.cc_X2[self.w2c[w]] += cost
                self.X2_feat[doc_idx,self.w2c[w]] = cost
                
    def _add_rwmd_costs(self, w1:list, w2:list, cost_X1:np.array, cost_X2:np.array, doc_idx:int):
        for idx,w in enumerate(w1):
            cost = np.sum(cost_X1[idx])
            self.wc_X1[w] += cost
            if self.sum_clusters:
                self.cc_X1[self.w2c[w]] += cost
                self.X1_feat[doc_idx,self.w2c[w]] = cost
                
        for idx,w in enumerate(w2):
            x2_idx = len(w1) + idx
            cost = np.sum(cost_X2[x2_idx])
            self.wc_X2[w] += cost
            if self.sum_clusters:
                self.cc_X2[self.w2c[w]] += cost
                self.X2_feat[doc_idx,self.w2c[w]] = cost

    def get_differences(self) -> None:
        self.wc_X1_diff = dict(self.wc_X1)
        self.wc_X2_diff = dict(self.wc_X2)
        self.wc_X1_diff = self._count_diff(self.wc_X1, self.wc_X2, self.wc_X1_diff)
        self.wc_X2_diff = self._count_diff(self.wc_X2, self.wc_X1, self.wc_X2_diff)

    def _count_diff(self, cluster1:dict, cluster2:dict, output:dict) -> dict:
        for k, v in cluster1.items():
            try:
                output[k] = v - cluster2[k]
            except:
                output[k] = v
        return output
    
class LC_RWMD():
    def __init__(self,
                 X1:Document,
                 X2:Document,
                 X1_nbow,X2_nbow,E:np.array)->None:
        self.D1, self.D2 = [], []
        self.X1 = X1
        self.X2 = X2
        self.X1_nbow = X1_nbow
        self.X2_nbow = X2_nbow
        self.E = E
        
    def get_D(self, metric:str='cosine')->None:
        # Atasu et al LC-RWMD: One-to-many
        for idx2, doc2 in enumerate(self.X2):
            if metric == 'cosine':
                Z = cosine_similarity(self.E, doc2.vecs).min(axis=1)
            if metric == 'euclidean':
                Z = euclidean_distances(self.E, doc2.vecs).min(axis=1)
            lc_rwmd = np.dot(self.X1_nbow.toarray(), Z)
            self.D1.append(lc_rwmd)

        for idx1, doc1 in enumerate(self.X1):
            if metric == 'cosine':
                Z = cosine_similarity(self.E, doc1.vecs).min(axis=1)
            if metric == 'euclidean':
                Z = euclidean_distances(self.E, doc1.vecs).min(axis=1)
            lc_rwmd = np.dot(self.X2_nbow.toarray(), Z)
            self.D2.append(lc_rwmd)

        self.D = np.maximum(np.vstack(self.D1), np.vstack(np.transpose(self.D2)))
        
    def get_L(self, n:int)->None:
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
            
            
    def get_rwmd(self)->None:
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
