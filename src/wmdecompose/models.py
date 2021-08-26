from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor
from .documents import Document
from pyemd import emd, emd_with_flow
from scipy.sparse.csr import csr_matrix
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from typing import Dict, List, Tuple

import bottleneck as bn
import itertools
import numpy as np
import time

"""
TODO
- Docstring parameters and return values with variable explanations.
"""

class WMD():
    """
    Full Word Mover's Distance with EMD implemented using the pyemd library.
    For details on WMD, see http://proceedings.mlr.press/v37/kusnerb15.html.
    """
    
    def __init__(self, 
                 X1:Document, 
                 X2:Document, 
                 E:np.ndarray, 
                 metric:str='cosine') -> None:
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
        
    def get_distance(self, 
                     idx2word:Dict[int, str] = None, 
                     decompose:bool = False) -> Tuple[float, 
                                                      List[float], 
                                                      List[float], 
                                                      List[str], 
                                                      List[str]]:
        """Get WMD between a pair of documents, with or without decomposed word-level distances.
        
        Args:
          idx2word: A dictionary mapping the index of word vectors to the words themselves.
          decompose: A boolean to determine whether word-level distances should be decomposed.

        Returns:
          wmd: The WMD between the pair of documents.
          flow: A . Returned only if word-level distances are decomposed.
          cost_m: Returned only if word-level distances are decomposed.
          w1: Returned only if word-level distances are decomposed.
          w2: Returned only if word-level distances are decomposed.
        """
        
        if not decompose:
            wmd = emd(np.array(self.X1_sig, dtype=np.double), 
                      np.array(self.X2_sig, dtype=np.double), 
                      np.array(self.C, dtype=np.double))
            
            return wmd
        
        elif decompose:
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
                return wmd,flow,cost_m, w1, w2

class RWMD(WMD):
    """
    Relaxed Word Mover's Distance with matrix operations in numpy. Inherits the WMD class.
    """
    
    def get_distance(self, 
                     idx2word:Dict[int, str] = None, 
                     decompose:bool = False) -> Tuple[float, 
                                                      List[float], 
                                                      List[float], 
                                                      List[float], 
                                                      List[float], 
                                                      List[str], 
                                                      List[str]]:
        if not decompose:
            rwmd, _, _, _, _ = self._rwmd()
            return rwmd
        
        elif decompose:
            if idx2word == None:
                print("idx2word argument is missing.")
            else:
                rwmd, flow_X1, flow_X2, cost_X1, cost_X2 = self._rwmd()
                w1 = [idx2word[idx] for idx in self.X1.idxs]
                w2 = [idx2word[idx] for idx in self.X2.idxs]
            return rwmd, flow_X1, flow_X2, cost_X1, cost_X2, w1, w2

    def _rwmd(self) -> Tuple[float, List[float], List[float], List[float], List[float]]:
        """
        Get the RWMD with word level flow and cost decomposed.
        """
        
        flow_X1, cost_X1 = self._rwmd_decompose(self.X1_sig, self.X2_sig)
        flow_X2, cost_X2 = self._rwmd_decompose(self.X2_sig, self.X1_sig)
        rwmd = max(np.sum(cost_X1), np.sum(cost_X2))
        return rwmd, flow_X1, flow_X2, cost_X1, cost_X2
    
    def _rwmd_decompose(self, 
                        source_sig:List[float], 
                        sink_sig:List[float]) -> Tuple[List[float], List[float]]:
        """
        Decompose RWMD into word-level flow and cost.
        """
        
        potential_flow = list(j for j, dj in enumerate(sink_sig) if dj > 0)
        flow = list(min(self.C[i, potential_flow]) for i in range(len(source_sig)))
        cost = np.multiply(flow, source_sig) 
        return flow, cost
            
class WMDPairs():
    """
    Word Mover's Distance between two sets of documents.
    """
    
    def __init__(self,
                 X1:Document,
                 X2:Document,
                 pairs:Dict[int, int],
                 E:np.ndarray,
                 idx2word:Dict[int, str],
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
        
    def _word_dict(self, docs:list) -> Dict[str, int]:
        """
        Create dictionary for decomposed and accumulated word distances in one document set.
        
        Args:
          docs: A list of all the documents in the document set.
          
        Returns:
          word_dict: A dictionary with words as keys and zeroes as values.
        """
        
        vocab = list(set(list(itertools.chain.from_iterable(doc.words for doc in docs))))
        word_dict = {word: 0 for word in vocab}
        return word_dict
        
    def get_distances(self, 
                      decompose: bool = False, 
                      sum_clusters: bool = False, 
                      w2c: Dict[str,int] = {}, 
                      c2w: Dict[int, str] = {},
                      thread:bool = False,
                      relax:bool = False) -> None:
        """
        Get the WMD or RWMD between two sets of documents, with or without decomposed word-level distances.
        """
        
        self.decompose = decompose
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
                if self.relax==True:
                    t = time.process_time()
                    for idx, pair in enumerate(self.pairs):
                        future = executor.submit(self._get_rwmd, pair, idx)
                        futures.append(future)
                        if idx % 1000 == 0:
                            elapsed = time.process_time() - t
                            print(f"Calculated distances between approximately {idx} documents."
                                  f"{time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")
                else:
                    for idx, pair in enumerate(self.pairs):
                        future = executor.submit(self._get_wmd, pair, idx)
                        futures.append(future) 
                        if idx % 1000 == 0:
                            print(f"Calculated distances between approximately {idx} documents.")
 
        
        else:
            t = time.process_time()
            for idx, pair in enumerate(self.pairs):
                if self.relax==True:
                    self._get_rwmd(pair, idx)
                else:
                    self._get_wmd(pair, idx)
                if idx % 1000 == 0:
                    elapsed = time.process_time() - t
                    print(f"Calculated distances between approximately {idx} documents."
                          f"{time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))} elapsed.")

    def _get_wmd(self, 
                 pair:Tuple[int,int], 
                 doc_idx:int) -> None:
        doc1 = self.X1[pair[0]]
        doc2 = self.X2[pair[1]]
        """
        Get the WMD between two documents, with or without decomposed word-level distances.
        """
        
        if self.decompose:
            wmd, _, cost_m, w1, w2 = WMD(doc1, doc2, self.E,metric=self.metric).get_distance(self.idx2word, 
                                                                                             decompose = True)
            self._add_word_costs(w1, w2, cost_m, doc_idx)
        else:
            wmd = WMD(doc1, doc2, self.E,metric=self.metric).get_distance()
        self.distances[pair[0], pair[1]] = wmd 
    
    def _get_rwmd(self, 
                 pair:Tuple[int,int], 
                 doc_idx:int) -> None:
        doc1 = self.X1[pair[0]]
        doc2 = self.X2[pair[1]]
        """
        Get the RWMD between two documents, with or without decomposed word-level distances.
        """

        if self.decompose:
            rwmd, _, _, cost_X1, cost_X2, w1, w2 = RWMD(doc1, 
                                                        doc2, 
                                                        self.E,
                                                        metric=self.metric).get_distance(self.idx2word,
                                                                                         decompose = True)
            self._add_rwmd_costs(w1, w2, cost_X1, cost_X2, doc_idx)
        else:
            rwmd = RWMD(doc1, doc2, self.E,metric=self.metric).get_distance()
        self.distances[pair[0], pair[1]] = rwmd 
    
    def _add_word_costs(self, 
                        w1:List[str], 
                        w2:List[str], 
                        cost_m:np.array,
                        doc_idx:int) -> None:
        """
        Add the cost of words from source to sink and vice versa for vanilla WMD. 
        If clusters are summed, then costs are added by clusters as well.
        """
        
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
                
    def _add_rwmd_costs(self, 
                        w1:List[str], 
                        w2:List[str], 
                        cost_X1:np.array, 
                        cost_X2:np.array, 
                        doc_idx:int) -> None:
        """
        Add the cost of words from source to sink and vice versa for RWMD. 
        If clusters are summed, then costs are added by clusters as well.
        """
        
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
        """
        Get differences in accumulated word-by-word distances between two sets of documents.
        For details, see equation 8 on page 5 in Brunila & Violette (2021).
        """
        
        self.wc_X1_diff = dict(self.wc_X1)
        self.wc_X2_diff = dict(self.wc_X2)
        self.wc_X1_diff = self._count_diff(self.wc_X1, self.wc_X2, self.wc_X1_diff)
        self.wc_X2_diff = self._count_diff(self.wc_X2, self.wc_X1, self.wc_X2_diff)

    def _count_diff(self, 
                    cluster1:Dict[str,float], 
                    cluster2:Dict[str,float], 
                    output:Dict[str,float]) -> Dict[str,float]:
        """
        Loop for retrieving the differences in accumulated word-by-word distances between two sets of documents.
        For details, see equation 8 on page 5 in Brunila & Violette (2021).
        """
        
        for k, v in cluster1.items():
            try:
                output[k] = v - cluster2[k]
            except:
                output[k] = v
        return output
    
class LC_RWMD():
    """Linear-Complexity Relaxed Word Mover's Distance using matrix operations in numpy.
    Implemented following Atasu et al (2020).
    For details, see https://arxiv.org/abs/1711.07227
    
    Attributes:
      X1_set: First set of documents.
      X2_set: Second set of documents.
      X1_nbow: Nbow or other vectorized (such as Tf-Idf) matrix representation of the documents in X1_set.
      X2_nbow: Nbow or other vectorized (such as Tf-Idf) matrix representation of the documents in X2_set.
      E: Embedding matrix for the words in the vocabulary.
      D1: The RWMD distance matrix of documents from X1_set to X2_set.
      D2: The RWMD distance matrix of documents from X2_set to X1_set.
    """
    
    def __init__(self,
                 X1_set:List[Document],
                 X2_set:List[Document],
                 X1_nbow:csr_matrix,
                 X2_nbow:csr_matrix,
                 E:np.ndarray) -> None:
        self.X1_set = X1_set
        self.X2_set = X2_set
        self.X1_nbow = X1_nbow
        self.X2_nbow = X2_nbow
        self.E = E
        self.D1, self.D2 = [], []
        """
        Initializes LC_RWMD class.
        """
        
    def get_D(self, metric:str='cosine')->None:
        """The one-to-many LC-RWMD in Atasu et al (2020).
        For details, see https://arxiv.org/abs/1711.07227
        
        Args:
          metric: Distance metric to be used, default is 'cosine' but can optionally be 'euclidean'.
        """
        
        for idx2, x2 in enumerate(self.X2_set):
            if metric == 'cosine':
                Z = cosine_distances(self.E, x2.vecs).min(axis=1)
            if metric == 'euclidean':
                Z = euclidean_distances(self.E, x2.vecs).min(axis=1)
            lc_rwmd = np.dot(self.X1_nbow.toarray(), Z)
            self.D1.append(lc_rwmd)

        for idx1, x1 in enumerate(self.X1_set):
            if metric == 'cosine':
                Z = cosine_distances(self.E, x1.vecs).min(axis=1)
            if metric == 'euclidean':
                Z = euclidean_distances(self.E, x1.vecs).min(axis=1)
            lc_rwmd = np.dot(self.X2_nbow.toarray(), Z)
            self.D2.append(lc_rwmd)

        self.D = np.maximum(np.vstack(self.D1), np.vstack(np.transpose(self.D2)))
