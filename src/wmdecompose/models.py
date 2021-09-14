from concurrent.futures import ThreadPoolExecutor
from .documents import Document
from pyemd import emd, emd_with_flow
from scipy.sparse.csr import csr_matrix
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from typing import Dict, List, Tuple

import itertools
import numpy as np
import time

"""
TODO
- Docstring parameters and return values with variable explanations.
- Rename all X1 and X2 to source and sink?
- Rename cost to distance, where appropriate
"""

class WMD():
    """Full Word Mover's Distance calculated for a pair of documents.
    For details on WMD, see http://proceedings.mlr.press/v37/kusnerb15.html.
    EMD implemented using the pyemd library: https://github.com/wmayner/pyemd
    
    Attributes:
      x1: The source document.
      x2: The sink document.
      E: Embedding matrix for the words in the vocabulary.
      metric: Distance metric, default is 'cosine' but can also be 'euclidean'.
    """
    
    def __init__(self, 
                 x1:Document, 
                 x2:Document, 
                 E:np.ndarray, 
                 metric:str='cosine') -> None:
        """Initializes class WMD."""
        
        self.x1 = x1
        self.x2 = x2
        self.T = E[x1.idxs + x2.idxs,]
        if metric == 'cosine':
            self.C = cosine_distances(self.T, self.T)
        if metric == 'euclidean':
            self.C = euclidean_distances(self.T, self.T)
        self.x1_sig = np.concatenate((x1.nbow[0,x1.idxs], 
                                      np.zeros(len(x2.idxs))))
        self.x2_sig = np.concatenate((np.zeros(len(x1.idxs)), 
                                      x2.nbow[0,x2.idxs]))
        
    def get_distance(self, 
                     idx2word:Dict[int, str] = None, 
                     decompose:bool = False) -> Tuple[float, 
                                                      List[float], 
                                                      List[float], 
                                                      List[str], 
                                                      List[str]]:
        """Get the WMD between a pair of documents, with or without decomposed word-level distances.
        
        Args:
          idx2word: A dictionary mapping the index of word vectors to the words themselves.
          decompose: A boolean to determine whether word-level distances should be decomposed.

        Returns:
          wmd: The WMD between the pair of documents.
          flow: 
          cost_m: A matrix of the 
          w1: List of the words in the source document.
          w2: List of the words in the sink document.
        """
        
        if not decompose:
            wmd = emd(np.array(self.x1_sig, dtype=np.double), 
                      np.array(self.x2_sig, dtype=np.double), 
                      np.array(self.C, dtype=np.double))
            
            return wmd
        
        elif decompose:
            if idx2word == None:
                print("idx2word argument is missing.")
            else:
                wmd, flow = emd_with_flow(np.array(self.x1_sig, dtype=np.double), 
                                          np.array(self.x2_sig, dtype=np.double), 
                                          np.array(self.C, dtype=np.double))
                w1 = [idx2word[idx] for idx in self.x1.idxs]
                w2 = [idx2word[idx] for idx in self.x2.idxs]
                cost_m = flow*self.C
                cost_m = cost_m[:len(self.x1.idxs),len(self.x1.idxs):].round(5)
                return wmd,flow,cost_m, w1, w2

class RWMD(WMD):
    """Relaxed Word Mover's Distance with matrix operations in numpy. Inherits the WMD class.
    
    Attributes:
      see WMD class
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
        """Get the RWMD between a pair of documents, with or without decomposed word-level distances.
        
        Args:
          idx2word: A dictionary mapping the index of word vectors to the words themselves.
          decompose: A boolean to determine whether word-level distances should be decomposed.

        Returns:
          rwmd: The RWMD between the pair of documents.
          flow_X1:
          flow_X2:
          cost_X1: 
          cost_X2:
          w1: A list of words in the source document.
          w2: A list of words in the sink document.
        """
        
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
        """Get the RWMD with word level flow and cost decomposed.
        
        Args:
          rwmd: The RWMD between the pair of documents.
          flow_X1: 
          flow_X2:
          cost_X1:
          cost_X2:
        """
        
        flow_X1, cost_X1 = self._rwmd_decompose(self.X1_sig, self.X2_sig)
        flow_X2, cost_X2 = self._rwmd_decompose(self.X2_sig, self.X1_sig)
        rwmd = max(np.sum(cost_X1), np.sum(cost_X2))
        return rwmd, flow_X1, flow_X2, cost_X1, cost_X2
    
    def _rwmd_decompose(self, 
                        source_sig:List[float], 
                        sink_sig:List[float]) -> Tuple[List[float], List[float]]:
        """Decompose RWMD into word-level flow and cost.
        
        Args:
          source_sig:
          sink_sig:
          
        Returns:
          flow:
          cost:
        """
        
        potential_flow = list(j for j, dj in enumerate(sink_sig) if dj > 0)
        flow = list(min(self.C[i, potential_flow]) for i in range(len(source_sig)))
        cost = np.multiply(flow, source_sig) 
        return flow, cost
            
class WMDPairs():
    """Word Mover's Distance between two sets of documents.
    
    Attributes:
      flows:
      X1_set:
      X2_set
      pairs: A list of tuples where each tuple has a pair of indices, indicating which document pairs between the two sets to calculate the wmd for.
      E: Embedding matrix for the words in the vocabulary.
      idx2word: A dictionary mapping the index of word vectors to the words themselves.
      metric: Distance metric, default is 'cosine' but can also be 'euclidean'.
    """
    
    def __init__(self,
                 X1_set:List[Document],
                 X2_set:List[Document],
                 pairs:List[Tuple[int, int]],
                 E:np.ndarray,
                 idx2word:Dict[int, str],
                 metric:str='cosine') -> None:
        self.flows = []
        self.wc_X1 = self._word_dict(X1_set)
        self.wc_X2 = self._word_dict(X2_set)
        self.distances = np.zeros((len(X1_set), len(X2_set)))
        self.X1_set = X1_set
        self.X2_set = X2_set
        self.E = E
        self.idx2word = idx2word
        self.pairs = pairs
        self.metric = metric
        
    def _word_dict(self, docs:list) -> Dict[str, int]:
        """Create dictionary for decomposed and accumulated word distances in one document set.
        
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
        """Get the WMD or RWMD between two sets of documents, with or without decomposed word-level distances.
        
        Args:
          decompose: A boolean to determine whether word-level distances should be decomposed.
          sum_clusters: A boolean to determine whether word-level distances should be summed by cluster.
          w2c: A dictionary mapping words to clusters.
          c2w: A dictionary mapping clusters to words.
          thread: A boolean to determine whether threading should be used.
          relax: A boolean to determine whether RWMD should be returned instead of full WMD.
          
        Note: 
          Threading adds only minimal gains to performance.
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
        """Get the WMD between two documents, with or without decomposed word-level distances.
        
        Args:
          pair: A tuple of the indexes for the documents in two sets for which WMD should be counted.
          doc_idx: The index of the pair in the list of pairs.
        """
        doc1 = self.X1_set[pair[0]]
        doc2 = self.X2_set[pair[1]]
        
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
        """Get the RWMD between two documents, with or without decomposed word-level distances.
        
        Args:
          pair: A tuple of the indexes for the documents in two sets for which RWMD should be counted.
          doc_idx: The index of the pair in the list of pairs.
        """
        doc1 = self.X1_set[pair[0]]
        doc2 = self.X2_set[pair[1]]
        
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
        """Add the cost of words from source to sink and vice versa for vanilla WMD. 
        If clusters are summed, then costs are added by clusters as well.
        
        Args:
          w1: A list of words in the source document
          w2: A list of words in the sink document
          cost_m:
          doc_idx:
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
        """Add the cost of words from source to sink and vice versa for RWMD. 
        If clusters are summed, then costs are added by clusters as well.
        
        Args:
          w1: A list of words in the source document
          w2: A list of words in the sink document
          cost_X1:
          cost_X2:
          doc_idx:
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
        """Get differences in accumulated word-by-word distances between two sets of documents.
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
        """Loop for retrieving the differences in accumulated word-by-word distances between two sets of documents.
        For details, see equation 8 on page 5 in Brunila & Violette (2021).
        
        Args:
          cluster1: A dictionary of words and their accumulated distances
          cluster2: A dictionary of words and their accumulated distances
          output: The difference between word distances when substracting distances in cluster2 from distances in cluster1
          
        Returns:
          output:
        """
        
        for k, v in cluster1.items():
            try:
                output[k] = v - cluster2[k]
            except:
                output[k] = v
        return output
    
class LC_RWMD():
    """Linear-Complexity Relaxed Word Mover's Distance using matrix operations in numpy.
    Implemented following Atasu et al. (2020).
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
        Initializes the LC_RWMD class.
        """
        
    def get_D(self, metric:str='cosine')->None:
        """The one-to-many LC-RWMD in Atasu et al (2020).
        For details, see https://arxiv.org/abs/1711.07227
        
        Args:
          metric: Distance metric, default is 'cosine' but can also be 'euclidean'.
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
