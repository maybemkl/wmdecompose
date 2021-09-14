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
"""

class WMD():
    """Full Word Mover's Distance calculated for a pair of documents.
    For details on WMD, see http://proceedings.mlr.press/v37/kusnerb15.html.
    EMD implemented using the pyemd library: https://github.com/wmayner/pyemd
    
    Attributes:
      source: The source document.
      sink: The sink document.
      E: Embedding matrix for the words in the vocabulary.
      metric: Distance metric, default is 'cosine' but can also be 'euclidean'.
    """
    
    def __init__(self, 
                 source:Document, 
                 sink:Document, 
                 E:np.ndarray, 
                 metric:str='cosine') -> None:
        """Initializes class WMD."""
        
        self.source = source
        self.sink = sink
        self.T = E[source.idxs + sink.idxs,]
        if metric == 'cosine':
            self.costs = cosine_distances(self.T, self.T)
        if metric == 'euclidean':
            self.costs = euclidean_distances(self.T, self.T)
        self.source_sig = np.concatenate((source.nbow[0,source.idxs], 
                                      np.zeros(len(sink.idxs))))
        self.sink_sig = np.concatenate((np.zeros(len(source.idxs)), 
                                      sink.nbow[0,sink.idxs]))
        
    def get_distance(self, 
                     i2w:Dict[int, str] = None, 
                     decompose:bool = False) -> Tuple[float, 
                                                      List[float], 
                                                      List[float], 
                                                      List[str], 
                                                      List[str]]:
        """Get the WMD between a pair of documents, with or without decomposed word-level distances.
        
        Args:
          i2w: A dictionary mapping the index of word vectors to the words themselves.
          decompose: A boolean to determine whether word-level distances should be decomposed.

        Returns:
          wmd: The WMD between the pair of documents.
          flow: 
          dist_m: A matrix of the wmd decomposed into word level distances with source in rows and sink in columns.
          w_source: List of the words in the source document.
          w_sink: List of the words in the sink document.
        """
        
        if not decompose:
            wmd = emd(np.array(self.source_sig, dtype=np.double), 
                      np.array(self.sink_sig, dtype=np.double), 
                      np.array(self.costs, dtype=np.double))
            
            return wmd
        
        elif decompose:
            if i2w == None:
                print("i2w argument is missing.")
            else:
                wmd, flow = emd_with_flow(np.array(self.source_sig, dtype=np.double), 
                                          np.array(self.sink_sig, dtype=np.double), 
                                          np.array(self.costs, dtype=np.double))
                w_source = [i2w[idx] for idx in self.source.idxs]
                w_sink = [i2w[idx] for idx in self.sink.idxs]
                dist_m = flow*self.costs
                dist_m = dist_m[:len(self.source.idxs),len(self.source.idxs):].round(5)
                return wmd,flow,dist_m, w_source, w_sink

class RWMD(WMD):
    """Relaxed Word Mover's Distance with matrix operations in numpy. Inherits the WMD class.
    
    Attributes:
      see WMD class
    """
    
    def get_distance(self, 
                     i2w:Dict[int, str] = None, 
                     decompose:bool = False) -> Tuple[float, 
                                                      List[float], 
                                                      List[float], 
                                                      List[float], 
                                                      List[float], 
                                                      List[str], 
                                                      List[str]]:
        """Get the RWMD between a pair of documents, with or without decomposed word-level distances.
        
        Args:
          i2w: A dictionary mapping the index of word vectors to the words themselves.
          decompose: A boolean to determine whether word-level distances should be decomposed.

        Returns:
          rwmd: The RWMD between the pair of documents.
          flow_source: A list of the 'mass' or amount of each word in the source to be moved to the words in the sink.
          flow_sink:  A list of the 'mass' or amount of each word in the sink to be moved to the words in the source.
          dist_source: Array of distance contributions of words from source to sink.
          dist_sink: Array of distance contributions of words from sink to source.
          w_source: A list of words in the source document.
          w_sink: A list of words in the sink document.
        """
        
        if not decompose:
            rwmd, _, _, _, _ = self._rwmd()
            return rwmd
        
        elif decompose:
            if i2w == None:
                print("i2w argument is missing.")
            else:
                rwmd, flow_source, flow_sink, dist_source, dist_sink = self._rwmd()
                w_source = [i2w[idx] for idx in self.source.idxs]
                w_sink = [i2w[idx] for idx in self.sink.idxs]
            return rwmd, flow_source, flow_sink, dist_source, dist_sink, w_source, w_sink

    def _rwmd(self) -> Tuple[float, List[float], List[float], List[float], List[float]]:
        """Get the RWMD with word level flow and distances decomposed.
        
        Args:
          rwmd: The RWMD between the pair of documents.
          flow_source: A list of the amount of flow sent from each word in the source.
          flow_sink: A list of the amount of flow sent from each word in the sink.
          dist_source: Array of distance contributed of words from source.
          dist_sink: Array of distance contributions of words from sink.
        """
        
        flow_source, dist_source = self._rwmd_decompose(self.source_sig, self.sink_sig)
        flow_sink, dist_sink = self._rwmd_decompose(self.sink_sig, self.source_sig)
        rwmd = max(np.sum(dist_source), np.sum(dist_sink))
        return rwmd, flow_source, flow_sink, dist_source, dist_sink
    
    def _rwmd_decompose(self, 
                        source_sig:List[float], 
                        sink_sig:List[float]) -> Tuple[List[float], List[float]]:
        """Decompose RWMD into word-level flow and distances.
        
        Args:
          source_sig:
          sink_sig:
          
        Returns:
          flow: A list with the amount of 
          dist: A list of the distance of words from source to sink.
        """
        
        potential_flow = list(j for j, dj in enumerate(sink_sig) if dj > 0)
        flow = list(min(self.costs[i, potential_flow]) for i in range(len(source_sig)))
        dist = np.multiply(flow, source_sig) 
        return flow, dist
            
class WMDPairs():
    """Word Mover's Distance between two sets of documents.
    
    Attributes:
      flows: 
      source_set: A list of the documents in the source set.
      sink_set: A list of the documents in the sink set.
      pairs: A list of tuples where each tuple has a pair of indices, indicating which document pairs between the two sets to calculate the wmd for.
      E: Embedding matrix for the words in the vocabulary.
      i2w: A dictionary mapping the index of word vectors to the words themselves.
      metric: A string specifying a distance metric. Default is 'cosine' but can also be 'euclidean'.
    """
    
    def __init__(self,
                 source_set:List[Document],
                 sink_set:List[Document],
                 pairs:List[Tuple[int, int]],
                 E:np.ndarray,
                 i2w:Dict[int, str],
                 metric:str='cosine') -> None:
        self.flows = []
        self.wd_source = self._word_dict(source_set)
        self.wd_sink = self._word_dict(sink_set)
        self.distances = np.zeros((len(source_set), len(sink_set)))
        self.source_set = source_set
        self.sink_set = sink_set
        self.E = E
        self.i2w = i2w
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
                      w_sinkc: Dict[str,int] = {}, 
                      c2w: Dict[int, str] = {},
                      thread:bool = False,
                      relax:bool = False) -> None:
        """Get the WMD or RWMD between two sets of documents, with or without decomposed word-level distances.
        
        Args:
          decompose: A boolean to determine whether word-level distances should be decomposed.
          sum_clusters: A boolean to determine whether word-level distances should be summed by cluster.
          w_sinkc: A dictionary mapping words to clusters.
          c2w: A dictionary mapping clusters to words.
          thread: A boolean to determine whether threading should be used.
          relax: A boolean to determine whether RWMD should be returned instead of full WMD.
          
        Note: 
          Threading adds only minimal gains to performance.
        """
        
        self.decompose = decompose
        self.sum_clusters = sum_clusters
        self.source_feat = np.zeros((len(self.pairs),len(c2w)))
        self.sink_feat = np.zeros((len(self.pairs),len(c2w)))
        self.relax = relax
        
        if sum_clusters:
            self.cd_source = {k: 0 for k in c2w.keys()}
            self.cd_sink = {k: 0 for k in c2w.keys()}
            self.w_sinkc = w_sinkc
        
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
                 pair_idx:int) -> None:
        """Get the WMD between two documents, with or without decomposed word-level distances.
        
        Args:
          pair: A tuple of the indexes for the documents in two sets for which WMD should be counted.
          pair_idx: The index of the pair in the list of pairs.
        """
        doc_source = self.source_set[pair[0]]
        doc_sink = self.sink_set[pair[1]]
        
        if self.decompose:
            wmd, _, dist_m, w_source, w_sink = WMD(doc_source, doc_sink, self.E,metric=self.metric).get_distance(self.i2w, 
                                                                                             decompose = True)
            self._add_word_dists(w_source, w_sink, dist_m, pair_idx)
        else:
            wmd = WMD(doc_source, doc_sink, self.E,metric=self.metric).get_distance()
        self.distances[pair[0], pair[1]] = wmd 
    
    def _get_rwmd(self, 
                 pair:Tuple[int,int], 
                 pair_idx:int) -> None:
        """Get the RWMD between two documents, with or without decomposed word-level distances.
        
        Args:
          pair: A tuple of the indexes for the documents in two sets for which RWMD should be counted.
          pair_idx: The index of the pair in the list of pairs.
        """
        doc_source = self.source_set[pair[0]]
        doc_sink = self.sink_set[pair[1]]
        
        if self.decompose:
            rwmd, _, _, dist_source, dist_sink, w_source, w_sink = RWMD(doc_source, 
                                                        doc_sink, 
                                                        self.E,
                                                        metric=self.metric).get_distance(self.i2w,
                                                                                         decompose = True)
            self._add_rwmd_dists(w_source, w_sink, dist_source, dist_sink, pair_idx)
        else:
            rwmd = RWMD(doc_source, doc_sink, self.E,metric=self.metric).get_distance()
        self.distances[pair[0], pair[1]] = rwmd 
    
    def _add_word_dists(self, 
                        w_source:List[str], 
                        w_sink:List[str], 
                        dist_m:np.array,
                        pair_idx:int) -> None:
        """Add the distance of words from source to sink and vice versa for vanilla WMD. 
        If clusters are summed, then distances are added by clusters as well.
        
        Args:
          w_source: A list of words in the source document
          w_sink: A list of words in the sink document
          dist_m: A matrix of the wmd decomposed into word level distances with source in rows and sink in columns.
          pair_idx: The index of the pair in the list of pairs.
        """
        
        for idx,w in enumerate(w_source):
            dist = np.sum(dist_m[idx,:])
            self.wd_source[w] += dist
            if self.sum_clusters:
                self.cd_source[self.w_sinkc[w]] += dist
                self.source_feat[pair_idx,self.w_sinkc[w]] = dist

        for idx,w in enumerate(w_sink):
            dist = np.sum(dist_m[:,idx])
            self.wd_sink[w] += dist
            if self.sum_clusters:
                self.cd_sink[self.w_sinkc[w]] += dist
                self.sink_feat[pair_idx,self.w_sinkc[w]] = dist
                
    def _add_rwmd_dists(self, 
                        w_source:List[str], 
                        w_sink:List[str], 
                        dist_source:np.array, 
                        dist_sink:np.array, 
                        pair_idx:int) -> None:
        """Add the distance contributions of words from source to sink and vice versa for RWMD. 
        If clusters are summed, then distances are added by clusters as well.
        
        Args:
          w_source: A list of words in the source document
          w_sink: A list of words in the sink document
          dist_source: Array of distance contributions of words from source to sink.
          dist_sink: Array of distance contributions of words from sink to source.
          pair_idx: The index of the pair in the list of pairs.
        """
        
        for idx,w in enumerate(w_source):
            dist = np.sum(dist_source[idx])
            self.wd_source[w] += dist
            if self.sum_clusters:
                self.cd_source[self.w_sinkc[w]] += dist
                self.source_feat[pair_idx,self.w_sinkc[w]] = dist
                
        for idx,w in enumerate(w_sink):
            sink_idx = len(w_source) + idx
            dist = np.sum(dist_sink[sink_idx])
            self.wd_sink[w] += dist
            if self.sum_clusters:
                self.cd_sink[self.w_sinkc[w]] += dist
                self.sink_feat[pair_idx,self.w_sinkc[w]] = dist

    def get_differences(self) -> None:
        """Get differences in accumulated word-by-word distances between two sets of documents.
        For details, see equation 8 on page 5 in Brunila & Violette (2021).
        """
        
        self.wd_source_diff = dict(self.wd_source)
        self.wd_sink_diff = dict(self.wd_sink)
        self.wd_source_diff = self._count_diff(self.wd_source, self.wd_sink, self.wd_source_diff)
        self.wd_sink_diff = self._count_diff(self.wd_sink, self.wd_source, self.wd_sink_diff)

    def _count_diff(self, 
                    cl_source:Dict[str,float], 
                    cl_sink:Dict[str,float], 
                    output:Dict[str,float]) -> Dict[str,float]:
        """Loop for retrieving the differences in accumulated word-by-word distances between two sets of documents.
        For details, see equation 8 on page 5 in Brunila & Violette (2021).
        
        Args:
          cl_source: A dictionary of words and their accumulated distances
          cl_sink: A dictionary of words and their accumulated distances
          output: A dictionary with the source word level distances as initial values.
        Returns:
          output: The updated difference between word distances when substracting distances in cl_sink from distances in cl_source
        """
        
        for k, v in cl_source.items():
            try:
                output[k] = v - cl_sink[k]
            except:
                output[k] = v
        return output
    
class LC_RWMD():
    """Linear-Complexity Relaxed Word Mover's Distance using matrix operations in numpy.
    Implemented following Atasu et al. (2020).
    For details, see https://arxiv.org/abs/1711.07227
    
    Attributes:
      source_set: First set of documents. 'X1' in original paper.
      sink_set: Second set of documents. 'X2' in original paper.
      source_nbow: Nbow or other vectorized (such as Tf-Idf) matrix representation of the documents in source_set.
      sink_nbow: Nbow or other vectorized (such as Tf-Idf) matrix representation of the documents in sink_set.
      E: Embedding matrix for the words in the vocabulary.
      D1: The RWMD distance matrix of documents from source_set to sink_set.
      D2: The RWMD distance matrix of documents from sink_set to source_set.
    """
    
    def __init__(self,
                 source_set:List[Document],
                 sink_set:List[Document],
                 source_nbow:csr_matrix,
                 sink_nbow:csr_matrix,
                 E:np.ndarray) -> None:
        self.source_set = source_set
        self.sink_set = sink_set
        self.source_nbow = source_nbow
        self.sink_nbow = sink_nbow
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
        
        for idsink, sink in enumerate(self.sink_set):
            if metric == 'cosine':
                Z = cosine_distances(self.E, sink.vecs).min(axis=1)
            if metric == 'euclidean':
                Z = euclidean_distances(self.E, sink.vecs).min(axis=1)
            lc_rwmd = np.dot(self.source_nbow.toarray(), Z)
            self.D1.append(lc_rwmd)

        for idsource, source in enumerate(self.source_set):
            if metric == 'cosine':
                Z = cosine_distances(self.E, source.vecs).min(axis=1)
            if metric == 'euclidean':
                Z = euclidean_distances(self.E, source.vecs).min(axis=1)
            lc_rwmd = np.dot(self.sink_nbow.toarray(), Z)
            self.D2.append(lc_rwmd)
        self.D = np.maximum(np.vstack(self.D1), np.vstack(np.transpose(self.D2)))
