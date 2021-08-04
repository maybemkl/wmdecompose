import numpy as np

class Document():
    def __init__(self, words:list, nbow:object, word2idx:dict, E:np.array, doc_idx:int):
        self.words = words
        #self.nbow = vectorizer.transform([" ".join(words)])
        self.nbow = nbow.toarray()
        self.weights_sum = np.sum(self.nbow)
        self.idxs = list(set([word2idx[word] for word in words]))
        self.vecs = E[self.idxs,]
        self.doc_idx = doc_idx
