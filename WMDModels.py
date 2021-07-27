random.seed(42)

PATH = "data/"
print("Loading and preparing data.")
sample = pd.read_pickle(f"{PATH}yelp_sample.pkl")
tokenizer = ToktokTokenizer()

pos = sample[sample.sentiment == "positive"].reset_index(drop=True)
neg = sample[sample.sentiment == "negative"].reset_index(drop=True)

pos = pos.review_clean.tolist()
neg = neg.review_clean.tolist()

pos_tok = list(map(lambda x: tokenize(x, tokenizer), pos))
neg_tok = list(map(lambda x: tokenize(x, tokenizer), neg))

pos_sample = [" ".join(doc) for doc in pos_tok]
neg_sample = [" ".join(doc) for doc in neg_tok]

print(f"Positive samples: {len(pos_sample)}")
print(f"Negative samples: {len(neg_sample)}")

finetuned = True

if not finetuned:
    print("Loading GoogleNews Vectors")
    model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
else:
    print("Loading GoogleNews Vectors finetuned using Yelp review data.")
    model = KeyedVectors.load_word2vec_format('embeddings/yelp_w2v.txt', binary=False)

print("Preparing vectorizer and getting oov words.")
corpus = pos_sample + neg_sample
vectorizer = TfidfVectorizer(use_idf=False, tokenizer=tfidf_tokenize, norm='l1')
vectorizer.fit(corpus)

oov = [word for word in vectorizer.get_feature_names() if word not in model.key_to_index.keys()]

print(f"There are {len(oov)} oov words.")
print(f"Example oov words:")
print(oov[:50])

print("Tokenizing samples.")
pos_sample = list(map(lambda x: remove_oov(x, tokenizer, oov), pos_sample))
neg_sample = list(map(lambda x: remove_oov(x, tokenizer, oov), neg_sample))

print("Example of tokenized positive sample:")
print(pos_sample[5])

print("Vectorizing corpus.")
corpus = pos_sample + neg_sample
vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tfidf_tokenize,norm='l1')
vectorizer.fit(corpus)

pos_nbow = vectorizer.transform(pos_sample)
neg_nbow = vectorizer.transform(neg_sample)

pos_tok = list(map(lambda x: tokenize(x, tokenizer), pos_sample))
neg_tok =list(map(lambda x: tokenize(x, tokenizer), neg_sample))

print("Example of vectorized sample:")
print(pos_tok[5][:20])

print("Removing oov.")
oov_ = [word for word in vectorizer.get_feature_names() if word not in model.key_to_index.keys()]
print(f"There are {len(oov_)} oov words left.")

features = vectorizer.get_feature_names()
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
idx2word = {idx: word for idx, word in enumerate(vectorizer.get_feature_names())}

## Create Kmeans for WMD
k = 100

km_base = cluster.KMeans(n_clusters=k,max_iter=300).fit(E)
labels = km_base.labels_
centroids = km_base.cluster_centers_

km_pca = cluster.KMeans(n_clusters=k,max_iter=300).fit(E_pca)
labels_pca = km_pca.labels_

km_umap = cluster.KMeans(n_clusters=k,max_iter=300).fit(E_umap)
labels_umap=km_umap.labels_

km_tsne = cluster.KMeans(n_clusters=k,max_iter=300).fit(E_tsne)
labels_tsne = km_tsne.labels_

lc_rwmd = LC_RWMD(pos_docs, neg_docs,pos_nbow,neg_nbow,E)
lc_rwmd.get_D()

if pairing == 'gale_shapeley':
    print("Running Gale-Shapeley pairing.")
    matcher = Matcher(lc_rwmd.D)
    engaged = matcher.matchmaker()
    matcher.check()
    pairs = engaged
if pairing == 'random':
    print("Running random pairing.")
    pos_idx = list(range(0,len(pos_docs)))
    neg_idx = list(range(0,len(neg_docs)))
    shuffle(pos_idx)
    shuffle(neg_idx)
    pairs = dict(zip(pos_idx, neg_idx))
if pairing == 'full':
    print("Running full pairing.")
    pos_idx = list(range(0,len(pos_docs)))
    neg_idx = list(range(0,len(neg_docs)))
    pairs = [(i,j) for i in pos_idx for j in neg_idx]