from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
import numpy as np
from time import time

dataset = datasets.load_files('mini_newsgroups', shuffle = False)

stopwords = set(stopwords.words('english'))

tfv = TfidfVectorizer(encoding = 'ISO-8859-1', stop_words = stopwords)

training_tfv = tfv.fit_transform(dataset.data).toarray()

t0 = time()

clustering = DBSCAN(eps = 3, min_samples = 5).fit(training_tfv)

print('花費時間:', time() - t0, '秒')

print(clustering.labels_)

#print('Silhouette Coefficient:', silhouette_score(training_tfv, clustering.labels_))