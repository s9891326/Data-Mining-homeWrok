from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
import numpy as np
from time import time

dataset = datasets.load_files('mini_newsgroups', shuffle = False)

stopwords = set(stopwords.words('english'))

tfv = TfidfVectorizer(encoding = 'ISO-8859-1', stop_words = stopwords)

training_tfv = tfv.fit_transform(dataset.data).toarray()

t0 = time()

kmeans = KMeans(n_clusters = 20).fit(training_tfv)

print('花費時間:', time() - t0, '秒')

print('Silhouette Coefficient:', silhouette_score(training_tfv, kmeans.labels_))

print('SSE:', kmeans.inertia_)

result_list = [[0 for i in range(20)] for j in range(20)]

for k in range(2000):
    result = kmeans.predict(training_tfv[k][:].reshape(1, -1))
    
    result_list[int(result)][int(k / 100)] += 1
    
print('result_list:', result_list)

purity_list = []

purity = 0
    
for l in range(20):
    result = max(result_list[l]) / sum(result_list[l])
    
    purity += sum(result_list[l]) / 2000 * result
    
print('purity:', purity)
