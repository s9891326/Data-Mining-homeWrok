# import os     
# os.environ["PATH"] += os.pathsep + 'E:\\nltk_data'

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from time import time
import tarfile
import numpy as np

# 解壓縮
tar = tarfile.open("mini_newsgroups.tar.gz", "r:gz")
tar.extractall()
tar.close()

# 原始資料集

original_dataset = datasets.load_files('mini_newsgroups')

# 設定stopwords為英文
stopwords = set(stopwords.words('english'))

# 轉換編碼方式、並記錄頻率
tfv = TfidfVectorizer(encoding = 'ISO-8859-1', stop_words = stopwords)
trans_dataset = tfv.fit_transform(original_dataset.data).toarray()
'''
# K-means
t1 = time()

kmeans = KMeans(n_clusters = 20).fit(trans_dataset)

print('K-means')
print('花費時間:', time() - t1, '秒')

print('Silhouette Coefficient:', silhouette_score(trans_dataset, kmeans.labels_))

print('SSE:', kmeans.inertia_)

kmeans_list = {i: [int(x / 100 + 1) for x in np.where(kmeans.labels_ == i)[0]] for i in range(kmeans.n_clusters)}
purity_list = []

purity = 0
count = 0
for i in range(20):
    for j in range(21):
        if count < kmeans_list[i].count(j):
            count = kmeans_list[i].count(j)
            
    result = count / len(kmeans_list[i])
    
    purity += len(kmeans_list[i]) / 2000 * result
    
print('purity:', purity)
'''

# DBSCAN
t2 = time()

dbscan = DBSCAN(eps = 1, min_samples = 4).fit(trans_dataset)

print('DBSCAN')
print('花費時間:', time() - t2, '秒')

dbscan_list = {i: [int(x / 100 + 1) for x in np.where(dbscan.labels_ == i)[0]] for i in range(dbscan.n_clusters)}
purity_list = []

purity = 0
count = 0
for i in range(20):
    for j in range(21):
        if count < dbscan_list[i].count(j):
            count = dbscan_list[i].count(j)
            
    result = count / len(dbscan_list[i])
    
    purity += len(dbscan_list[i]) / 2000 * result
    
print('purity:', purity)

'''
# 階層式分群
def plot_dendrogram(model, **kwargs):

    children = model.children_

    distance = np.arange(children.shape[0])

    no_of_observations = np.arange(2, children.shape[0]+2)

    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

t3 = time()

hierarchical = AgglomerativeClustering(n_clusters=20)
hierarchical = hierarchical.fit(trans_dataset)

print('hierarchical')
print('花費時間:', time() - t3, '秒')

plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(hierarchical, )
plt.savefig('dendrogram')

hiera_list = {i: [int(x / 100 + 1) for x in np.where(hierarchical.labels_ == i)[0]] for i in range(hierarchical.n_clusters)}
purity_list = []

purity = 0
count = 0
for i in range(20):
    for j in range(21):
        if count < hiera_list[i].count(j):
            count = hiera_list[i].count(j)
            
    result = count / len(hiera_list[i])
    
    purity += len(hiera_list[i]) / 2000 * result
    
print('purity:', purity)
'''