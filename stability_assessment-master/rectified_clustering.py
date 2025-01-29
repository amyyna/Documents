import numpy as np
import scipy as sc

from scipy.sparse import csgraph
from numpy import linalg as LA

import pandas as pd
from netneurotools import cluster
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import SpectralClustering

from sklearn.neighbors import kneighbors_graph

import matplotlib.pyplot as plt

np.random.seed(13967)

#---------------------------------------------------------------------
#Spectral Clustering
def cluster_labels(embed):
    graph = kneighbors_graph(embed, mode='connectivity', n_neighbors = 10)
    laplacian = csgraph.laplacian(graph, normed=True)
    eigenvalues, eigenvectors = sc.sparse.linalg.eigs(laplacian)
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1]
    list_nb_clusters = index_largest_gap + 1
    first, second = list_nb_clusters[0], list_nb_clusters[1]
    clustering = SpectralClustering(affinity = 'precomputed', n_clusters = first).fit(graph) #this is not deteministic
    fused_labels = clustering.labels_
    print('second : ', second)
    return fused_labels, first

#---------------------------------------------------------------------
#Consensus Clustering
def consenus(embed, n):
    ci=[]
    for i in range(n):
        fused_labels, first = cluster_labels(embed) #could be optimized to compute affinities and n_clusters only once
    ci.append(list(fused_labels))
    consensus = cluster.find_consensus(np.column_stack(ci), seed=1234)
    a, =np.unique(consensus).shape
    return consensus, a

#---------------------------------------------------------------------

raw_data = pd.read_csv("files/synthetic_data.csv", index_col=0)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data[['X', 'Y']])

# Get only the clinical-scale features (omitting those marked as Covariate, if applicable)
data = raw_data.copy(deep = True)
data[['X', 'Y']] = data_scaled

nb_points = data.shape[0]

points = data[['X', 'Y']].values

"""
# perform eigendecomposition and find eigengap
graph = kneighbors_graph(points, mode='connectivity', n_neighbors = 20)
laplacian = csgraph.laplacian(graph, normed=True)
eigenvalues, eigenvectors = sc.sparse.linalg.eigs(laplacian)
index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:5]
nb_clusters = index_largest_gap + 1

print('Number of clusters:', nb_clusters)

nCl = nb_clusters[0]
spectral_model_rbf = SpectralClustering(n_clusters=nCl, affinity='rbf')
y_cluster = spectral_model_rbf.fit_predict(graph)

data['spectral clustering'] = y_cluster + 1
"""
fused_labels, nCl = consenus(points, 30)
print('Number of clusters:', nCl)

print("ghid ")

raw_data['spectral clustering'] = fused_labels

for ii in range(nCl):
    cluster = raw_data[raw_data['spectral clustering'] == ii+1]
    plt.scatter(cluster['X'], cluster['Y'], label = str(ii+1))

plt.legend()
plt.savefig('figs/rectified_spectral_clusters.png')
