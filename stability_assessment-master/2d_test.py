import os
import numpy as np
import pandas as pd
from snf import compute
from statistics import median
from netneurotools import cluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.metrics.cluster import adjusted_rand_score
#from sklearn.metrics.cluster import adjusted_mutual_info_score

from math import sqrt

import matplotlib.pyplot as plt

np.random.seed(1)

metrics = ['cityblock', 'cosine', 'euclidean', 'seuclidean', 'correlation']
#---------------------------------------------------------------------
def sample_circular(npoints, ndim=2):
    vec = np.matrix(np.zeros((npoints, ndim)))
    ii = 0
    while ii<npoints:
        point = np.random.randn(ndim, 1)
        if sqrt(point[0]**2 + point[1]**2)<=1:
            vec[ii,:] = point.T
            ii = ii+1
    return vec

#---------------------------------------------------------------------
def create_sample(center, npoints):
    vec = sample_circular(npoints, ndim=2)
    return vec + np.matrix(center)

#---------------------------------------------------------------------
#Spectral Clustering
def cluster_labels(embed):
    affinities = compute.make_affinity(embed, metric=metr)
    first, second = compute.get_n_clusters(affinities)
    fused_labels = spectral_clustering(affinities, n_clusters=2) #this is not deteministic
    #print('second : ', second)
    return fused_labels,first

#---------------------------------------------------------------------
#Consensus Clustering
def consenus(embed, n):
    ci=[]
    for i in range(n):
        fused_labels,first=cluster_labels(embed) #could be optimized to compute affinities and n_clusters only once
    ci.append(list(fused_labels))
    consensus = cluster.find_consensus(np.column_stack(ci), seed=1234)
    a, =np.unique(consensus).shape
    return consensus, a

#---------------------------------------------------------------------
# Compute Clustering Stability
def ami_cluster_stability(data, true_labels, k, split= 0.20):
    X_sample, X_rest, y_sample, y_rest = train_test_split(data, true_labels, test_size=split)
    affinities = compute.make_affinity(X_sample, metric=metr)
    y_cluster = spectral_clustering(affinities, n_clusters=k)
    return adjusted_rand_score(y_cluster, y_sample)

#---------------------------------------------------------------------
def calculate_metrics(data, n_clusters, labels, iterations=20):
    cluster_stability = []
    for i in range(iterations):
        cluster_stability.append(ami_cluster_stability(data, labels, n_clusters))
    return cluster_stability

#---------------------------------------------------------------------
if not os.path.exists('figs'):
    os.mkdir('figs')
if not os.path.exists('files'):
    os.mkdir('files')

npoints = 200
#For 2-cluster generation
centers = [[2.,4], [3.1,3.5], [3.2,4], [-2,3.6], [-2,4.]]
colors = np.random.rand(5)
raw_data = pd.DataFrame(columns = ['X', 'Y', 'fine cluster', 'coarse cluster'])

# Visualize generated data 
for ii, c_i in enumerate(centers):
    cluster_ii = create_sample(c_i, npoints)
    raw_data_ii = pd.DataFrame(columns = ['X', 'Y', 'fine cluster', 'coarse cluster'])
    raw_data_ii['X'] = pd.Series(cluster_ii[:,0].T.tolist()[0])
    raw_data_ii['Y'] = pd.Series(cluster_ii[:,1].T.tolist()[0])
    raw_data_ii['fine cluster'] = ii+1
    raw_data_ii['coarse cluster'] = ((ii+1)<=3)*1 + ((ii+1)>=4)*2
    raw_data = pd.concat([raw_data, raw_data_ii])
    plt.scatter(np.array(cluster_ii[:,0]), np.array(cluster_ii[:,1]), label = str(ii+1))
plt.legend()
plt.savefig('figs/clusters.png')
plt.figure()
plt.scatter(raw_data['X'], raw_data['Y'], c='k')
plt.savefig('figs/synthetic_data.png')
raw_data.to_csv('files/synthetic_data.csv')
raw_data = pd.read_csv("files/synthetic_data.csv", index_col=0)


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data[['X', 'Y']])
data = raw_data.copy(deep = True)
data[['X', 'Y']] = data_scaled
nb_points = data.shape[0]
points = data[['X', 'Y']].values

for metr in metrics:
    fused_labels_0, clusters_0 = consenus(points, 30)
    data['spectral clustering'] = fused_labels_0
    print('Number of clusters:', clusters_0)
    cluster_stability_0 = calculate_metrics(points, clusters_0, fused_labels_0)
    stability_0 = median(cluster_stability_0)
    print(metr+" stability (current) : ", stability_0)
    #visualize spectral clusters
    plt.figure()
    for ii in range(clusters_0):
        cluster_ii = raw_data[data['spectral clustering'] == ii+1]
        plt.scatter(cluster_ii['X'], cluster_ii['Y'], label = str(ii+1))
    plt.legend()
    plt.title('spectral clusters for '+metr+' metric')
    plt.savefig('figs/spectral_clusters_'+metr+'.png')