import numpy as np
import pandas as pd
from snf import compute
from statistics import median
from netneurotools import cluster
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.metrics.cluster import adjusted_mutual_info_score

np.random.seed(1)

#---------------------------------------------------------------------
#Spectral Clustering
def cluster_labels(embed):
    affinities = compute.make_affinity(embed, metric='cosine')
    #print("affinities = ", affinities)
    first, second = compute.get_n_clusters(affinities)
    fused_labels = spectral_clustering(affinities, n_clusters=first)#this is not deteministic
    #print('second : ', second)
    return fused_labels,first

#---------------------------------------------------------------------
#Consensus Clustering
def consenus(embed, n):
    ci=[]
    for i in range(n):
        fused_labels,first=cluster_labels(embed)#could be optimized to compute affinities and n_clusters only once
        ci.append(list(fused_labels))
    consensus = cluster.find_consensus(np.column_stack(ci), seed=1234)
    a, =np.unique(consensus).shape
    return consensus, a

#---------------------------------------------------------------------
# Compute Clustering Stability
def ami_cluster_stability(data, true_labels, k, split= 0.20):
    X_sample, X_rest, y_sample, y_rest = train_test_split(data, true_labels, test_size=split)
    affinities = compute.make_affinity(X_sample, metric='cosine')
    y_cluster = spectral_clustering(affinities, n_clusters=k)
    return adjusted_mutual_info_score(y_cluster, y_sample)

#---------------------------------------------------------------------
def calculate_metrics(data, n_clusters, labels, iterations=20):
    cluster_stability = []
    for i in range(iterations):
        cluster_stability.append(ami_cluster_stability(data, labels, n_clusters))
    return cluster_stability

#---------------------------------------------------------------------
data = pd.read_csv("files/synthetic_data.csv", index_col=0)
nb_points = data.shape[0]

points = data[['X', 'Y']].values
#fused_labels_1, clusters_1 = consenus(points, 30)
#print('Number of clusters:', clusters_1)

clusters_1 = 5
fused_labels_1 = list(data['fine cluster'].values)
cluster_stability_1 = calculate_metrics(points, clusters_1, fused_labels_1)
stability_1 = median(cluster_stability_1)
print("5-clusters' stability : ", stability_1)

clusters_2 = 2
fused_labels_2 = list(data['coarse cluster'].values)
cluster_stability_2 = calculate_metrics(points, clusters_2, fused_labels_2)
stability_2 = median(cluster_stability_2)
print("2-clusters' stability : ", stability_2)


#Rectified stability
##This version is deterministic (as a proof of concept)

data_A = data[data['coarse cluster'] == 1]
points_A = data_A[['X', 'Y']].values
clusters_A = 3
fused_labels_A = list(data_A['fine cluster'].values)

data_B = data[data['coarse cluster'] == 2]
points_B = data_B[['X', 'Y']].values
clusters_B = 2
fused_labels_B = list(data_B['fine cluster'].values)


wcs_A = median(calculate_metrics(points_A, clusters_A, fused_labels_A))*len(fused_labels_A)/nb_points
print("within cluster stability (right) : ", wcs_A)
wcs_B = median(calculate_metrics(points_B, clusters_B, fused_labels_B))*len(fused_labels_B)/nb_points
print("within cluster stability (left) : ", wcs_B)

wcs = wcs_A + wcs_B
print("within cluster stability : ", wcs)

rectified_stability = stability_2 - wcs
print("rectified 2-clusters' stability : ", rectified_stability)


"""
first_A = spectral_clustering(A)
first_B = spectral_clustering(B)
"""