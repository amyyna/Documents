import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import radius_neighbors_graph
 

raw_data = pd.read_csv("files/synthetic_data.csv", index_col=0)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data[['X', 'Y']])
# Get only the clinical-scale features (omitting those marked as Covariate, if applicable)
data = raw_data.copy(deep = True)
data[['X', 'Y']] = data_scaled
points = data[['X', 'Y']].values


A = radius_neighbors_graph(points, 0.3, mode='connectivity', metric='minkowski', p=2, include_self=True)
print("Shape of adjacency matrix:",A.shape)

# Visualizing adjacency matrix

sns.heatmap(A.todense())
plt.savefig("figs/Adjacency.jpg")
plt.show()

# Creating degree matrix
# By calculating degrees using row sum
D = np.diag(A.todense().sum(axis=1).ravel().tolist()[0])

print("Shape of degree matrix:",D.shape)

# Visualizing adjacency matrix
sns.heatmap(D)
plt.savefig("figs/Degree.jpg")
plt.show()

# Calculating Graph Laplacian
L = D - A

# Visualizing Laplacian matrix
plt.figure()
sns.heatmap(L)
plt.savefig("figs/Laplacian.jpg")
plt.show()

# Calculating eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(L)

# Getting threshold
# Second smallest eigenvalue is chosen
#threshold = np.where(eigenvalues == np.partition(eigenvalues,1)[1])
#print("Threshold:",threshold[0][0])

sorted_eigenvalues = np.sort(eigenvalues)
eigengap = np.abs(np.diff(sorted_eigenvalues))
index_largest_gap = np.argmax(eigengap)
nCl = index_largest_gap + 1
print('Number of clusters:', nCl)


spec = SpectralClustering(n_clusters = nCl, affinity='nearest_neighbors', assign_labels='kmeans').fit(points)

# Getting clusters
clusters = spec.labels_

# Visualizing the clusters
plt.figure()
plt.scatter(points[:, 0], points[:, 1], c=clusters)
plt.title("Spectral Clustering with radius neighbors graph")
plt.savefig("figs/radius_SC.jpg") # Saving plot as a file
plt.show()

print('sorted eigs : ', sorted_eigenvalues[0:10])
print('eigengaps : ', eigengap[0:10])

from math import sqrt
sparsity = np.array(np.zeros((A.shape[0])))
sqr = []
weighted_cumsum = []
for ii in range(A.shape[0]):
    size_Aii = (ii+1)**2
    sparsity[ii] = 1.0 - A[:ii+1,:][:,:ii+1].count_nonzero() / size_Aii
    sqr.append(sqrt(sparsity[ii]))
    weighted_cumsum.append(np.mean((sparsity[:ii+1])))
weighted_cumsum = np.array(weighted_cumsum)

import scipy
from scipy.signal import savgol_filter
sgf = savgol_filter(sparsity**2, 31, 2)
d1 = np.insert(np.diff(sgf), 0, 0)
d1 = d1/np.max(d1)

all_start_indices = []
all_end_indices = []
cluster_switch = []
all_slopes = []
slope = 0
index = 0
for ii in range(1,A.shape[0]):
    if d1[ii]>=d1[ii-1]:
        slope = slope + (d1[ii]-d1[ii-1])
    else:
        if slope>0.1:
            all_start_indices.append(index)
            all_end_indices.append(ii-1)
            cluster_switch.append((all_start_indices[-1]+all_end_indices[-1])/2)
            all_slopes.append(slope)
        slope = 0
        index = ii

print('\n\n\nindices : ', all_start_indices)
print('\n\n\nindices : ', all_end_indices)
print('\n\n\nnew cluster at : ', cluster_switch)
print('\n\n\nslopes : ', all_slopes)


d1smooth = savgol_filter(d1**2, 31, 2)
d2 = np.insert(np.diff(d1smooth), 0, 0)
d2 = d2/np.max(d2)

d2smooth = savgol_filter(d2**2, 31, 2)
d3 = np.insert(np.diff(d2smooth), 0, 0)
d3 = d3/np.max(d3)

plt.figure()
plt.plot(sparsity, label = 'sparsity')
plt.plot(sparsity**2, label = 'sparsity^2')
for node in cluster_switch:
    plt.axvline(x=node, color = 'm')
plt.title('sparsity curve')
plt.legend()
plt.savefig('figs/sparsity.png')

plt.figure()
plt.plot(sparsity-weighted_cumsum)
plt.title('cumsum')
plt.savefig('figs/cumsum.png')

plt.figure()
plt.plot(d1, label = '1st deriv')
#plt.plot(d2, label = '2nd deriv')
#plt.plot(d3, label = '3rd deriv')
for node in cluster_switch:
    plt.axvline(x=node, color = 'm')
plt.title('sparsity derivative')
plt.legend()
plt.savefig('figs/d1.png')