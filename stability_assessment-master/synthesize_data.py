import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt


np.random.seed(1)

def sample_spherical(npoints, ndim=2):
    """
    Parameters
    ----------
    npoints : nb points to generate
    ndim : dimension of the sampling space: 2 for a disc, 3 for a sphere, etc
    Returns
    -------
    vec : array containing generated points in a 0-centered disc/sphere (rows are the points, cols are the coordinates)

    """
    vec = np.matrix(np.zeros((npoints, ndim)))
    ii = 0
    while ii<npoints:
        point = np.random.randn(ndim, 1)
        #print(point[0,0], point[1,0])
        if point[0,0]**2 + point[1,0]**2 <= 1:
            vec[ii,:] = point.T
            ii = ii+1
    return vec

def create_sample(center, npoints):
    """
    Parameters
    ----------
    center : centre of the disc/sphere
    npoints : nb points
    Returns
    -------
    points in the disc with a center defined by center
    """
    vec = sample_spherical(npoints, ndim=2)
    return vec + np.matrix(center)

npoints = 100
#For stable 5-cluster generation
#centers = [[2.,4], [3.1,2.1], [4.2,4], [-2,2.], [-2,4.1]]
#For 2-cluster generation
centers = [[2.,4], [3.1,3.5], [3.2,4], [-2,3.6], [-2,4.]]
colors = np.random.rand(5)

data = pd.DataFrame(columns = ['X', 'Y', 'fine cluster', 'coarse cluster'])
for ii, c_i in enumerate(centers):
    cluster = create_sample(c_i, npoints)
    data_ii = pd.DataFrame(columns = ['X', 'Y', 'fine cluster', 'coarse cluster'])
    data_ii['X'] = pd.Series(cluster[:,0].T.tolist()[0])
    data_ii['Y'] = pd.Series(cluster[:,1].T.tolist()[0])
    data_ii['fine cluster'] = ii+1
    data_ii['coarse cluster'] = ((ii+1)<=3)*1 + ((ii+1)>=4)*2
    #print("ghi1")
    if ii == 0:
        data = data_ii
    else:
        data = pd.concat([data, data_ii])
    #print("ghi2")
    plt.scatter(np.array(cluster[:,0]), np.array(cluster[:,1]), label = str(ii+1))
plt.legend()
plt.savefig('figs/clusters.png')

plt.figure()
plt.scatter(data['X'], data['Y'], c='k')
plt.savefig('figs/synthetic_data.png')

data.to_csv('files/synthetic_data.csv')
plt.show()