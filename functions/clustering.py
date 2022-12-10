from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pyspark
from scipy.spatial.distance import pdist, squareform

def display_clusters(X, y=None, means=None):
    if y is None:
        y = 'grey'
    plt.scatter(X[:, 0], X[:, 1], c=y)
    if means is not None:
        m = np.vstack(means)
        plt.scatter(m[:,0], m[:,1], c=range(m.shape[0]), marker='^')
    plt.show()

def k_means(X, K, steps=100, display=False, early_stopping=.005):
    rdd = sc.parallelize(X)
    # initialize the first k means randomly
    means = rdd.takeSample(False, K)
    y = None
    for _ in range(steps):
        if display:
            display_clusters(X, y, means)
        # Assign step
        # calculate all the distances AND
        # take the min for each of the samples, and assign it
        clusters = rdd.keyBy(lambda x: \
                       np.argmin( [np.linalg.norm(x - m) for m in means] )
                    )

        if early_stopping is not None:
            # assign clusters
            y_new = np.array(clusters.keys().collect())
            diff = sum(y_new != y) / X.shape[0]
            if diff < early_stopping:
                break
            y = y_new

        # Update step
        # take the mean of each group and update the means
        sum_by_cluster = clusters.reduceByKey(lambda a,b: a + b)
        size_by_cluster = clusters.countByKey()
        means = [m / size_by_cluster[c] for c,m in sorted(sum_by_cluster.collect())]

    return y, means

def distance_matrix(X,y,means):
    # Average Centroid Distance
    K = len(means)
    dist = np.zeros((K,K))
    for i in range(K):
        # from cluster i
        for j in range(K):
            # to cluster j
            # find all the points in the cluster 
            cluster = X[y==j]
            # find all the distances from the centroid
            dist[i,j] = np.linalg.norm(cluster - means[i], axis=1).mean()
    return dist / np.var(X)

# average distance between all of the objects and the cluster center
def avg_intra_cluster_distance(X,y,means):
    '''Centroid Diameter Distance'''
    return np.diag(distance_matrix(X,y,means)).mean()

# distance between the center of a cluster and all the objects belonging to a different cluster
def avg_inter_cluster_distance(X,y,means):
    '''Average Centroid Linkage Distance'''
    dist = distance_matrix(X,y,means)
    rest = []
    for i in range(dist.shape[1]): 
        for j in range(dist.shape[0]):
            if i==j:
                continue
            rest.append(dist[i,j])
    return np.array(rest).mean()

# # average distance between all the objects belonging to the same cluster
# def avg_intra_cluster_distance(X,y,means):
#     '''Average Diameter Distance'''
#     k = len(means)
#     dist = np.zeros(k)
#     for i in range(k):
#         cluster = X[y==i]
#         # print(len(cluster))
#         dist[i] = np.mean(pdist(cluster, 'seuclidean'))

#     return 2*dist.mean()

# # average distance between all pairs of elements in different clusters
# def avg_inter_cluster_distance(X,y,means):
#     '''Average Linkage Distance'''
#     # distance matrix
#     dist = squareform(pdist(X))
#     k = len(means)
#     score = np.zeros(k)
#     for i in range(k):
#         # selects all the distances from a certain cluster to all the other clusters
#         mask = np.logical_not(np.logical_xor(np.array([y==i]).T, y!=i))
#         score[i] = dist[mask].mean() 
#     return score.sum()

def elbow(func, X, k_values=range(2,11), subsample=1,steps=20, k_means=k_means):
    score = defaultdict(lambda: 0)
    # subsampling from the whole dataset
    samples = np.random.randint(X.shape[0], size = int(X.shape[0]*subsample))
    X = X[samples]
    for _ in range(steps):
        for k in k_values:
            y, means = k_means(X,k, display=False)
            score[k]  += func(X,y,means)

    return {k:s/steps for k,s in score.items()}

def elbow(func, X, Y, means, k_values):
    score = defaultdict(lambda: 0)
    for i, k in enumerate(k_values):
        y = Y[i,:]
        score[k] += func(X,y, means[i])

    return score