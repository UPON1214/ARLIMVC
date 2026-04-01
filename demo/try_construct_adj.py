import scipy.io
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances as pair


def get_similarity_matrix(features, method='heat'):
    """Get the similarity matrix"""
    dist = None
    if method == 'heat': # 使用热核（heat kernel）方法计算相似性矩阵：论文中的Sv
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        # features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    return dist

def get_graph2(features, topk=1440, method='heat'):
    """Generate graph adjacency matrix using different similarity methods"""
    dist = get_similarity_matrix(features, method=method)
    num_nodes = dist.shape[0]

    # Initialize the matrix with -1, assuming node indices are non-negative.
    neighbors_matrix = np.full((num_nodes,num_nodes), -1, dtype=int)

    for i in range(num_nodes):
        # Get the indices of the topk+1 largest elements (including the node itself)
        ind = np.argpartition(dist[i, :], -topk)[-topk:]
        # Remove the node itself from the neighbors' list
        #ind = ind[ind != i]
        # Sort the remaining indices by similarity in descending order
        sorted_ind = ind[np.argsort(dist[i, ind])[::-1]]
        # Only keep the topk neighbors
        neighbors_matrix[i, :min(topk, len(sorted_ind))] = sorted_ind[:topk]

    return neighbors_matrix