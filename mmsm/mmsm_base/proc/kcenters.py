"""K-Centers clustering"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>
# Modified by: Nir Nitskansky <nir.nitskansky@mail.huji.ac.il>

import numpy as np
import numba
from sklearn.neighbors import NearestNeighbors
from mmsm.mmsm_base.util import get_unique_id
from mmsm.mmsm_base.base_discretizer import BaseDiscretizer

__all__ = ["KCentersDiscretizer"]

def k_centers(data : np.ndarray, k=None, cutoff=None):
    """k_centers.
    Implementation of k-centers clustering algorithm [1]_.


    Parameters
    ----------
    data : np.ndarray((N,d))
        a set of N datapoints with dimension d
    k :
        maximum unmber of centers
    cutoff :
        maximum distance between points and their cluster centers

    Returns
    -------
    (nn, clusters, centers)
    nn : sklearn.neighbors.NearestNeighbors
        a nearest neighbors object fitted on the cluster centers
    clusters : np.ndarray(N, dtype=np.int)
        the cluster assignments of the datapoints
    centers : list
        a list of the cluster centers

    References
    ----------
    [1] Gonzalez T (1985) Clustering to minimize the maximum intercluster distance. Theor Comput Sci 38:293
    """
    if k is None and cutoff is None:
        raise ValueError("at least one of k or cutoff must be defined")
    N = data.shape[0]
    clusters = np.zeros(N, dtype=int)
    centers = []
    distances = np.zeros(N)
    stop_conditions = []

    if k is not None:
        stop_conditions.append(lambda : len(centers) >= k)
    if cutoff is not None:
        stop_conditions.append(lambda : np.max(distances) <= cutoff)

    stop_condition = lambda : np.any([check_condition() for check_condition in stop_conditions])

    centers.append(data[np.random.choice(N)])
    distances = np.linalg.norm(data - centers[0], axis=1)

    i = 0
    while not stop_condition():
        i += 1
        _k_centers_step(i, data, clusters, centers, distances)
    return NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers), clusters, centers

def extend_k_centers(data, nn, centers, cutoff):
    """extend_k_centers.
    Given a set of center points, and a new set of data points, extends the list of centers
    until all the new points are within a given distance from a cluster center.

    Parameters
    ----------
    data :
        data
    nn : sklearn.neighbors.NearestNeighbors
        a nearest neighbors object fitted on the cluster centers
    centers : list
        centers
    cutoff :
        cutoff

    Returns
    -------
    (nn, clusters, centers)
    nn : sklearn.neighbors.NearestNeighbors
        a nearest neighbors object fitted on the cluster centers
    clusters : np.ndarray(N, dtype=np.int)
        the cluster assignments of the datapoints
    centers : list
        a list of the cluster centers
    """
    distances, clusters = nn.kneighbors(data)
    distances = distances.squeeze()
    clusters = clusters.squeeze()
    i = len(centers)

    while np.max(distances) >= cutoff:
        _k_centers_step(i, data, clusters, centers, distances)
        i += 1
    return NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers), clusters, centers


def _k_centers_step(i, data, clusters, centers, distances):
    new_center = _k_centers_step_njit(i, data, clusters, distances)
    centers.append(new_center)


@numba.njit
def _k_centers_step_njit(i, data, clusters, distances):
    """_k_centers_step.
    """
    new_center = data[np.argmax(distances)]
    dist_to_new_center = np.sum((data - new_center) ** 2, axis=1) ** 0.5
    # dist_to_new_center_san = np.linalg.norm(data - new_center, axis=1)
    updated_points = np.where(dist_to_new_center < distances)[0]
    clusters[updated_points] = i
    distances[updated_points] = dist_to_new_center[updated_points]
    return new_center

class KCentersDiscretizer(BaseDiscretizer):
    """K-Centers clustering.


    Parameters
    ----------
    cutoff : float
        The maximum radius of a single cluster
    representative_sample_size : int, optional
        The maximum number of representatives to keep from each cluster

    Examples
    --------
    """

    def __init__(self, cutoff, representative_sample_size=10):
        super().__init__(representative_sample_size)
        self._centers = []
        self._nearest_neighbors = None
        self._k_centers_initiated = False
        self._cluster_inx_2_id = dict()
        self._id_2_cluster_inx = dict()

        self.cutoff = cutoff

    @property
    def n_states(self):
        return len(self._centers)

    @property
    def centers(self):
        return np.array(self._centers)

    def get_centers_by_ids(self, ids):
        indices = np.array([self._id_2_cluster_inx[cid] for cid in ids], dtype=int)
        return self.centers[indices]

    def _coarse_grain_states(self, data : np.ndarray):
        """Get the cluster ids of data points.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, n_features)
            The data points to cluster

        Returns
        -------
        labels : list of int
            A list of length n_samples, such that labels[i] is the label of the point data[i]
        """
        max_cluster_id = self.n_states-1
        if self._k_centers_initiated:
            self._nearest_neighbors, clusters, self._centers = extend_k_centers(data,
                                                                self._nearest_neighbors,
                                                                self._centers, self.cutoff)
        else:
            self._nearest_neighbors, clusters, self._centers = k_centers(data, cutoff=self.cutoff)
            self._k_centers_initiated = True

        if np.max(clusters) > max_cluster_id:
            new_clusters = np.unique(clusters)
            new_clusters = new_clusters[np.where(new_clusters > max_cluster_id)]
            self._add_clusters(new_clusters)

        return self._get_ids(clusters)

    def _get_ids(self, cluster_indices):
        if len(cluster_indices.shape) == 0:
            return [self._cluster_inx_2_id[int(cluster_indices)]]
        return [self._cluster_inx_2_id[inx] for inx in cluster_indices]

    def _add_clusters(self, new_clusters):
        for cluster in new_clusters:
            if self._cluster_inx_2_id.get(cluster):
                continue
            uid = get_unique_id()
            self._cluster_inx_2_id[cluster] = uid
            self._id_2_cluster_inx[uid] = cluster

