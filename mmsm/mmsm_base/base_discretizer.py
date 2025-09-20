"""Discretizer base class"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from abc import ABC, abstractmethod
import numpy as np
from mmsm.mmsm_base.util import count_dict

class BaseDiscretizer(ABC):
    """BaseDiscretizer.
    Base class for coarse graining of continuous spaces into discrete states.
    """
    def __init__(self, representative_sample_size=10):
        self._representative_sample_size = representative_sample_size
        self._representatives = dict()
        self._cluster_count = count_dict()


    def get_coarse_grained_states(self, data : np.ndarray):
        """Get the state ids of data points.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            The data points to cluster into discrete states.

        Returns
        -------
        labels : list of int
            A list of length n_samples, such that labels[i] is the label of the point data[i]
        """
        clusters = self._coarse_grain_states(data)
        self._sample_representatives(clusters, data)
        return clusters
        

    def sample_from(self, cluster_id):
        """Get a sample representative from a given cluster.
        The sample returned will be one of the points previously labeled as cluster_id, such that
        the probability of each point that was seen from this cluster_id to be sampled is 1/n,
        where n is the number of points from this cluster that were observed.

        Parameters
        ----------
        cluster_id : int
            The id of the cluster to sample from

        Returns
        -------
        x : np.ndarray of shape (n_features)
            The sampled point

        Notes
        -----
        While the distribution of each individual sample is uniform over all seen samples apriori,
        the distribution of more than one sample is not uniform. This is because a finite set of
        representatives is kept from each cluster.
        The greater the parameter representative_sample_size is, the closer this distribution is
        to uniform.
        """
        random_index = np.random.randint(len(self._representatives[cluster_id]))
        return self._representatives[cluster_id][random_index].copy()

    def _sample_representatives (self, cluster_ids, data):
        """
        Keep representative samples for each cluster, such that the probability of a representative
        x from cluster j being sampled (when sampling from the representatives) is 1/n, where n is
        the number of points x' that have been observed in j.

        Parameters
        ----------
        cluster_ids : Iterable[int]
            The cluster ids of the data points
        data : np.ndarray of shape (n_samples, n_features)
            The clustered data points
        """
        r = self._representative_sample_size
        for i, cid in enumerate(cluster_ids):
            self._cluster_count[cid] += 1
            n = self._cluster_count[cid]
            reps = self._representatives.setdefault(cid, [])
            if len(reps) < r:
                reps.append(data[i].copy())
            else:
                j = np.random.randint(n)
                if j < r:
                    reps[j] = data[i].copy()

    @property
    @abstractmethod
    def n_states(self):
        pass

    @abstractmethod
    def _coarse_grain_states(self, data : np.ndarray):
        """_coarse_grain_states.

        Parameters
        ----------
        data : np.ndarray((N,d))
            a set of N datapoints with dimension d

        Returns
        -------
        labels : list of int
            A list of length n_samples, such that labels[i] is the label of the point data[i]
        """
        pass

    @abstractmethod
    def get_centers_by_ids(self, cluster_ids):
        """Get cluster centers of the given cluster ids.
        """
        pass
