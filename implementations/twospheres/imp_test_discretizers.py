import numpy as np
from mmsm.mmsm_base.base_discretizer import BaseDiscretizer
from mmsm.mmsm_base.proc.kcenters import KCentersDiscretizer
from scipy.cluster import vq


class DistanceDiscretizer(BaseDiscretizer):
    """A simple wrapper for KCentersDiscretizer, made for the 2-particle
    IMP model. Input data is converted to a single feature of pair distance."""
    def __init__(self, cutoff, representative_sample_size=10):
        super().__init__(representative_sample_size)
        self._kcenters = KCentersDiscretizer(cutoff, representative_sample_size)

    @property
    def n_states(self):
        return self._kcenters.n_states

    def _coarse_grain_states(self, data):
        distances = dist_np(data)
        return self._kcenters._coarse_grain_states(distances[:, None])

    def get_centers_by_ids(self, cluster_ids):
        return self._kcenters.get_centers_by_ids(cluster_ids)


def dist_np(traj):
    return np.sum(np.diff(traj, axis=1) ** 2, axis=2)[:, 0] ** 0.5
