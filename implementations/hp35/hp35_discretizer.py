import numpy as np
import numba
from mmsm.mmsm_base.base_discretizer import BaseDiscretizer
from mmsm.mmsm_base.proc.kcenters import KCentersDiscretizer
from scipy.spatial import cKDTree
from mmsm.mmsm_base.util import get_unique_id


PATH_MEANS = "./runfiles/means.npy"
PATH_STDS = "./runfiles/stds.npy"
PATH_PCS = "./runfiles/pcs10.npy"


@numba.njit
def _farthest_first_indices(data, cutoff):
    """
    Compute farthest-first indices up to the cutoff.
    Returns an array of center indices.
    """
    N, d = data.shape
    dist2 = np.empty(N, np.float64)
    center_idxs = np.empty(N, np.int64)
    # pick the first point deterministically (index 0)
    c0 = 0
    center_idxs[0] = c0
    for i in range(N):
        tmp = 0.0
        for j in range(d):
            diff = data[i, j] - data[c0, j]
            tmp += diff * diff
        dist2[i] = tmp
    count = 1
    cutoff2 = cutoff * cutoff
    while True:
        # find farthest point
        max_idx = 0
        max_val = dist2[0]
        for i in range(1, N):
            if dist2[i] > max_val:
                max_val = dist2[i]
                max_idx = i
        if max_val <= cutoff2:
            break
        center_idxs[count] = max_idx
        # update distances
        for i in range(N):
            tmp = 0.0
            for j in range(d):
                diff = data[i, j] - data[max_idx, j]
                tmp += diff * diff
            if tmp < dist2[i]:
                dist2[i] = tmp
        count += 1
    return center_idxs[:count]

class FastKCentersDiscretizer(BaseDiscretizer):
    """
    Efficient farthest-first (k-centers) discretizer with adaptive extension.
    """
    def __init__(self, cutoff, representative_sample_size=10):
        super().__init__(representative_sample_size)
        self.cutoff = cutoff
        self._centers = None
        self._tree = None
        self._k_centers_initiated = False
        self._cluster_inx_2_id = {}
        self._id_2_cluster_inx = {}

    @property
    def n_states(self):
        return len(self._centers) if self._centers is not None else 0

    @property
    def centers(self):
        return np.array(self._centers)

    def get_centers_by_ids(self, ids):
        idxs = np.array([self._id_2_cluster_inx[c] for c in ids], dtype=int)
        return self.centers[idxs]

    def _coarse_grain_states(self, data: np.ndarray):
        X = np.asarray(data)
        if not self._k_centers_initiated:
            idxs = _farthest_first_indices(X, self.cutoff)
            self._centers = [X[i] for i in idxs]
            self._tree = cKDTree(self._centers)
            dists, clusters = self._tree.query(X)
            self._k_centers_initiated = True
        else:
            dists, clusters = self._tree.query(X)
            while dists.max() > self.cutoff:
                idx = int(np.argmax(dists))
                self._centers.append(X[idx])
                self._tree = cKDTree(self._centers)
                dists, clusters = self._tree.query(X)

        # assign unique IDs to new clusters
        current_max = len(self._cluster_inx_2_id) - 1
        if clusters.max() > current_max:
            new_idxs = np.unique(clusters)
            new_idxs = new_idxs[new_idxs > current_max]
            for ci in new_idxs:
                uid = get_unique_id()
                self._cluster_inx_2_id[int(ci)] = uid
                self._id_2_cluster_inx[uid] = int(ci)

        return [self._cluster_inx_2_id[int(c)] for c in clusters]

    def remove_centers_by_id(self, uids):
        # 1) validate and collect internal indices
        cidxs = []
        for uid in uids:
            if uid not in self._id_2_cluster_inx:
                raise KeyError(f"Unknown center ID {uid}")
            cidxs.append(self._id_2_cluster_inx[uid])
        cidxs = sorted(set(cidxs), reverse=True)

        # 2) remove from centers list and mappings
        for cidx in cidxs:
            uid = self._cluster_inx_2_id.pop(cidx)
            del self._id_2_cluster_inx[uid]
            self._centers.pop(cidx)

        # 3) rebuild the KD-tree
        self._tree = cKDTree(self._centers)

        # 4) re-index the remaining clusters
        new_ci2id = {}
        new_id2ci = {}
        for new_idx, uid in enumerate(self._cluster_inx_2_id.values()):
            new_ci2id[new_idx] = uid
            new_id2ci[uid] = new_idx
        self._cluster_inx_2_id = new_ci2id
        self._id_2_cluster_inx = new_id2ci

class HP35DiscretizerPreexisting(BaseDiscretizer):
    def __init__(self, cutoff, representative_sample_size=10, pca=True):
        super().__init__(representative_sample_size)
        self._kcenters = FastKCentersDiscretizer(cutoff, representative_sample_size)
        self.means = np.load(PATH_MEANS)
        self.stds = np.load(PATH_STDS)
        self.pc = np.load(PATH_PCS)
        self.pca = pca

    @property
    def n_states(self):
        return self._kcenters.n_states

    def _coarse_grain_states(self, data):
        data_std = (data - self.means) / self.stds
        if self.pca:
            dat_prj = data_std @ self.pc.T
        else:
            dat_prj = data_std
        return self._kcenters._coarse_grain_states(dat_prj)

    def get_centers_by_ids(self, cluster_ids):
        return self._kcenters.get_centers_by_ids(cluster_ids)
