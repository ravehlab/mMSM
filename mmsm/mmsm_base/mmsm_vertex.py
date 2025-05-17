"""HierarchicalMSMVertex"""
# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>
# Modified by: Nir Nitskansky <nir.nitskansky@mail.huji.ac.il>

import numpy as np
from msmtools.analysis import stationary_distribution
from mmsm.mmsm_base.util import get_parent_update_condition, get_unique_id
from mmsm.mmsm_config import mMSMConfig


class MultiscaleMSMVertex:
    def __init__(self, tree, children, parent, tau, height, partition_estimator, vertex_sampler, config: mMSMConfig):
        self.__id = get_unique_id()
        self.tree = tree
        self._children = set(children)
        if parent is None:
            self.parent = self.__id
        else:
            self.parent = parent
        self.tau = tau
        self.height = height
        self._partition_estimator = partition_estimator
        self._vertex_sampler = vertex_sampler
        self._neighbors = []
        self._parent_update_condition = get_parent_update_condition(config.parent_update_condition,
                                                                    config.parent_update_threshold)
        self.config = config
        self.index_2_id = None
        self.id_2_index = None
        self.partition_changed = False


    @property
    def children(self):
        return list(self._children)

    @property
    def n(self):
        return len(self._children)

    @property
    def id(self):
        return self.__id

    @property
    def T(self):
        return self._T_tau

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent_id):
        self._parent = parent_id

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def is_root(self):
        return self.parent == self.id

    @property
    def local_stationary(self):
        return self._local_stationary

    @property
    def local_stationary_ids(self):
        return self._local_stationary_ids

    def set_parent(self, parent):
        self.parent = parent

    def add_children(self, children_ids):
        """
        Add children to this tree.
        NOTE: this should *only* be called by HierarchicalMSMTree as this does NOT do anything to
        maintain a valid tree structure - only changes local variables of this vertex
        """
        self._children.update(children_ids)

    def remove_children(self, children_ids):
        """
        NOTE: this should *only* be called by HierarchicalMSMTree as this does NOT do anything to
        maintain a valid tree structure - only changes local variables of this vertex
        """
        self._children -= set(children_ids)

    def update(self):
        self.n_samples = np.sum([self.tree.get_n_samples(child) for child in self.children])
        return self._update_T()

    def get_external_T(self, tau=1) -> tuple:
        """get_external_T.
        Get the transition probabilities between this vertex and its neighbors at the same level
        of the tree.

        Parameters
        ----------
        tau : int
            the time-resolution of the transition probabilities
        Returns
        -------
        ids : ndarray
            An array of the ids of the neighbors of this vertex
        transition_probabilities : ndarray
            An array of the transition probabilities to the neighbors of this vertex, such that
            transition_probabilities[i] is the probability of getting to ids[i] in tau steps from
            this vertex.
        """
        ids, external_T = self._external_T
        ids = ids.copy()
        external_T = external_T.copy()
        if tau != 1:
            external_T[0] = np.power(external_T[0], tau) # The probability of no transition
            # The relative probabilities of the other transitions haven't changed, so normalize:
            external_T[1:] = (external_T[1:]/np.sum(external_T[:,1])) * (1-external_T[0])
        return ids.copy(), external_T.copy()

    def sample_microstate(self, n_samples):
        """Get a set of microstate from this MSM, ideally chosen such that sampling a random walk
        from these microstates is expected to increase some objective function.

        Returns:
        -------
        samples : list
            A list of ids of microstates.
        """
        # a sample of n_samples vertices from this msm:
        sample = list(self._vertex_sampler(self, n_samples))
        if self.height == 1:
            return sample

        # now for each of those samples, get a sample from that vertex, the number of times
        # it appeared in sample
        vertices, counts = np.unique(sample, return_counts=True)
        recursive_sample = []
        for i, vertex in enumerate(vertices):
            recursive_sample += self.tree.sample_states(counts[i], vertex)
        return recursive_sample

    def sample_from_stationary(self):
        return np.random.choice(self.children, p=self.local_stationary)

    def get_all_microstates(self):
        """get_all_microstates.

        Returns
        -------
        microstates : set
            A set of all the ids of the microstates at the leaves of the subtree of which this
            vertex is the root.
        """
        if self.height == 1:
            return self.children
        microstates = set()
        for child in self.children:
            microstates.update(self.tree.get_microstates(child))
        return microstates

    def _check_parent_update_condition(self):
        if self.is_root:
            return False
        return True
        # return self._parent_update_condition(self)

    def _update_T(self):
        assert self.n > 0
        T_rows = dict()
        column_ids = set(self.children)

        # Get the rows of T corresponding to each child
        for child in self._children:
            ids, row = self.tree.get_external_T(child)
            T_rows[child] = (ids, row)
            column_ids = column_ids.union(ids)

        # Get a mapping between indices of the matrix T and the vertex_ids they represent
        id_2_index, index_2_id, full_n = self._get_id_2_index_map(column_ids)
        self.index_2_id = index_2_id
        self.id_2_index = id_2_index

        # Initialize the transition matrix
        self._T = np.zeros((full_n, full_n))
        n_external = full_n - self.n  # number of external vertices

        # Now fill the matrix self._T
        for child, (ids, T_row) in T_rows.items():
            row = id_2_index[child]
            for id, transition_probability in zip(ids, T_row):
                column = id_2_index[id]
                self._T[row, column] += transition_probability

        for i in range(self.n, full_n):
            ids, row = self.tree.get_external_T(index_2_id[i])
            for dst_i, dst in enumerate(ids):
                if dst in self.id_2_index:
                    self._T[i, id_2_index[dst]] += row[dst_i]
                else:
                    self._T[i, i] += row[dst_i]

        # get the transition matrix in timestep resolution self.tau
        self._T_tau = np.linalg.matrix_power(self._T, self.tau)
        self._set_local_stationary_distribution()
        self._T_is_updated = True
        self._update_external_T()

    def _get_id_2_index_map(self, column_ids):
        """Return a dict that maps each id to the index of its row/column in T, and the
        dimension of the full transition matrix.
        """
        id_2_parent_id = {}
        id_2_index = {}
        index_2_id = {}
        neighbors = set()
        n = self.n
        # get the parents of vertices in column_ids which aren't me (my neighbors)
        for column_id in column_ids:
            if column_id in self.children:
                parent = self.id
            else:
                parent = self.tree.get_parent(column_id)
                id_2_parent_id[column_id] = parent
                if parent not in column_ids:
                    neighbors.add(parent)

        # my own children are the first n columns and rows:
        for j, id in enumerate(self.children):
            id_2_index[id] = j
            index_2_id[j] = id
        # the last columns and rows are my neighbors:
        # for j, id in enumerate(neighbors):
        for j, id in enumerate(id_2_parent_id):
            id_2_index[id] = j + n
            index_2_id[j+n] = id
        full_n = len(index_2_id)
        return id_2_index, index_2_id, full_n

    def _set_local_stationary_distribution(self):
        """update self._local_stationary and self._local_stationary_ids"""
        n = self.n
        sd = stationary_distribution(self._T)

        # Check for numerical errors, recalculate if necessary
        if np.all(sd[:n] == 0):
            reg = np.zeros_like(sd)
            reg[n:] += 1e-6
            T_reg = self._T + reg[:, None]
            T_reg /= T_reg.sum(axis=1)[:, None]
            sd = stationary_distribution(T_reg)

        self._local_stationary = sd[:n] / sd[:n].sum()
        self._local_stationary_ids = [self.index_2_id[i] for i in range(n)]

    def _update_external_T(self):
        T = self._T  # this is T(1) as opposed to self.T which is T(tau)
        n = self.n
        local_stationary = self.local_stationary
        local_stationary = np.resize(local_stationary, T.shape[0])
        local_stationary[n:] = 0  # pad with 0's.

        full_transition_probabilities = local_stationary.dot(T)
        parents = np.array([self.tree.get_parent(v) for v in self.id_2_index])

        ids_to_ext_index = {self.id: 0}
        ids_to_ext_index.update(({nid: i for i, nid in enumerate(np.unique(parents[parents != self.id]), 1)}))

        external_T = np.zeros(len(ids_to_ext_index))
        ids = [0 for _ in range(len(ids_to_ext_index))]
        for p, i in ids_to_ext_index.items():
            external_T[i] = full_transition_probabilities[parents == p].sum()
            ids[i] = p
        ids = np.array(ids)
        self._external_T = ids, external_T
        self._neighbors = ids

    def _check_split_condition(self):
        if self.config.max_height != -1:
            if self.is_root and self.tree.height >= self.config.max_height:
                return False
        return self._partition_estimator.check_split_condition(self)

    def _get_partition(self):
        # returns (partition, taus) where partition is an iterable of iterables of children_ids
        return self._partition_estimator.get_metastable_partition(self)

    def _split(self):
        index_partition = self._get_partition()
        id_partition = [[self.index_2_id[index] for index in subset] for subset in index_partition]
        return id_partition, self, self.parent
