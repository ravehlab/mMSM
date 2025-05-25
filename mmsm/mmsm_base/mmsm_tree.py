"""MultiscaleMSMTree"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>
# Modified by: Nir Nitskansky <nir.nitskansky@mail.huji.ac.il>

import numpy as np
import numba
from mmsm.mmsm_config import mMSMConfig
from mmsm.mmsm_base.util import UniquePriorityQueue, count_dict
from mmsm.mmsm_base.mmsm_vertex import MultiscaleMSMVertex
from mmsm.external.msmtools.msmtools import stationary_distribution
from collections import defaultdict
from mmsm.mmsm_base.proc.leiden import LeidenPartition
from mmsm.mmsm_base.proc.vertex_samplers import get_vertex_sampler
from mmsm.mmsm_base.proc.dirichlet_mmse import dirichlet_MMSE

@numba.njit
def mod_gains_njit(T, ids, vertex_tau, cur_assignment, child_indices, neighbors, test_assignments):
    T = np.linalg.matrix_power(T, vertex_tau)
    dmod = T - T.sum(axis=0) / len(ids)
    np.fill_diagonal(dmod, 0)
    gains = [dmod[child_indices][:, cur_assignment == assignment].sum() +
             dmod[cur_assignment == assignment][:, child_indices].sum()
             for assignment in test_assignments]
    return gains

class MultiscaleMSMTree:
    UPDATE_T = 0
    REFINE = 1

    def __init__(self, config:mMSMConfig, timestep):
        self.config = config
        self._partition_estimator = LeidenPartition()
        self._vertex_sampler = get_vertex_sampler(config)
        self._microstate_parents = dict()
        self._microstate_counts = count_dict(depth=2)
        # The above is a (sparse) count matrix, which may be weighted. visit_count is the actual number of visits.
        self._microstate_visit_counts = count_dict(depth=2)
        self._microstate_transitions = dict()
        self._last_update_sent = dict()
        self._update_queue = UniquePriorityQueue()
        self._levels = defaultdict(list)
        self.vertices = dict()
        self._init_root()
        self.base_timestep = timestep
        self.total_sim_time_ns = 0

    def _assert_valid_vertex(self, vertex_id):
        return self._is_microstate(vertex_id) or \
               (self.vertices.get(vertex_id) is not None)

    def _is_root(self, vertex_id):
        return vertex_id == self.root

    def _is_microstate(self, vertex_id):
        if self._microstate_parents.get(vertex_id) is None:
            return False
        return True

    def _init_root(self):
        root = MultiscaleMSMVertex(self, children=set(),
                                           parent=None,
                                           tau=1,
                                           height=1,
                                           partition_estimator=self._partition_estimator,
                                           vertex_sampler=self._vertex_sampler,
                                           config=self.config)
        self._add_vertex(root)
        self._root = root.id

    @property
    def alpha(self):
        return self.config.alpha

    @property
    def root(self):
        return self._root

    @property
    def height(self):
        """
        Length of the longest path from the root of the tree to another vertex.
        """
        return self.vertices[self.root].height

    def get_level(self, level):
        assert level <= self.height
        if level==0:
            return list(self._microstate_parents.keys())
        return self._levels[level].copy()

    def get_parent(self, child_id):
        """
        Get the id of a vertex's parent vertex
        """
        if self._is_microstate(child_id):
            return self._microstate_parents[child_id]
        return self.vertices[child_id].parent

    def get_external_T(self, vertex_id, tau=1):
        """get_external_T.
        Get the transition probabilities between a vertex and other vertices on the same level
        of the tree.
        """
        if self._is_microstate(vertex_id):
            assert tau==1
            ids, ext_T = self._microstate_transitions[vertex_id]
            self._last_update_sent[vertex_id] = [ids, ext_T]
            return ids, ext_T
        return self.vertices[vertex_id].get_external_T(tau)

    def get_n_samples(self, vertex):
        """get_n_samples.
        Get the total number of times this vertex has appeared in trajectories used to estimate
        this HMSM.
        """
        if self._is_microstate(vertex):
            return max(1, sum(self._microstate_visit_counts[vertex].values()))
        return self.vertices[vertex].n_samples

    def get_microstates(self, vertex=None):
        """get_microstates.
        Get the ids of all the microstates in this tree.
        """
        if vertex is None:
            vertex = self.root
        return self.vertices[vertex].get_all_microstates()

    def get_longest_timescale(self):
        """get_longest_timescale.
        Get the timescale associated with the slowest process represented by the HMSM.
        """
        return self.vertices[self.root].timescale

    def update_model_from_trajectories(self, dtrajs):
        if len(dtrajs) == 0:
            return
        updated_microstates, parents_2_new_microstates = self._count_transitions(dtrajs)
        # add newly discovered microstates to their parents
        for parent, children in parents_2_new_microstates.items():
            self.vertices[parent].add_children(children)

        # update transition probabilities from observed transitions
        if self.config.transition_estimator == 'Dirichlet_MMSE':
            for vertex_id in updated_microstates:  # vertex here is actually a microstate?
                self._microstate_transitions[vertex_id] = dirichlet_MMSE(self._microstate_counts[vertex_id], self.alpha)
        else:
            raise ValueError("Unknown transition estimator specified.")

        # update the vertices of the tree from the leaves upwards
        for vertex_id in updated_microstates:
            if True:
                parent_id = self._microstate_parents[vertex_id]
                self._update_vertex(parent_id)
        self.total_sim_time_ns += np.sum([len(traj) for traj in dtrajs]) * self.base_timestep * 1e-6


    def force_update_all(self):
        """force_update_all.
        Update all vertices in the tree, regardless of the last time they were updated.
        """
        for vertex in self.vertices:
            self._update_vertex(vertex)
        self.do_all_updates_by_height()

    def force_rebuild_tree(self):
        """force_rebuild_tree.
        Remove all existing vertices, and rebuild entire tree, keeping only microstates and all
        observed transitions between them.
        """
        del self._levels
        del self.vertices
        self._levels = defaultdict(list)
        self.vertices = dict()
        self._init_root()
        self._connect_to_new_parent(self._microstate_parents.keys(), self.root)
        self.force_update_all()

    def sample_full_T(self, nsamples):
        n = len(self._microstate_parents)
        T = np.zeros((nsamples,n,n))
        level = self.get_level(0)
        for i, src in enumerate(level):
            ids, row = self._microstate_transitions[src]

            neighbor_ids = self._microstate_counts[src].keys()
            neighbor_indices = [level.index(nid) for nid in neighbor_ids]
            counts = np.fromiter(self._microstate_counts[src].values(), dtype=float)
            T[:,i,neighbor_indices] = np.random.dirichlet(counts + self.alpha, nsamples)
        return T, level

    def get_full_T(self):
        """get_full_T.
        Get the full transition matrix between all microstates.
        """
        level = self.get_level(0)
        n = len(level)
        T = np.zeros((n,n))
        for i, src in enumerate(level):
            ids, row = self._microstate_transitions[src]
            for nid, transition_probability in zip(ids, row):
                assert transition_probability >= 0
                j = level.index(nid)
                T[i,j] = transition_probability

        return T, level

    def get_full_stationary_distribution(self):
        """get_full_stationary_distribution.
        Get the stationary distribution over all microstates.

        Returns
        -------
        pi : dict
            A dictionary mapping microstate ids to their stationary distribution.
        """
        T, level = self.get_full_T()
        st = stationary_distribution(T)
        pi = {}
        for index, nid in enumerate(level):
            pi[nid] = st[index]
        return pi

    def do_all_updates_by_height(self):
        refine_history = set()
        while self._update_queue.not_empty():
            # print(f"\rVertices: {len(self.vertices)} Updates: {updates_done} | {len(self._update_queue._queue.queue)}, Refines: {refines_done}/{len(refine_queue)}", end="")
            (height, task), vertex_id = self._update_queue.get()
            vertex: MultiscaleMSMVertex = self.vertices.get(vertex_id)
            if vertex is None:
                continue  # this vertex no longer exists
            if task == self.UPDATE_T:
                vertex.update()
                self._update_queue.put(item=((vertex.height, self.REFINE), vertex_id))
            elif task == self.REFINE:
                # self._debug_verify_tree(level=height)
                if (vertex.id, vertex.height) not in refine_history:
                    refine_history.add((vertex.id, vertex.height))
                    self._refine_partition(vertex)
        # print(f"Merges: {self.moves_done} / {self.moves_tried}")

    def _refine_partition(self, vertex: MultiscaleMSMVertex):
        if vertex._check_split_condition():
            partition, split_vertex, parent = vertex._split()
            self._update_split(partition, split_vertex, parent)
            self.vertices[parent].partition_changed = True
            return
        self.partition_changed = False
        if len(vertex.neighbors) > 2:
            if self._merge_vertex(vertex):
                return
        if vertex._check_parent_update_condition():
            self._update_vertex(vertex.parent)

    def _merge_vertex(self, vertex: MultiscaleMSMVertex):
        T, ids = self._get_T_multiple_vertices(vertex.neighbors)

        cur_assignment = np.array([self.get_parent(v) for v in ids])
        test_assignments = np.array([p for p in np.unique(cur_assignment) if p in vertex.neighbors])
        child_indices = np.isin(ids, vertex.children).nonzero()[0]

        gains = mod_gains_njit(T, ids, vertex.tau, cur_assignment, child_indices, vertex.neighbors, test_assignments)
        new_assignment = test_assignments[np.argmax(gains)]


        # Move all the vertex's children to new_assignment
        if new_assignment != vertex.id:
            self._move_children(vertex.id, {new_assignment: vertex.children})
            return True
        return False

    def _get_T_of_subtree(self, vertex: MultiscaleMSMVertex, depth=1):
        if vertex.height - depth < 0:
            raise ValueError("Attempted to get subtree at depth lower than vertex height.")

        participating_vertices = [vertex.id]
        for d in range(depth):
            participating_vertices = [c for v in participating_vertices for c in self.vertices[v].children]
        all_ext_Ts = [self.get_external_T(v) for v in participating_vertices]
        id_to_index = {vid: i for i, vid in enumerate(participating_vertices)}
        T = np.zeros((len(participating_vertices), len(participating_vertices)))
        for src_i, (ids, row) in enumerate(all_ext_Ts):
            for dst, prob in zip(ids, row):
                if dst in participating_vertices:
                    T[src_i, id_to_index[dst]] += prob

        return T, participating_vertices

    def _get_T_multiple_vertices(self, vertex_list):
        children_ext_Ts = {c: self.get_external_T(c) for v in vertex_list
                           for c in self.vertices[v].children}
        participating_vertices = np.array(list({v for ids, _ in children_ext_Ts.values() for v in ids}))
        id_to_index = {vid: i for i, vid in enumerate(participating_vertices)}
        T = np.zeros((len(participating_vertices), len(participating_vertices)))
        for src, (ids, row) in children_ext_Ts.items():
            T[id_to_index[src], [id_to_index[i] for i in ids]] = row
        non_children_ext_Ts = {c: self.get_external_T(c) for c in participating_vertices
                               if c not in children_ext_Ts}
        for src, (ids, row) in non_children_ext_Ts.items():
            for i, vertex_id in enumerate(ids):
                if vertex_id in id_to_index:
                    T[id_to_index[src], id_to_index[vertex_id]] += row[i]
                else:
                    T[id_to_index[src], id_to_index[src]] += row[i]
        return T, participating_vertices

    def get_ancestor(self, vertex, level):
        """
        Get the ancestor of a vertex in the tree, on a specified level.
        """
        # level should be > 0
        parent = self.get_parent(vertex) if self._is_microstate(vertex) else vertex
        while self.vertices[parent].height < level:
            parent = self.vertices[parent].parent
        return parent

    def get_level_T(self, level, tau):
        """get_level_T.
        Get the transition matrix between vertices on a single level, in a given lag time.
        """
        level = self.get_level(level)
        n = len(level)
        index_2_id = dict(zip(range(n), level))
        id_2_index = dict(zip(level, range(n)))
        T = np.zeros((n,n))
        for vertex in level:
            i = id_2_index[vertex]
            ids, transition_probabilities = self.get_external_T(vertex)
            for neighbor, transition_probability in zip(ids, transition_probabilities):
                j = id_2_index[neighbor]
                T[i,j] = transition_probability
        return np.linalg.matrix_power(T, tau), level

    def get_level_stationary_distribution(self, level, return_indices=False):
        """get_full_stationary_distribution.
        Get the stationary distribution over all microstates.

        Returns
        -------
        pi : dict
            A dictionary mapping microstate ids to their stationary distribution.
        """
        T, ids = self.get_level_T(level, 1)
        st = stationary_distribution(T)
        pi = {}
        for index, nid in enumerate(ids):
            pi[nid] = st[index]
        return pi

    def sample_from_stationary(self, vertex, level=0):
        """sample_from_stationary.
        Sample on of the descendents of vertex, from the stationary distribution of the vertex.

        Parameters
        ----------
        vertex : int
           id of the vertex from which a sample will be taken
        level : int
            the level of the tree from which to sample

        Returns
        -------
        sample : int
            the id of the sampled vertex
        """
        sample = vertex
        for _ in range(self.vertices[sample].height - level):
            sample = self.vertices[sample].sample_from_stationary()
        return sample

    def sample_states(self, n_samples, vertex_id=None):
        """sample_states.
        """
        if vertex_id is None:
            vertex_id = self.root
        return self.vertices[vertex_id].sample_microstate(n_samples)

    def sample_states_mid(self, n_samples, vertex_id=None, level=None, debug=False):
        n_microstates = len(self._microstate_counts)
        logbase = self.config.partition_diameter_threshold
        start_from_lvl = np.minimum(self.vertices[self.root].height,
                                    np.floor(np.emath.logn(logbase, n_microstates)))

        lvl_sd = self.get_level_stationary_distribution(start_from_lvl)

        # A hack to avoid redefining function in vertex_samplers
        dummy_vertex = MultiscaleMSMVertex(self, set(lvl_sd.keys()), -1, 1, self.height,
                                             partition_estimator=self._partition_estimator,
                                             vertex_sampler=self._vertex_sampler,
                                             config=self.config)
        dummy_vertex._local_stationary = list(lvl_sd.values())
        dummy_vertex._local_stationary_ids = list(lvl_sd.keys())
        dummy_vertex._T_is_updated = True

        start_vertex = self._vertex_sampler(dummy_vertex, n_samples)
        return [self.vertices[v].sample_microstate(1)[0] for v in start_vertex]

    def get_path_to_root(self, vertex_id):
        path = [vertex_id]
        if self._is_microstate(vertex_id):
            path.append(self._microstate_parents[vertex_id])
        while not self.vertices[path[-1]].is_root:
            path.append(self.get_parent(path[-1]))
        return path

    def _count_transitions(self, dtrajs):
        updated_microstates = set()
        parents_2_new_microstates = defaultdict(set)
        ms_weight = defaultdict(lambda: 1)
        for dtraj in dtrajs:
            if len(dtraj) == 0:
                continue
            updated_microstates.update(dtraj)
            src = dtraj[0]
            weight = ms_weight[src]

            # check if the first state is new, this should only happen at initialization, where
            # the microstate level is the top level MSM, sot the parent is always the root.
            if not self._is_microstate(src):
                assert self.height == 1
                self._microstate_parents[src] = self.root
                _=self._microstate_counts[src][src]
                _=self._microstate_visit_counts[src][src]
                parents_2_new_microstates[self.root].add(src)

            for i in range(1, len(dtraj)):
                dst = dtraj[i]
                # count the observed transition
                self._microstate_counts[src][dst] += weight
                self._microstate_visit_counts[src][dst] += 1
                # evaluate the reverse transition, so that it will be set to 0 if none have been
                # observed yet. This is so that the prior weight of this transition will be alpha
                _ = self._microstate_counts[dst][src]

                # assign newly discovered microstates to the MSM they were discovered from
                if self._microstate_parents.get(dst) is None:
                    parent = self._microstate_parents[src]
                    self._microstate_parents[dst] = parent
                    parents_2_new_microstates[parent].add(dst)
                src = dst
        return updated_microstates, parents_2_new_microstates

    # def _check_parent_update_condition(self, microstate):
    #     """Check if a microstates transition probabilities have changed enough to trigger
    #     its parent to update
    #     """
    #     # check if no updates have been made yet
    #     if self._last_update_sent.get(microstate) is None:
    #         return True
    #     # check if new transitions have been observed since last update
    #     if set(self._last_update_sent[microstate][0])!=set(self._microstate_transitions[microstate][0]):
    #         return True
    #
    #     max_change_factor = max_fractional_difference(self._last_update_sent[microstate], \
    #                                                        self._microstate_transitions[microstate])
    #     return max_change_factor >= self.config.parent_update_threshold

    def _connect_to_new_parent(self, children_ids, parent_id):
        """
        Set the parent of all children_ids to be parent_id, and add them to the parents
        children
        """
        for child_id in children_ids:
            vertex = self.vertices.get(child_id)
            if vertex is None: # this means the child is a microstate
                self._microstate_parents[child_id] = parent_id
            else:
                vertex.set_parent(parent_id)
        self.vertices[parent_id].add_children(children_ids)

    def _update_split(self, partition, split_vertex: MultiscaleMSMVertex, parent):
        if len(partition) == 1:  # this is a trivial partition
            self._update_vertex(parent)
            return
        new_vertices = []
        for i, subset in enumerate(partition):
            vertex = MultiscaleMSMVertex(self, subset, parent,
                                           split_vertex.tau, split_vertex.height,
                                           partition_estimator=self._partition_estimator,
                                           vertex_sampler=self._vertex_sampler,
                                           config=self.config)
            self._add_vertex(vertex)
            self._connect_to_new_parent(subset, vertex.id)
            new_vertices.append(vertex.id)
            self._update_vertex(vertex.id)

        if not split_vertex.is_root:
            self.vertices[parent].add_children(new_vertices)
            self._remove_vertex(split_vertex.id)
        else:
            # the root is going up a level, so its children will be the new vertices
            new_tau = split_vertex.tau * 2
            self._levels[self.height].remove(self.root)
            split_vertex.remove_children(split_vertex.children)
            self.vertices[parent].add_children(new_vertices)
            split_vertex.height += 1
            split_vertex.tau = new_tau
            self._add_vertex(split_vertex)
        self._update_vertex(parent)

    def _update_merge(self, s_merge, s_into):
        pass

    def _move_children(self, previous_parent_id, parent_2_children):
        previous_parent = self.vertices[previous_parent_id]
        # move all the children:
        for new_parent, children in parent_2_children.items():
            previous_parent.remove_children(children)
            self.vertices[new_parent].add_children(children)
            if previous_parent.height == 1: # the children are microstates
                for child in children:
                    self._microstate_parents[child] = new_parent
            else:
                for child in children:
                    self.vertices[child].set_parent(new_parent)
                    for neigh in self.vertices[child].neighbors:
                        self._update_vertex(neigh)
            self._update_vertex(new_parent)

        if previous_parent.n == 0: # this vertex is now empty, we want to delete it
            self._remove_vertex(previous_parent_id)
        else:
            for neigh in previous_parent.neighbors:
                self._update_vertex(neigh)
        # print('yo')

    def _update_vertex(self, vertex_id, height=None):
        if self._is_microstate(vertex_id):
            return
        if height is None:
            height = self.vertices[vertex_id].height
        self._update_queue.put(item=((height, self.UPDATE_T), vertex_id))

    def _add_vertex(self, vertex):
        """
        Add a vertex to the tree

        Parameters
        ----------
        vertex : HierarchicalMSMVertex
            the new vertex
        """
        assert isinstance(vertex, MultiscaleMSMVertex)
        assert vertex.tree is self
        self.vertices[vertex.id] = vertex
        self._levels[vertex.height].append(vertex.id)

    def _remove_vertex(self, vertex):
        parent = self.get_parent(vertex)
        height = self.vertices[vertex].height
        self._levels[height].remove(vertex)
        neighbors = [n for n in self.vertices[vertex].neighbors if n != vertex]
        del self.vertices[vertex]
        # self.vertex_history.add(vertex)
        self.vertices[parent].remove_children([vertex])
        # update the parent and neighbors of the removed vertex

        for neigh in neighbors:
            self._update_vertex(neigh, height=height)

        # If the parent is now empty, remove it too.
        if self.vertices[parent].n == 0:
            self._remove_vertex(parent)
        else:
            self._update_vertex(parent, height=self.vertices[parent].height)

        # assert vertex not in self._microstate_parents.values()
        # for v in self.vertices.values():
        #     assert v.parent is not vertex

    def _random_step(self, vertex_id):
        next_states, transition_probabilities = self.get_external_T(vertex_id)
        return np.random.choice(next_states, p=transition_probabilities)

    def __getstate__(self):
        self._update_queue = None
        ret = dict(self.__dict__)
        self._update_queue = UniquePriorityQueue()
        return ret

    def __setstate__(self, state):
        self.__dict__ = state
        self._update_queue = UniquePriorityQueue()
