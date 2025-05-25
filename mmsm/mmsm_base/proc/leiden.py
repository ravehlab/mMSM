import numpy as np
import networkx as nx
import igraph
import leidenalg as la
from mmsm.external.msmtools.msmtools import timescales

__all__ = ['LeidenPartition']


class LeidenPartition:
    def __init__(self):
        pass

    def check_split_condition(self, vertex) -> bool:
        if vertex.n == 1:
            return False
        if vertex.is_root:
            return self._root_split_condition(vertex)
        if vertex.partition_changed:
            return True
        if self._get_graph_diameter(vertex) > vertex.config.partition_diameter_threshold:
            return True
        return False

    def get_metastable_partition(self, vertex):
        n = vertex.n
        T = vertex.T[:n, :n]  # reminder: this is T^tau
        # Tn = normalize_rows(T)
        return self.cluster_graph_from_matrix(T)

    def cluster_graph_from_matrix(self, T, max_comm_size=0):
        g = igraph.Graph.Adjacency((T > 0).tolist())

        # Add edge weights and node labels.
        g.es['weight'] = T[T.nonzero()]
        partition = la.find_partition(g, la.ModularityVertexPartition, weights='weight',
                                      n_iterations=2,
                                      max_comm_size=max_comm_size)
        cs = list(partition)
        return cs

    def _get_graph_diameter(self, vertex):
        n = vertex.n
        adj = (vertex._T[:n, :n] > 0).astype('int')
        gnx = nx.from_numpy_array(adj)
        if not nx.is_connected(gnx):
            return np.inf
        return nx.approximation.diameter(gnx)

    def _root_split_condition(self, vertex):
        ts = timescales(vertex.T + np.finfo('float').eps, 1)[1:]
        log_ts = np.log(ts)
        groups = group_timescales(log_ts, max_r=1)
        if len(np.unique(groups)) > 1:
            return True
            # cutoff = ts[np.argmax(groups >= np.max(groups)-1)]
            # if vertex.tree.total_sim_time_ns > cutoff*vertex.tree.base_timestep*vertex.tau*5*1e-4:
            #     return True
        return False


def group_timescales(data, dist=1, max_r=3):
    assignment = np.zeros(len(data))
    rs = [0]
    cur = 0
    last_pt = data[0]
    for i, pt in enumerate(data[1::], 1):
        d = np.abs(last_pt - pt)
        if d <= dist and rs[cur] <= max_r:
            assignment[i] = cur
            rs[cur] += d
        else:
            cur += 1
            assignment[i] = cur
            rs.append(0)
        last_pt = pt
    return assignment
