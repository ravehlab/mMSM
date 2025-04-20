import numpy as np
from mmsm.mmsm_config import mMSMConfig

class WeightedVertexSampler:
    def __init__(self, heuristics, weights):
        self._weights = weights
        self._heuristics = heuristics

    def __call__(self, vertex, n_samples):
        return np.random.choice(vertex.children,
                                size=n_samples,
                                p=self._get_distribution(vertex) )

    def _get_distribution(self, vertex):
        distributions = np.array([heuristic(vertex) for heuristic in self._heuristics])
        return self._weights.dot(distributions)

    def _get_full_distribution(self, vertex):
        local_distribution = self._get_distribution(vertex)
        # base case:
        if vertex.height == 1:
            return dict(zip(vertex.children, local_distribution))

        # recursive construction:
        global_distribution = {}
        for i, child_id in enumerate(vertex.children):
            child = vertex.tree.vertices[child_id]
            child_distribution = self._get_full_distribution(child)
            for microstate_id, local_probability in child_distribution.items():
                global_distribution[microstate_id] = (local_probability*local_distribution[i])
        return global_distribution


def _get_sampler_by_name(name):
    samplers = {'auto': exploration,
                'exploration': exploration,
                'uniform': uniform_sample,
                'equilibrium': equilibrium,
                'equilibrium_inv': equilibrium_inv,
                'min_prob': min_prob,}
    if name not in samplers:
        raise NotImplementedError(f"Optimizer {name} not implemented.")
    return samplers[name]


def get_vertex_sampler(config:mMSMConfig=None, sampling_heuristics=None, weights=None):
    if config is not None:
        sampling_heuristics = config.sampling_heuristics
        weights = config.sampling_heuristic_weights
    heuristics = [_get_sampler_by_name(h) for h in sampling_heuristics]
    weights = np.array(weights)
    return WeightedVertexSampler(heuristics, weights)
    # return MicrostateSampler(heuristics, weights)


def equilibrium(vertex):
    ids = {vid: ind for ind, vid in enumerate(vertex.local_stationary_ids)}
    return np.array([vertex.local_stationary[ids[c]] for c in vertex.children])

def equilibrium_inv(vertex):
    local_sd = np.array(equilibrium(vertex))
    return np.exp(-local_sd) / np.exp(-local_sd).sum()

def min_prob(vertex):
    expl = exploration(vertex)
    ls = np.zeros_like(expl)
    ls[np.argmin(expl)] = 1
    return ls


def uniform_sample(vertex):
    """Samples one of this vertex's children uniformly.
    """
    return np.ones(vertex.n)/vertex.n


def exploration(vertex):
    children = vertex.children
    eps = 1e-6
    p = np.ndarray(vertex.n)
    for i, child in enumerate(children):
        p[i] = max(np.exp(-vertex.tree.get_n_samples(child)/1e3), eps)
    p = p/np.sum(p)
    return np.nan_to_num(p)
