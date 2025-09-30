"""mMSMConfig"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>
# Modified by: Nir Nitskansky <nir.nitskansky@mail.huji.ac.il>

from dataclasses import dataclass, field


@dataclass
class mMSMConfig:
    n_trajectories: int = 10
    trajectory_len: int = 10**4
    base_tau: int = 1

    sampling_heuristics: list = field(default_factory=lambda: ["equilibrium", "exploration", "equilibrium_inv"])
    sampling_heuristic_weights: list = field(default_factory=lambda: [0.6, 0.2, 0.2])
    adaptive_sampling: bool = True

    vertex_sampler_kwargs: dict = field(default_factory=dict)
    vertex_sampler: str = 'auto'  # defaults to WeightedVertexSampler

    # Kinetics estimation parameters ###########
    count_stride: int = 1
    alpha: float = 1.  # Dirichlet prior
    transition_estimator: str='Dirichlet_MMSE'

    # Tree structure parameters #############
    parent_update_condition: str = 'auto'
    parent_update_threshold: float = 0.1
    partition_estimator: str = 'leiden'
    partition_kwargs: dict = field(default_factory=dict)

    partition_diameter_threshold: int = 4
    lag_time_ratio: int = 2
    random_split: float = 0.0
    max_height: int = -1


