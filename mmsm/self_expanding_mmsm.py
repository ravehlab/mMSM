"""Self Expanding Multiscale MSM base class"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>
# Modified by: Nir Nitskansky <nir.nitskansky@mail.huji.ac.il>

import warnings
import numpy as np
from mmsm.mmsm_config import mMSMConfig
from mmsm.mmsm_base.base_trajectory_sampler import BaseTrajectorySampler
from mmsm.mmsm_base.base_discretizer import BaseDiscretizer
from mmsm.mmsm_base.mmsm_tree import MultiscaleMSMTree
from mmsm.mmsm_base.util import get_threshold_check_function


import time
from contextlib import contextmanager

class SelfExpandingMultiscaleMSM:
    def __init__(self, sampler:BaseTrajectorySampler, discretizer:BaseDiscretizer, x_init,
                 config:mMSMConfig=None, **config_kwargs):
        self._sampler = sampler
        self._discretizer = discretizer
        if config is None:
            self.config = mMSMConfig(**config_kwargs)
        else:
            self.config = config
        self.x_init = np.copy(x_init)
        self._tau0 = self._sampler.timestep * self.config.base_tau * self.config.count_stride
        self._mmsm_tree: MultiscaleMSMTree = self._init_tree()
        self._n_samples = 0

    def _init_tree(self):
        tree = MultiscaleMSMTree(self.config)
        return tree

    @property
    def batch_size(self):
        return self.config.n_trajectories * self.config.trajectory_len

    @property
    def timestep(self):
        return self._tau0

    @property
    def tree(self):
        return self._mmsm_tree

    @property
    def discretizer(self):
        return self._discretizer

    def get_timescale(self):
        return self._mmsm_tree.get_longest_timescale() * self.timestep

    def _discretize_trajectories(self, trajs):
        return [self.discretizer.get_coarse_grained_states(traj) for traj in trajs]

    def _get_dtraj(self, initial_microstates, num_trajs, traj_len, step_size, configurations=False):
        if configurations or self.config.adaptive_sampling is False:
            initial_points = initial_microstates
        else:
            initial_points = [self.discretizer.sample_from(ms) for ms in initial_microstates]


        dtrajs = self._sampler.sample_from_states(initial_points, traj_len, num_trajs, step_size)

        dtrajs = self._discretize_trajectories(dtrajs)

        if not configurations and self.config.adaptive_sampling:
            for i in range(len(initial_microstates)):
                for j in range(num_trajs):
                    if dtrajs[num_trajs * i + j][0] != initial_microstates[i]:
                        dtrajs[num_trajs * i + j][0] = initial_microstates[i]
        return dtrajs

    def expand(self, max_cputime=np.inf, max_samples=np.inf, max_batches=np.inf):

        if max_cputime == max_samples == max_batches == np.inf:
            warnings.warn("At least one of the parameters max_cputime, max_samples, or \
                              min_timescale_sec must be given")
            return

        max_values = {'n_samples': max_samples, 'n_batches': max_batches}
        stop_condition = get_threshold_check_function(max_values, max_time=max_cputime)
        n_samples = 0
        n_batches = 0

        while not stop_condition(n_samples=n_samples, n_batches=n_batches):
            if self._n_samples == 0:
                start_states = [np.copy(self.x_init) for _ in range(self.config.n_trajectories)]
                trajs = self._get_dtraj(start_states, 1, self.config.trajectory_len, self.config.base_tau,
                                        configurations=True)
            else:
                if self.config.adaptive_sampling:
                    start_states = self.tree.sample_states_mid(self.config.n_trajectories)
                else:
                    start_states = []
                trajs = self._get_dtraj(start_states, 1, self.config.trajectory_len, self.config.base_tau)

            self._mmsm_tree.update_model_from_trajectories(trajs)
            self._mmsm_tree.do_all_updates_by_height()

            n_samples += self.batch_size
            n_batches += 1
            self._n_samples += self.batch_size
