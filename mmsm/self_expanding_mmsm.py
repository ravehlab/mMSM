"""Self Expanding Multiscale MSM base class"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>
# Modified by: Nir Nitskansky <nir.nitskansky@mail.huji.ac.il>

from abc import ABC

import warnings
import numpy as np
from mmsm.mmsm_config import mMSMConfig
from mmsm.mmsm_base.base_trajectory_sampler import BaseTrajectorySampler
from mmsm.mmsm_base.base_discretizer import BaseDiscretizer
from mmsm.mmsm_base.mmsm_tree import MultiscaleMSMTree
from mmsm.mmsm_base.util import get_threshold_check_function

class SelfExpandingMultiscaleMSM(ABC):

    def __init__(self, sampler:BaseTrajectorySampler, discretizer:BaseDiscretizer,
                 config:mMSMConfig=None, **config_kwargs):
        self._sampler = sampler
        self._discretizer = discretizer
        if config is None:
            self.config = mMSMConfig(**config_kwargs)
        else:
            self.config = config
        self._effective_timestep = self._sampler.timestep_size * self.config.base_tau
        self._hmsm_tree: MultiscaleMSMTree = self._init_tree()
        self._n_samples = 0
        self._init_sample()

    def _init_tree(self):
        tree = MultiscaleMSMTree(self.config, self._effective_timestep)
        return tree

    def _init_sample(self):
        dtrajs = self._sampler.get_initial_sample(self.config.trajectory_len,
                                                  self.config.n_trajectories,
                                                  self.config.base_tau)
        dtrajs = self._discretize_trajectories(dtrajs)
        self._hmsm_tree.update_model_from_trajectories(dtrajs)
        self._hmsm_tree.do_all_updates_by_height()
        self._n_samples += self.batch_size

    @property
    def batch_size(self):
        return self.config.n_trajectories * self.config.trajectory_len

    @property
    def timestep(self):
        return self._effective_timestep

    @property
    def total_simulation_time(self):
        return self._n_samples * self._effective_timestep

    @property
    def tree(self):
        return self._hmsm_tree

    @property
    def discretizer(self):
        return self._discretizer

    def get_timescale(self):
        return self._hmsm_tree.get_longest_timescale() * self.timestep

    def _discretize_trajectories(self, trajs):
        return [self.discretizer.get_coarse_grained_states(traj) for traj in trajs]

    def _get_dtraj(self, initial_microstates, num_trajs, traj_len, step_size):
        initial_points = [self.discretizer.sample_from(ms) for ms in initial_microstates]
        # initial_points = self._get_reps_from_microstates(initial_microstates)
        dtrajs = self._sampler.sample_from_states(initial_points, traj_len, num_trajs, step_size)
        dtrajs = self._discretize_trajectories(dtrajs)

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


        # Main loop: 1) sample from equilibrium 2) adaptive sampling for refinement 3) update; repeat
        while not stop_condition(n_samples=n_samples, n_batches=n_batches):
            start_states = self.tree.sample_states_mid(self.config.n_trajectories)
            # start_states = self.tree.sample_states(self.config.n_trajectories)
            trajs = self._get_dtraj(start_states, 1, self.config.trajectory_len, self.config.base_tau)
            self._hmsm_tree.update_model_from_trajectories(trajs)
            self._hmsm_tree.do_all_updates_by_height()

            n_samples += self.batch_size
            n_batches += 1
            self._n_samples += self.batch_size
