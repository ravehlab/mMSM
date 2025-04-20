import numpy as np
import IMP.atom
import IMP.algebra
import IMP.core
from mmsm.mmsm_base.base_trajectory_sampler import BaseTrajectorySampler


class IMPBrownianDynamicsSampler(BaseTrajectorySampler):
    """Testing a trajectory sampler using IMP.
    REMINDER: trajectories are a list of (trajectory length), with each step
    being [# particles, 3]."""

    def __init__(self, model: IMP.Model, scoring_func: IMP.ScoringFunction, temperature=300, step_size_fs=1000.0,
                 init_state=None, warmup: int = None, *args, **kwargs):
        super().__init__()
        self.model = model
        self.scoring_func = scoring_func
        self.init_state = init_state
        self.warmup = warmup
        self.step_size_fs = step_size_fs
        self.temperature = temperature
        self.bd = None
        self._h_root = None
        self._all_xyzs = None
        self.init_bd()

        self.first_state = self.get_current_coordinates()

        # These are only used for serialization purposes
        self._last_time = 0
        self._last_state = None

    @property
    def timestep_size(self):
        return self.step_size_fs

    def init_bd(self, model=None, scoring_func=None, deserialized=False):
        """Sets up the Brownian dynamics object."""
        if model is not None:
            self.model = model
        if scoring_func is not None:
            self.scoring_func = scoring_func
        self.bd = IMP.atom.BrownianDynamics(self.model)
        self.bd.set_scoring_function(self.scoring_func)
        self.bd.set_maximum_time_step(self.step_size_fs)
        self.bd.set_temperature(self.temperature)
        self._h_root = IMP.atom.Hierarchy(self.model.get_particle(IMP.ParticleIndex(0)))
        self._all_xyzs = [IMP.core.XYZ(p) for p in IMP.atom.get_leaves(self._h_root)]

        if deserialized and self._last_state is not None:
            self.bd.set_current_time(self._last_time)
            self.set_all_coordinates(self._last_state)

    def get_initial_sample(self, sample_len, n_samples, sample_interval=1):
        trajs = [[] for _ in range(n_samples)]
        for i in range(n_samples):
            self.reset_coordinates()
            if self.warmup is not None:
                prev_time = self.bd.get_current_time()
                self.bd.optimize(self.warmup)
                self.bd.set_current_time(prev_time)
            for j in range(sample_len):
                self.bd.optimize(sample_interval)
                trajs[i].append(self.get_current_coordinates())
        return trajs

    def sample_from_states(self, states, sample_len, n_samples, sample_interval=1):
        """Sample trajectories from a given list of states."""
        trajs = []
        for s in states:
            for i in range(n_samples):
                temp_traj = []
                self.set_all_coordinates(s)
                for j in range(sample_len):
                    self.bd.optimize(sample_interval)
                    temp_traj.append(self.get_current_coordinates())
                trajs.append(temp_traj)
        return trajs

    def perturb_state(self, state, steps=5, step_mod=0.1):
        self.set_all_coordinates(state)
        self.bd.set_maximum_time_step(self.step_size_fs * step_mod)
        self.bd.optimize(steps)
        self.bd.set_maximum_time_step(self.step_size_fs)
        return self.get_current_coordinates()

    def sample_from_current_state(self, sample_len, n_samples, sample_interval=1):
        """Sample trajectories from the current configuration of the system."""
        start_point = self.get_current_coordinates()
        trajs = [[] for _ in range(n_samples)]
        for i in range(n_samples):
            self.set_all_coordinates(start_point)
            for j in range(sample_len):
                self.bd.optimize(sample_interval)
                trajs[i].append(self.get_current_coordinates())
        return trajs

    def get_current_coordinates(self):
        """Returns the current coordinates of all terminal particles."""
        return np.array([(pxyz.get_x(), pxyz.get_y(), pxyz.get_z()) for pxyz in self._all_xyzs])

    def get_current_coordinates_numpy(self):
        """Returns the current coordinates of all terminal particles."""
        # TODO: This probably has a bug
        xyzs, rs = self.model.get_spheres_numpy()
        return xyzs[np.all(np.isfinite(xyzs), axis=1)].copy()

    def set_all_coordinates(self, vals):
        """Sets all particle coordinates to vals. vals should be an iterable of
        coordinates for each particle."""
        for i, v in enumerate(self._all_xyzs):
            v.set_coordinates(IMP.algebra.Vector3D(*vals[i]))

    def reset_coordinates(self):
        """Sets all coordinates to their initial state."""
        set_state = None
        if self.init_state is None:
            # zvec = IMP.algebra.Vector3D(0, 0, 0)
            # for v in self._all_xyzs:
            #     v.set_coordinates(zvec)
            # set_state = [[0, 0, 0] for _ in self._all_xyzs]
            set_state = self.first_state
        elif isinstance(self.init_state, int) and self.init_state == 0:
            set_state = [[0, 0, 0] for _ in self._all_xyzs]
        elif callable(self.init_state):
            set_state = self.init_state()
        else:
            set_state = self.init_state
        self.set_all_coordinates(set_state)

    # TODO: Find a way to save/serialize Swig objects
    def __getstate__(self):
        if self.bd is not None:
            self._last_time = self.bd.get_current_time()
            self._last_state = self.get_current_coordinates()
        m = self.model
        rsf = self.scoring_func
        hroot = self._h_root
        xyzs = self._all_xyzs
        bd = self.bd
        self.model = None
        self.scoring_func = None
        self._h_root = None
        self._all_xyzs = None
        self.bd = None
        ret = dict(self.__dict__)
        self.model = m
        self.scoring_func = rsf
        self._h_root = hroot
        self._all_xyzs = xyzs
        self.bd = bd
        return ret

    def __setstate__(self, state):
        # warnings.warn("Deserializing IMP BD sampler: re-initialize this instance "
        #               "with init_bd(model, scoring function) to avoid errors when sampling.")
        self.__dict__ = state
