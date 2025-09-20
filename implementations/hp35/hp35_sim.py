import numpy as np
from mmsm.mmsm_base.base_trajectory_sampler import BaseTrajectorySampler
from scipy.ndimage import gaussian_filter1d

class HP35SamplerPreexisting(BaseTrajectorySampler):
    def __init__(self, dihs_traj_path):
        self.dihs_Traj_path = dihs_traj_path
        self.current_frame = 0

        data = np.loadtxt(dihs_traj_path).reshape(-1, 42)

        sigma_ns = 2.0
        dt_ns = 0.2

        sigma_frames = float(sigma_ns) / float(dt_ns)


        self.data = np.empty_like(data)
        for j in range(data.shape[1]):
            self.data[:, j] = gaussian_filter1d(data[:, j], sigma=sigma_frames, mode="nearest")

        # mu = self.data.mean(axis=0)  # shape (42,)
        # sigma = self.data.std(axis=0, ddof=1)  # shape (42,)
        # sigma[sigma == 0] = 1.0  # avoid division by zero for constant features
        # X_full_std = (self.data - mu) / sigma
        #
        # U, S, Vt = np.linalg.svd(X_full_std, full_matrices=False)
        # Wk = Vt[:10]

    @property
    def timestep(self):
        return 200  # This is in picoseconds

    @property
    def total_simulation_time(self):
        return self.current_frame * self.timestep

    def sample_from_states(self, states, sample_len, n_samples, sample_interval=1):
        if self.current_frame > len(self.data):
            raise Exception
        ret = self.data[self.current_frame:self.current_frame+sample_len].copy()
        self.current_frame += sample_len-1
        return [ret]
