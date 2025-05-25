import warnings
import argparse
import time
import numpy as np
import pickle
import os
import signal
import openmm.app as ommapp
from implementations.alaninedp.alaninedp_sim import DialanineOMMSampler
from implementations.alaninedp.alaninedp_discretizers import disc_dhdrl, disc_dhdrl_vs, disc_dhdrl_7
from scipy.stats import binned_statistic_2d
from experiments.runners.run_utils import count_transitions, GracefulKiller
warnings.filterwarnings("ignore", category=UserWarning)


def d_fn(traj):
    return np.vstack((disc_dhdrl_vs(traj))).T

def d_fn2(traj):
    return np.vstack(disc_dhdrl_7(traj))

def coarse_hist(hist, num_bins=30):
    rng = np.linspace(-180, 180, hist.shape[0] + 1)
    rng = np.array([(rng[i] + rng[i+1]) / 2 for i in range(len(rng) - 1)])
    xs, ys = np.meshgrid(rng, rng, indexing='ij')
    chist = np.histogram2d(xs.flatten(), ys.flatten(), weights=hist.flatten(), bins=(num_bins, num_bins), range=((-180, 180), (-180, 180)))[0]
    return (chist > 0).sum() / chist.size

def calc_exp_rate_kcntrs(tp):
    zero_rows_count = np.sum(~np.all(tp.count_matrix == 0, axis=1))
    return zero_rows_count / tp.kcenters.n_states

def calc_exp_rate(tp):
    non_zero_centers = ~np.all(tp.count_matrix == 0, axis=1)
    non_zero_centers = tp.kcenters.centers[non_zero_centers]
    chist = np.histogram2d(non_zero_centers[:, 0], non_zero_centers[:, 1], bins=(30, 30),
                           range=((-180, 180), (-180, 180)))[0]
    return (chist > 0).sum() / chist.size

def calc_exp_rate_2dgrid(tp):
    non_zero_centers = ~np.all(tp.count_matrix == 0, axis=1)
    non_zero_centers_idx = np.where(non_zero_centers)[0]
    nz_x, nz_y = np.unravel_index(non_zero_centers_idx, shape=tp.num_bins)
    nz_phi = [(tp.bins[0][i] + tp.bins[0][i+1]) / 2 for i in nz_x]
    nz_psi = [(tp.bins[1][i] + tp.bins[1][i+1]) / 2 for i in nz_y]
    chist = np.histogram2d(nz_phi, nz_psi, bins=(30, 30),
                           range=((-180, 180), (-180, 180)))[0]
    return (chist > 0).sum() / chist.size


class TrajProcessor2DGrid:
    def __init__(self, discretizer_fn, edges, num_bins):
        self.dfn = discretizer_fn
        self.edges = edges
        self.num_bins = num_bins
        self.bins = [np.linspace(edges[0][0], edges[0][1], num=num_bins[0]+1),
                     np.linspace(edges[1][0], edges[1][1], num=num_bins[1]+1)]
        self.n_states = self.num_bins[0] * self.num_bins[1]
        self.last_config = None
        self.count_matrix = np.zeros((self.n_states, self.n_states))
        self.total_runtime_ns = 0
        self.run_statistics = {"time_ns": [], "exp_rate": []}

    def process(self, data):
        dtraj = self.dfn(data)
        self.last_config = data[-1]
        binned = binned_statistic_2d(dtraj[:, 0], dtraj[:, 1], None, statistic='count',
                                     bins=self.bins, expand_binnumbers=True)
        traj = np.ravel_multi_index((binned.binnumber-1), self.num_bins)
        self.count_matrix += count_transitions(traj, n_states=self.n_states)

    def index_to_state(self, state_ids):
        """Converts a state/s id in [0,1,...,self.n_states] to a 2D configuration."""
        nz_x, nz_y = np.unravel_index(state_ids, shape=self.num_bins)
        nz_xval = [(self.bins[0][i] + self.bins[0][i + 1]) / 2 for i in nz_x]
        nz_yval = [(self.bins[1][i] + self.bins[1][i + 1]) / 2 for i in nz_y]
        return np.vstack([nz_xval, nz_yval]).T

class TrajProcessorKmeds:
    def __init__(self, discretizer_fn, kcenters):
        self.dfn = discretizer_fn
        self.kcenters = kcenters
        self.count_matrix = np.zeros((self.kcenters.n_states, self.kcenters.n_states))
        self.last_config = None
        self.total_runtime_ns = 0
        self.run_statistics = {"time_ns": [], "exp_rate": []}

    def process(self, data):
        dtraj = self.dfn(data)
        self.last_config = data[-1]
        dists, clstrs = self.kcenters._nearest_neighbors.kneighbors(dtraj, n_neighbors=1)
        self.count_matrix += count_transitions(clstrs.flatten(), n_states=self.kcenters.n_states)


def naive_2d_run(runtime_ns, out_file, tp_cont):
    STEP_SIZE_FS = 2
    TAU_MULT = 10
    TRAJ_SIZE = 2000
    sampler = DialanineOMMSampler(PATH_TOP,
                                  cuda=False, temp0=400, return_vs=True, dt_ps=STEP_SIZE_FS * 1e-3,
                                  concurrent_sims=1)
    # inpcrd = ommapp.AmberInpcrdFile(PATH_CRD).positions
    # startp = DialanineOMMSampler.quantity_to_array(inpcrd)
    # startp = np.stack([startp, np.zeros_like(startp)])
    startp = np.load(PATH_CRD_MIN)

    if tp_cont is None:
        tp = TrajProcessor2DGrid(discretizer_fn=d_fn, edges=[[-180, 180], [-180, 180]], num_bins=(100, 100))
        tp.last_config = startp
    else:
        tp = tp_cont

    run_sim2(tp, sampler, runtime_ns, TRAJ_SIZE, TAU_MULT, out_file, stats=True)


def run_sim2(tp, sampler, runtime_ns, traj_size, traj_step, out_file, stats=False):
    # traj_step == the number of base (dt) steps between trajectory frames
    sim_start = time.time()
    start_sim_time_ns = tp.total_runtime_ns
    sim_time_ns = 0
    traj = []
    killer = GracefulKiller()
    save_timer = time.time()

    # rnd_config = tp.last_config

    print(f"A simulation of {runtime_ns} ns started at {time.ctime()}")
    while sim_time_ns < runtime_ns:
        traj = sampler.sample_from_states([tp.last_config], traj_size, 1, traj_step)[0]
        tp.process(traj)
        sim_time_ns += traj_size * sampler.dt * 1e-3 * traj_step
        tp.total_runtime_ns += traj_size * sampler.dt * 1e-3 * traj_step

        traj = []
        print(
            f"\r({os.getpid()}) Simulated {sim_time_ns + start_sim_time_ns:.3f}/{runtime_ns + start_sim_time_ns} ns ({3600 * sim_time_ns / (time.time() - sim_start):.3f} ns/h, "
            f"running for {(time.time() - sim_start) / 3600:.2f} hrs)", end="")
        if stats:
            tp.run_statistics['time_ns'].append(tp.total_runtime_ns)
            # tp.run_statistics['exp_rate'].append(calc_exp_rate(tp))
            tp.run_statistics['exp_rate'].append(calc_exp_rate_2dgrid(tp))
        if (time.time() - save_timer) / 3600 >= 0.25:  # Saves every <rhs>> hours

            print(
                f"\r({os.getpid()}) Simulated {sim_time_ns + start_sim_time_ns:.3f}/{runtime_ns + start_sim_time_ns} ns ({3600 * sim_time_ns / (time.time() - sim_start):.3f} ns/h, "
                f"running for {(time.time() - sim_start) / 3600:.2f} hrs)", end="", flush=True)
            with open(out_file, 'wb') as f2:
                pickle.dump(tp, f2)
            with open(out_file + "_bu", 'wb') as f2:
                pickle.dump(tp, f2)
            save_timer = time.time()
        if killer.kill_now:
            break
    with open(out_file, 'wb') as f2:
        pickle.dump(tp, f2)
    with open(out_file + "_bu", 'wb') as f2:
        pickle.dump(tp, f2)
    traj = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Dialanine simulation.')
    parser.add_argument('--out', type=str, help='Output TP file.', required=True)
    parser.add_argument('--runtime_ns', type=int, help='Simulation time (ns).', default=5)
    parser.add_argument('--c', type=str, help='Continue previous run.')
    args = parser.parse_args()

    PATH_TOP = "../../implementations/alaninedp/runfiles/diala.top"
    PATH_CRD = "../../implementations/alaninedp/runfiles/diala.crd"
    PATH_CRD_MIN = "../../implementations/alaninedp/runfiles/minimized_initial_state.npy"

    if args.c is not None:
        with open(args.c, 'rb') as f:
            tp_prev = pickle.load(f)
    else:
        tp_prev = None
    naive_2d_run(args.runtime_ns, args.out, tp_prev)
