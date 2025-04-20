import warnings
import argparse
import time
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import signal

# from dialanine.amber_sampler import AmberDialanineSampler
import openmm.app as ommapp
from systems.alaninedp.alaninedp_sim import DialanineOMMSampler
from systems.alaninedp.alaninedp_discretizers import disc_dhdrl, disc_dhdrl_vs, disc_dhdrl_7
from scipy.stats import binned_statistic_2d
from msmtools.estimation import count_matrix
from datetime import datetime
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

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


class TrajProcessor2D:
    def __init__(self, discretizer_fn, edges, num_bins=(50, 50), num_trajs=1):
        self.dfn = discretizer_fn
        self.edges = edges
        self.num_bins = num_bins
        self.n_states = self.num_bins[0] * self.num_bins[1]
        self.hist = np.zeros(num_bins)
        self.last_configs = [None for _ in range(num_trajs)]
        self.statistics = dict()
        self.t_counts = np.zeros((self.n_states, self.n_states))

    def process(self, data, traj_id=0):
        self.last_configs[traj_id] = data[-1]
        dtraj = self.dfn(data)
        binned = binned_statistic_2d(dtraj[:, 0], dtraj[:, 1], None, statistic='count',
                                     bins=self.num_bins, range=self.edges, expand_binnumbers=True)
        self.hist += binned.statistic
        traj = np.ravel_multi_index((binned.binnumber-1), self.num_bins)
        self.t_counts += count_matrix(traj, lag=1, sparse_return=False, nstates=self.n_states)


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
        self.count_matrix += count_matrix(traj, lag=1, sparse_return=False, nstates=self.n_states)

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
        self.count_matrix += count_matrix(clstrs.flatten(), lag=1, sparse_return=False, nstates=self.kcenters.n_states)


def naive_2d_run(runtime_ns, out_file, tp_cont):
    STEP_SIZE_FS = 2
    TAU_MULT = 10
    TRAJ_SIZE = 20000
    sampler = DialanineOMMSampler(PATH_TOP,
                                  cuda=False, temp0=400, return_vs=True, dt_ps=STEP_SIZE_FS * 1e-3,
                                  concurrent_sims=1)
    # inpcrd = ommapp.AmberInpcrdFile(PATH_CRD).positions
    # startp = DialanineOMMSampler.quantity_to_array(inpcrd)
    # startp = np.stack([startp, np.zeros_like(startp)])
    startp = np.load(PATH_CRD_MIN)
    # with open(PATH_KCNTRS, 'rb') as f1:
    #     kcenters = pickle.load(f1)

    if tp_cont is None:
        # tp = TrajProcessorKmeds(discretizer_fn=d_fn, kcenters=kcenters)
        tp = TrajProcessor2DGrid(discretizer_fn=d_fn, edges=[[-180, 180], [-180, 180]], num_bins=(100, 100))
        tp.last_config = startp
    else:
        tp = tp_cont

    run_sim2(tp, sampler, runtime_ns, TRAJ_SIZE, TAU_MULT, out_file, stats=True)
    # run_sim_adap(tp, sampler, runtime_ns, TRAJ_SIZE, TAU_MULT, out_file, stats=True)


def naive_2d_run_old(runtime_ns, out_file, tp_cont):
    STEP_SIZE_FS = 2
    TAU_MULT = 1
    TRAJ_SIZE = 1000
    sampler = DialanineOMMSampler('/home/nirn/ravehlab/misc/amber_dialanine/diala.top',
                                  cuda=True, temp0=400, return_vs=True, dt_ps=STEP_SIZE_FS * 1e-3,
                                  concurrent_sims=1)
    inpcrd = ommapp.AmberInpcrdFile('/home/nirn/ravehlab/misc/amber_dialanine/diala.crd').positions
    startp = DialanineOMMSampler.quantity_to_array(inpcrd)
    startp = np.stack([startp, np.zeros_like(startp)])
    kcenters_path = "/home/nirn/ravehlab/code/test/kcenters2d.pkl"
    with open(kcenters_path, 'rb') as f1:
        kcenters = pickle.load(f1)

    if tp_cont is None:
        tp = TrajProcessor2DGrid(discretizer_fn=d_fn, edges=[[-180, 180], [-180, 180]], num_bins=(100, 100))
        tp.last_config = startp
    else:
        tp = tp_cont

    run_sim2(tp, sampler, runtime_ns, TRAJ_SIZE, TAU_MULT, out_file)


def naive_7d_run(runtime_ns, out_file, tp_cont):
    STEP_SIZE_FS = 2
    TAU_MULT = 20
    TRAJ_SIZE = 1000
    sampler = DialanineOMMSampler('/home/nirn/ravehlab/misc/amber_dialanine/diala.top',
                                  cuda=True, temp0=400, return_vs=True, dt_ps=STEP_SIZE_FS * 1e-3,
                                  concurrent_sims=1)
    inpcrd = ommapp.AmberInpcrdFile('/home/nirn/ravehlab/misc/amber_dialanine/diala.crd').positions
    startp = DialanineOMMSampler.quantity_to_array(inpcrd)
    startp = np.stack([startp, np.zeros_like(startp)])
    kcenters_path = "/home/nirn/ravehlab/code/test/kcenters7d.pkl"
    with open(kcenters_path, 'rb') as f1:
        kcenters = pickle.load(f1)

    if tp_cont is None:
        tp = TrajProcessorKmeds(discretizer_fn=d_fn2, kcenters=kcenters)
        tp.last_config = startp
    else:
        tp = tp_cont

    run_sim2(tp, sampler, runtime_ns, TRAJ_SIZE, TAU_MULT, out_file)


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


def run_sim_adap(tp, sampler, runtime_ns, traj_size, traj_step, out_file, stats=False):
    # traj_step == the number of base (dt) steps between trajectory frames
    sim_start = time.time()
    start_sim_time_ns = tp.total_runtime_ns
    sim_time_ns = 0
    traj = []
    killer = GracefulKiller()
    save_timer = time.time()

    start_config = tp.last_config

    with open("/home/nirn/ravehlab/code/test/hinf_cls_disc", 'rb') as f:
        disc = pickle.load(f)

    print(f"A simulation of {runtime_ns} ns started at {time.ctime()}")
    while sim_time_ns < runtime_ns:
        traj = sampler.sample_from_states([start_config], traj_size, 1, traj_step)[0]
        tp.process(traj)
        sim_time_ns += traj_size * sampler.dt * 1e-3 * traj_step
        tp.total_runtime_ns += traj_size * sampler.dt * 1e-3 * traj_step

        state_p = tp.count_matrix.sum(axis=1)
        state_p = np.exp(-state_p / 1e3)
        state_p = state_p / np.sum(state_p)
        state = np.random.choice(tp.kcenters.n_states, p=state_p)
        start_config = disc.sample_from(disc._kcenters._cluster_inx_2_id[state])


        traj = []
        print(
            f"\r({os.getpid()}) Simulated {sim_time_ns + start_sim_time_ns:.3f}/{runtime_ns + start_sim_time_ns} ns ({3600 * sim_time_ns / (time.time() - sim_start):.3f} ns/h, "
            f"running for {(time.time() - sim_start) / 3600:.2f} hrs)", end="")

        if (time.time() - save_timer) / 3600 >= 0:  # Saves every <rhs> hours
            if stats:
                tp.run_statistics['time_ns'].append(tp.total_runtime_ns)
                tp.run_statistics['exp_rate'].append(calc_exp_rate(tp))
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


def run_sim(runtime_ns, out_file, tp_cont):
    STEP_SIZE_FS = 2
    TRAJ_SIZE = 100000
    # sampler = AmberDialanineSampler(dt=STEP_SIZE_FS,
    #                                 parmtop_path="/cs/labs/ravehb/nirnits/out/misc/amber_dialanine_400k/diala.top",
    #                                 init_coords_path='/cs/labs/ravehb/nirnits/out/misc/amber_dialanine_400k/diala.crd',
    #                                 input_file_path='/cs/labs/ravehb/nirnits/out/misc/amber_dialanine_400k/md.in',
    #                                 tmp_location='/cs/labs/ravehb/nirnits/out/MMSM/dialanine/naive/tmps/amber_tmp',
    #                                 concurrent_sims=1)
    sampler = DialanineOMMSampler('/home/nirn/ravehlab/misc/amber_dialanine/diala.top',
                                  cuda=False, temp0=400, return_vs=False, dt_ps=STEP_SIZE_FS*1e-3)
    inpcrd = ommapp.AmberInpcrdFile('/home/nirn/ravehlab/misc/amber_dialanine/diala.crd').positions
    startp = DialanineOMMSampler.quantity_to_array(inpcrd)

    if tp_cont is None:
        tp = TrajProcessor2D(discretizer_fn=d_fn, edges=[[-180, 180], [-180, 180]], num_bins=(100, 100), num_trajs=1)
        for i in range(len(tp.last_configs)):
            tp.last_configs[i] = startp
        tp.statistics["step_size"] = STEP_SIZE_FS
        tp.statistics['time'] = []
        tp.statistics['coverage'] = []
    else:
        tp = tp_cont

    sim_start = time.time()
    start_time_ns = tp.statistics['time'][-1] if len(tp.statistics['time']) > 0 else 0.0
    sim_time_fs = 0
    traj = []
    cur_traj = 0
    killer = GracefulKiller()
    save_timer = time.time()
    while sim_time_fs * 1e-6 < runtime_ns:
        traj = sampler.sample_from_states([tp.last_configs[cur_traj]], TRAJ_SIZE, 1)[0]
        tp.process(traj, traj_id=cur_traj)
        sim_time_fs += TRAJ_SIZE * STEP_SIZE_FS
        tp.statistics['time'].append(sim_time_fs * 1e-6 + start_time_ns)
        tp.statistics['coverage'].append(coarse_hist(tp.hist))
        cur_traj = (cur_traj + 1) % len(tp.last_configs)
        traj = []
        print(f"\r({os.getpid()}) Simulated {sim_time_fs * 1e-6 + start_time_ns:.3f}/{runtime_ns + start_time_ns} ns ({3600*(sim_time_fs * 1e-6) / (time.time() - sim_start):.3f} ns/h, "
              f"simulated for {(time.time() - sim_start) / 3600:.2f} hrs)", end="")

        if (time.time() - save_timer) / 3600 >= 0.01:
            print(f"({os.getpid()}) Simulated {sim_time_fs * 1e-6 + start_time_ns:.3f}/{runtime_ns + start_time_ns} ns ({3600 * (sim_time_fs * 1e-6) / (time.time() - sim_start):.3f} ns/h, "
                f"simulated for {(time.time() - sim_start) / 3600:.2f} hrs)", flush=True)
            with open(out_file, 'wb') as f:
                pickle.dump(tp, f)
            with open(out_file + "_bu", 'wb') as f:
                pickle.dump(tp, f)
            save_timer = time.time()
        if killer.kill_now:
            break
    with open(out_file, 'wb') as f:
        pickle.dump(tp, f)
    with open(out_file + "_bu", 'wb') as f:
        pickle.dump(tp, f)
    traj = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Dialanine simulation.')
    parser.add_argument('--out', type=str, help='Output TP file.', required=True)
    parser.add_argument('--simtime', type=int, help='Simulation time (ns).', default=5)
    parser.add_argument('--c', type=str, help='Continue previous run.')
    args = parser.parse_args()

    PATH_TOP = "/home/nirn/ravehlab/misc/amber_dialanine/diala.top"
    PATH_CRD = "/home/nirn/ravehlab/misc/amber_dialanine/diala.crd"
    PATH_CRD_MIN = "/home/nirn/ravehlab/code/runfiles/minimized_initial_state.npy"
    PATH_KCNTRS = "/home/nirn/ravehlab/code/test/kcenters2d.pkl"

    # PATH_TOP = "/cs/labs/ravehb/nirnits/sbatch_remote/runfiles/diala.top"
    # PATH_CRD = "/cs/labs/ravehb/nirnits/sbatch_remote/runfiles/diala.crd"
    # PATH_CRD_MIN = "/cs/labs/ravehb/nirnits/sbatch_remote/runfiles/minimized_initial_state.npy"
    # PATH_KCNTRS = "/cs/labs/ravehb/nirnits/sbatch_remote/runfiles/kcenters2d.pkl"

    if args.c is not None:
        with open(args.c, 'rb') as f:
            tp_prev = pickle.load(f)
    else:
        tp_prev = None
    naive_2d_run(args.simtime, args.out, tp_prev)
    # naive_7d_run(args.simtime, args.out, tp_prev)
    # naive_2d_run_old(args.simtime, args.out, tp_prev)
    # run_sim(args.simtime, args.out, tp_prev)
    # run_sim2(args.simtime, args.out, tp_prev)
    # inspect_tp()
