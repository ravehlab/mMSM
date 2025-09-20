import pickle
import time
import os
import argparse
from implementations.twospheres.imp_test_models import *
from implementations.twospheres.imp_bd_sampler import IMPBrownianDynamicsSampler
from implementations.twospheres.imp_test_discretizers import dist_np
from scipy.stats import binned_statistic
from experiments.runners.run_utils import count_transitions, GracefulKiller


class TrajProcessor:
    def __init__(self, num_bins, bin_range, discretizer_fn):
        self.bin_range = bin_range
        self.num_bins = num_bins
        bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
        self.clstrs = np.array([(bins[b] + bins[b+1])/2 for b in range(len(bins)-1)])

        self.t_counts = np.zeros((self.clstrs.shape[0], self.clstrs.shape[0]))
        self.dfn = discretizer_fn
        self.hist = np.zeros_like(self.clstrs)
        self.last_config = None
        self.statistics = dict()

    def process(self, data):
        self.last_config = data[-1]
        dtraj = self.dfn(np.array(data))
        binned_s = binned_statistic(x=dtraj, values=None, statistic='count', bins=self.num_bins, range=self.bin_range)
        self.hist += binned_s.statistic
        dtraj = np.clip(binned_s.binnumber - 1, 0, self.num_bins-1)
        self.count_transitions(dtraj)

    def count_transitions(self, dtraj):
        self.t_counts += count_transitions(dtraj, n_states=self.clstrs.shape[0])


def coverage(hist, left, right):
    rng = np.linspace(left, right, hist.shape[0] + 1)
    rng = np.array([(rng[i] + rng[i+1]) / 2 for i in range(len(rng) - 1)])
    chist = np.histogram(rng.flatten(),  weights=hist.flatten(), bins=35, range=(left, right))[0]
    return (chist > 0).sum() / chist.size


def stat_fn_1d(tp: TrajProcessor, impbdsamp: IMPBrownianDynamicsSampler, edges):
    if "time_ns" not in tp.statistics:
        tp.statistics["time_ns"] = []
    if "exp_rate" not in tp.statistics:
        tp.statistics["exp_rate"] = []
    tp.statistics["time_ns"].append(impbdsamp.bd.get_current_time() * 1e-6)
    tp.statistics["exp_rate"].append(coverage(tp.hist, edges[0], edges[-1]))


def run_sim(out_file: str, runtime_ns, tp_cont=None):

    m, rsf = model1()

    edges = [21.8, 24.41, 30.32, 39, 47.68, 53.6, 56.2]
    mins = [22.7, 27., 34., 44., 51., 55.25]
    step_size_fs = 30
    n_bins = 300

    def state_init_fn():
        return [[0., 0., 0.], [mins[-1], 0., 0.]]

    print(f"Starting {out_file} | step size: {step_size_fs}, start point: {mins[-1]}")

    impbd_samp = IMPBrownianDynamicsSampler(m, rsf, warmup=None, step_size_fs=step_size_fs,
                                            init_state=state_init_fn)

    if tp_cont is None:
        tp = TrajProcessor(n_bins, (edges[0], edges[-1]), dist_np)

        tp.last_config = state_init_fn()
        impbd_samp.set_all_coordinates(tp.last_config)
        impbd_samp.bd.set_current_time(0.0)
        tp.statistics["step_size"] = step_size_fs
        tp.statistics['time_ns'] = []
    else:
        tp = tp_cont
        impbd_samp.set_all_coordinates(tp.last_config)
        impbd_samp.bd.set_current_time(tp.statistics["time_ns"][-1] / 1e-6)
        if step_size_fs != tp.statistics["step_size"]:
            print("Step size mismatch, config: {0}, actual: {1}".format(step_size_fs,
                                                                        tp.statistics["step_size"]))

    start_time = impbd_samp.bd.get_current_time() * 1e-6
    sim_start = time.time()
    traj = []
    cur_traj = 0
    killer = GracefulKiller()
    while impbd_samp.bd.get_current_time() * 1e-6 - start_time < runtime_ns:
        traj += impbd_samp.sample_from_current_state(500000, 1, 1)[0]
        tp.process(traj)
        stat_fn_1d(tp, impbd_samp, edges)
        traj = []
        with open(out_file, 'wb') as f:
            pickle.dump(tp, f)
        with open(f'{out_file}_bu', 'wb') as f:
            pickle.dump(tp, f)
        with open(f'{out_file}_stats', 'wb') as f:
            pickle.dump({'centers': tp.clstrs, 'hist': tp.hist, 'count_m': tp.t_counts, 'stats': tp.statistics}, f)
        print(f"({os.getpid()}) Simulated {impbd_samp.bd.get_current_time() * 1e-6}/{runtime_ns + start_time} ns "
              f"({3600*(impbd_samp.bd.get_current_time() * 1e-6 - start_time) / (time.time() - sim_start)} ns/h)")
        if killer.kill_now:
            break
    traj = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive simulation.')
    parser.add_argument('--out', type=str, help='Output path to save TrajProcessor object.', required=True)
    parser.add_argument('--runtime_ns', type=int, help='Simulation time (ns).', default=5)
    parser.add_argument('--c', type=str, help='Continue previous run.')
    cargs = parser.parse_args()

    if cargs.c is not None:
        with open(cargs.c, 'rb') as f1:
            tp_prev = pickle.load(f1)
    else:
        tp_prev = None

    run_sim(cargs.out, cargs.runtime_ns, tp_prev)
