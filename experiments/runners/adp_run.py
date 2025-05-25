import numpy as np
import pickle
import argparse
import warnings
from implementations.alaninedp.alaninedp_sim import DialanineOMMSampler
from mmsm.mmsm_config import mMSMConfig
from implementations.alaninedp.alaninedp_discretizers import DialanineDiscretizerV, AlanineAngleDiscretizer
from experiments.runners.run_utils import run_hmsm

warnings.filterwarnings("ignore", category=UserWarning)

def dial_mmsm_statfn(mmsm, data):
    if "time_ns" not in data:
        data["time_ns"] = []
    if "exp_rate" not in data:
        data["exp_rate"] = []
    centers = mmsm.discretizer._kcenters.centers
    # dists, assigns = kcenters._nearest_neighbors.kneighbors(centers, n_neighbors=1)
    # visited = np.sum(dists.flatten() < kcenters.cutoff)

    chist = np.histogram2d(centers[:, 0], centers[:, 1], bins=(30, 30),
                           range=((-180, 180), (-180, 180)))[0]
    data["time_ns"].append(mmsm.total_simulation_time * 1e-3)
    data["exp_rate"].append((chist > 0).sum() / chist.size)


def run_2d(args):
    print_tree_info = True
    STEP_SIZE_FS = 2
    runtime_ns = args.runtime_ns
    f_name = args.out
    stats_fn = dial_mmsm_statfn
    save_interval = 3
    time_mult = 1e-3
    base_tau = 1

    if args.c is None:
        sampler = DialanineOMMSampler(PATH_TOP, cuda=False, temp0=400, return_vs=True, dt_ps=STEP_SIZE_FS*1e-3,
                                      concurrent_sims=7)
        sampler.initial_state = np.load(PATH_CRD_MIN)

        discretizer = DialanineDiscretizerV(3)
        cnfg = mMSMConfig(n_trajectories=7, trajectory_len=10**4, max_height=-1, partition_diameter_threshold=8,
                          sampling_heuristics=['equilibrium', 'exploration', 'equilibrium_inv'],
                          sampling_heuristic_weights=[0.5, 0.25, 0.25], base_tau=base_tau
                          )
        run_hmsm(sampler, discretizer, cnfg, f_name, runtime_ns, print_tree_info=print_tree_info,
                 stats_fn=stats_fn, save_interval=save_interval, time_mult=time_mult)
    else:
        with open(f'{args.c}.pkl', 'rb') as f:
            mmsm = pickle.load(f)
        with open(f'{args.c}.data', 'rb') as f:
            data = pickle.load(f)
        run_hmsm(None, None, None, f_name, runtime_ns, print_tree_info=print_tree_info,
                 hmsm_init=mmsm, hmsm_init_data=data, stats_fn=stats_fn, save_interval=save_interval,
                 time_mult=time_mult)


def run_7d(args):
    print_tree_info = True
    STEP_SIZE_FS = 2
    runtime_ns = args.runtime_ns
    f_name = args.out
    stats_fn = None
    save_interval = 5
    time_mult = 1e-3
    if args.c is None:
        sampler = DialanineOMMSampler(PATH_TOP, cuda=False, temp0=400, return_vs=True, dt_ps=STEP_SIZE_FS * 1e-3,
                                      concurrent_sims=14)
        sampler.initial_state = np.load(PATH_CRD_MIN)
        discretizer = AlanineAngleDiscretizer(60, representative_sample_size=100)
        cnfg = mMSMConfig(n_trajectories=14, trajectory_len=10**3, max_height=-1, partition_diameter_threshold=8,
                          sampling_heuristics=['equilibrium'],
                          sampling_heuristic_weights=[1.0],
                          base_tau=50
                          )
        run_hmsm(sampler, discretizer, cnfg, f_name, runtime_ns, print_tree_info=print_tree_info,
                 stats_fn=stats_fn, save_interval=save_interval, time_mult=time_mult)
    else:
        with open(f'{args.c}.pkl', 'rb') as f:
            mmsm = pickle.load(f)
        with open(f'{args.c}.data', 'rb') as f:
            data = pickle.load(f)
        run_hmsm(None, None, None, f_name, runtime_ns, print_tree_info=print_tree_info,
                 hmsm_init=mmsm, hmsm_init_data=data, stats_fn=stats_fn, save_interval=save_interval,
                 time_mult=time_mult)

if __name__ == '__main__':
    PATH_TOP = "../../implementations/alaninedp/runfiles/diala.top"
    PATH_CRD = "../../implementations/alaninedp/runfiles/diala.crd"
    PATH_CRD_MIN = "../../implementations/alaninedp/runfiles/minimized_initial_state.npy"

    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime_ns', type=int)
    parser.add_argument('--out', type=str)
    parser.add_argument('--c', type=str, required=False)
    cargs = parser.parse_args()

    run_2d(cargs)
    # run_7d(cargs)
