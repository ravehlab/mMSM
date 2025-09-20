import pickle
from implementations.twospheres.imp_bd_sampler import IMPBrownianDynamicsSampler
from implementations.twospheres.imp_test_discretizers import DistanceDiscretizer
from mmsm.self_expanding_mmsm import SelfExpandingMultiscaleMSM
from implementations.twospheres.imp_test_models import *
from experiments.runners.run_utils import run_hmsm
from mmsm.mmsm_config import mMSMConfig
import argparse


def coverage_hmsm_1d(clstrs, left, right):
    chist = np.histogram(clstrs, bins=35, range=(left, right))
    return (chist[0] > 0).sum() / chist[0].size


def hmsm_full_hist(hmsm, disc_fn):
    st_hmsm = hmsm.tree.get_level_stationary_distribution(0)
    ids = list(st_hmsm.keys())
    hmsm_clstrs = hmsm.discretizer.get_centers_by_ids(st_hmsm.keys())
    hmsm_clstrs = np.array([disc_fn(cl) for cl in hmsm_clstrs]).flatten()
    st_hmsm = np.array([st_hmsm[ms] for ms in ids])
    hmsm_clstrs_incr = np.argsort(hmsm_clstrs)
    st_hmsm = st_hmsm[hmsm_clstrs_incr]
    hmsm_clstrs = hmsm_clstrs[hmsm_clstrs_incr]
    return hmsm_clstrs, st_hmsm, hmsm_clstrs_incr


def hmsm_stats_1d(hmsm: SelfExpandingMultiscaleMSM, mmsm_data, disc_fn, edges):
    if "time_ns" not in mmsm_data:
        mmsm_data["time_ns"] = []
    if "exp_rate" not in mmsm_data:
        mmsm_data["exp_rate"] = []

    hmsm_clstrs, st_hmsm, incr = hmsm_full_hist(hmsm, disc_fn)
    mmsm_data["time_ns"].append(hmsm._sampler.total_simulation_time * 1e-6)
    mmsm_data["exp_rate"].append(coverage_hmsm_1d(hmsm_clstrs, edges[0], edges[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime_ns', type=int)
    parser.add_argument('--out', type=str)
    parser.add_argument('--c', type=str, required=False)
    args = parser.parse_args()

    model_fn = model1
    f_name = args.out
    runtime_ns = args.runtime_ns
    disc_cutoff = 0.05
    diameter_threshold = 4
    step_size = 30
    base_tau = 1
    rep_sample_size = 50
    print_tree_info = False
    save_interval = 3

    edges = [22, 24.41, 30.32, 39, 47.68, 53.6, 56]
    mins = [22.7, 27., 34., 44., 51., 55.25]

    def state_init_fn():
        return [[0., 0., 0.], [mins[-1], 0., 0.]]

    def stat_fn_model1(hmsm, data2):
        hmsm_stats_1d(hmsm, data2, lambda x: x, edges)

    if args.c is None:
        sampler = IMPBrownianDynamicsSampler(*model_fn(), warmup=None, step_size_fs=step_size)
        discretizer = DistanceDiscretizer(cutoff=disc_cutoff, representative_sample_size=rep_sample_size)
        cnfg = mMSMConfig(n_trajectories=5, trajectory_len=10**4, max_height=-1,
                          sampling_heuristics=['equilibrium', 'exploration'],
                          sampling_heuristic_weights=[0.3, 0.7], partition_diameter_threshold=diameter_threshold,
                          base_tau=base_tau, random_split=0.05)
        run_hmsm(sampler, discretizer, x_init=state_init_fn(), mmsmconfig=cnfg, f_name=f_name, runtime_ns=runtime_ns,
                 print_tree_info=print_tree_info, stats_fn=stat_fn_model1, save_interval=save_interval)
    else:
        prev_name = args.c
        with open(f'{prev_name}.pkl', 'rb') as f:
            mmsm = pickle.load(f)
        with open(f'{prev_name}.data', 'rb') as f:
            data = pickle.load(f)
        mmsm._sampler.init_bd(*model_fn(), deserialized=True)
        run_hmsm(None, None, x_init=state_init_fn(), mmsmconfig=None, f_name=f_name,
                 runtime_ns=runtime_ns, print_tree_info=print_tree_info, mmsm_init=mmsm, mmsm_init_data=data,
                 stats_fn=stat_fn_model1, save_interval=save_interval)


