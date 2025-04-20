"""
Generic code to run mMSMs and gather run statistics.
"""

import numpy as np
import pickle
import time
import os
from mmsm.self_expanding_mmsm import SelfExpandingMultiscaleMSM
from mmsm.mmsm_config import mMSMConfig


def run_hmsm(sampler, discretizer, hmsmconfig: mMSMConfig = None, f_name=None, runtime_ns=100, hmsm_init=None, hmsm_init_data=None,
             stats_fn=None, print_tree_info=False, save_interval=1, time_mult=1e-6):
    # time_mult = the multiplier to change the sampler's units to nanoseconds
    print(f"({os.getpid()}) {f_name}")
    if hmsm_init is None:
        hmsm = SelfExpandingMultiscaleMSM(sampler, discretizer, config=hmsmconfig)
        data = dict()
        total_runtime = runtime_ns
        data["start_times"] = [0.0]
    else:
        hmsm = hmsm_init
        data = hmsm_init_data
        total_runtime = hmsm.total_simulation_time*time_mult + runtime_ns
        data["start_times"].append(hmsm.total_simulation_time*time_mult)

    def save_all():
        if f_name is not None:
            with open(f_name + ".pkl", 'wb') as f:
                pickle.dump(hmsm, f)
            with open(f_name + ".data", 'wb') as f:
                pickle.dump(data, f)
            with open(f_name + "_bu.pkl", 'wb') as f:
                pickle.dump(hmsm, f)
            with open(f_name + "_bu.data", 'wb') as f:
                pickle.dump(data, f)

    # the total_simulation_time in the mmsm is in the time units of its sampler.
    # time_mult should be the conversion factor
    start_time_ns = hmsm.total_simulation_time*time_mult
    s_time = time.time()
    save_counter = 0
    while hmsm.total_simulation_time*time_mult < total_runtime:
        e = time.time()
        if e - s_time > 0:
            print("\r({3}) Total simulation time: {0:.3f}/{1} ns ({2:.4f}ns/h)".format(hmsm.total_simulation_time*time_mult, total_runtime,
                                                                     3600*((hmsm.total_simulation_time*time_mult - start_time_ns) / (time.time() - s_time)),
                                                                                   os.getpid()))
        hmsm.expand(max_batches=save_interval)
        if stats_fn is not None:
            stats_fn(hmsm, data)
        save_counter += 1
        if save_counter >= save_interval*2:
            print('saved')
            save_all()
            save_counter = 0
        if print_tree_info:
            print("\r")
            print("Height: {0} | Vertices: {1} | Microstates: {2}".format(
                hmsm.tree.height,
                [len(hmsm.tree.get_level(i)) for i in range(hmsm.tree.height + 1)],
                len(hmsm.tree._microstate_counts)))
    print("")
    print("Final tree structure:")
    print("Height: {0} | Vertices: {1} | Microstates: {2}".format(
        hmsm.tree.height,
        [len(hmsm.tree.get_level(i)) for i in range(hmsm.tree.height + 1)],
        len(hmsm.tree.get_microstates())))
    save_all()


def subset_transition(T, sd, assignment):
    ids = np.unique(assignment).astype('int')
    num_subsets = len(ids)
    T_reduced = np.zeros((num_subsets, num_subsets))
    for i, subset_from in enumerate(ids):
        for j, subset_to in enumerate(ids):
            res = T[assignment == subset_from, :][:, assignment == subset_to]
            res = res.sum(axis=1).dot(sd[assignment == subset_from])
            if res == 0:
                T_reduced[i, j] = 0
            else:
                T_reduced[i, j] = res / sd[assignment == subset_from].sum()

    return np.nan_to_num(T_reduced)
