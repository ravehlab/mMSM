"""
Generic code to run mMSMs and gather run statistics, as well as utilities for naive simulations.
"""

import numpy as np
import pickle
import time
import os
import signal
from mmsm.self_expanding_mmsm import SelfExpandingMultiscaleMSM
from mmsm.mmsm_config import mMSMConfig

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


def print_levels(mmsm: SelfExpandingMultiscaleMSM):
    print("Height: {0} | Vertices: {1} | Microstates: {2}".format(
        mmsm.tree.height,
        [len(mmsm.tree.get_level(i)) for i in range(mmsm.tree.height + 1)],
        len(mmsm.tree._microstate_counts)))


def run_hmsm(sampler, discretizer, x_init, mmsmconfig: mMSMConfig = None, f_name=None, runtime_ns=100, mmsm_init=None, mmsm_init_data=None,
             stats_fn=None, print_tree_info=False, save_interval=1, time_mult=1e-6):
    # time_mult = the multiplier to change the sampler's units to nanoseconds
    print(f"({os.getpid()}) {f_name}")
    if mmsm_init is None:
        mmsm = SelfExpandingMultiscaleMSM(sampler, discretizer, x_init=x_init, config=mmsmconfig)
        data = dict()
        total_runtime = runtime_ns
        data["start_times"] = [0.0]
    else:
        mmsm = mmsm_init
        data = mmsm_init_data
        sampler = mmsm._sampler
        total_runtime = sampler.total_simulation_time*time_mult + runtime_ns
        data["start_times"].append(sampler.total_simulation_time*time_mult)

    def save_all():
        if f_name is not None:
            with open(f_name + ".pkl", 'wb') as f:
                pickle.dump(mmsm, f)
            with open(f_name + ".data", 'wb') as f:
                pickle.dump(data, f)
            with open(f_name + "_bu.pkl", 'wb') as f:
                pickle.dump(mmsm, f)
            with open(f_name + "_bu.data", 'wb') as f:
                pickle.dump(data, f)

    # the total_simulation_time in the mmsm is in the time units of its sampler.
    # time_mult should be the conversion factor
    start_time_ns = sampler.total_simulation_time*time_mult
    s_time = time.time()
    save_counter = 0
    while sampler.total_simulation_time*time_mult < total_runtime:
        e = time.time()
        if e - s_time > 0:
            print("\r({3}) Total simulation time: {0:.3f}/{1} ns ({2:.4f}ns/h)".format(sampler.total_simulation_time*time_mult, total_runtime,
                                                                     3600*((sampler.total_simulation_time*time_mult - start_time_ns) / (time.time() - s_time)),
                                                                                   os.getpid()))
        mmsm.expand(max_batches=save_interval)
        if stats_fn is not None:
            stats_fn(mmsm, data)
        save_counter += 1
        if True:
            save_all()
            save_counter = 0
        if print_tree_info:
            print("\r")
            print_levels(mmsm)
    print("")
    print("Final tree structure:")
    print_levels(mmsm)
    save_all()

    # times = TimingLogger.get_times()
    # print(times)
    # with open(f'{f_name}.times', 'wb') as f:
    #     pickle.dump(times, f)

def count_transitions(traj: np.ndarray,
                      n_states: int,
                      counts: np.ndarray = None) -> np.ndarray:
    """
    Count state‐to‐state transitions in a trajectory.

    Parameters
    ----------
    traj : np.ndarray, shape (T,)
        Array of integer states in [0, n_states).
    n_states : int
        Number of possible states.
    counts : np.ndarray or None, shape (n_states, n_states)
        If provided, this matrix will be incremented in place.
        Otherwise a new zero‐initialized matrix is created.

    Returns
    -------
    counts : np.ndarray, shape (n_states, n_states)
        counts[i, j] is the number of times traj[t] == i and traj[t+1] == j.
    """
    if counts is None:
        counts = np.zeros((n_states, n_states), dtype=np.int64)
    # slice off last and first to get all transitions
    src = traj[:-1]
    dst = traj[1:]
    # increment counts[src[t], dst[t]] by 1 for each t
    np.add.at(counts, (src, dst), 1)
    return counts
