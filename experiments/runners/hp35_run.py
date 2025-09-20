import numpy as np
import pickle
import argparse
import openmm.app as ommapp
import openmm.unit as ommunit
from implementations.hp35.hp35_sim import HP35SamplerPreexisting
from implementations.hp35.hp35_discretizer import HP35DiscretizerPreexisting
from mmsm.mmsm_config import mMSMConfig
from experiments.runners.run_utils import run_hmsm

def run_preexisting(args):
    print_tree_info = True
    runtime_ns = args.runtime_ns
    f_name = args.out
    stats_fn = None
    save_interval = 15
    time_mult = 1e-3

    sampler = HP35SamplerPreexisting(dihs_traj_path=PATH_TRAJ)
    # discretizer = HP35DiscretizerPreexisting(cutoff=1.5, representative_sample_size=100000) # PCA 5
    # discretizer = HP35DiscretizerPreexisting(cutoff=5, representative_sample_size=100000) # NO PCA
    discretizer = HP35DiscretizerPreexisting(cutoff=3.5, representative_sample_size=100000, pca=True) # PCA 10
    cnfg = mMSMConfig(n_trajectories=1, trajectory_len=10**5, max_height=-1, partition_diameter_threshold=8,
                      count_stride=500)
    # pdb = ommapp.PDBFile(PATH_PDB)
    # start_coords = np.array(pdb.positions.value_in_unit(ommunit.nanometer)).copy()
    # start_state = np.array([start_coords, np.zeros_like(start_coords)])
    start_state = np.load("../../hp35/runfiles/dummy_start_state.npy")

    run_hmsm(sampler, discretizer, x_init=start_state, mmsmconfig=cnfg, f_name=f_name, runtime_ns=runtime_ns,
             print_tree_info=print_tree_info, stats_fn=stats_fn, save_interval=save_interval, time_mult=time_mult)



if __name__ == '__main__':
    PATH_TRAJ = "PATH-TO/hp35.mindists2"


    parser = argparse.ArgumentParser()
    # For the full DE Shaw traj, set run time to 310000 ns
    parser.add_argument('--runtime_ns', type=int)
    parser.add_argument('--out', type=str)
    parser.add_argument('--c', type=str, required=False)
    cargs = parser.parse_args()

    run_preexisting(cargs)