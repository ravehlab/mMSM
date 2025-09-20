# README

This folder contains the test scripts for running the two spheres system and the alanine dipeptide system, as described in our accompanying manuscript (Nitskansky et al., 2025).

## Files

- `run_mmsm.py`: Generic mMSM-explore run script.
- `twosph_naive.py`: Runs a naive simulation for the two spheres system.
- `twosph_run.py`: Runs mMSM-explore on the two spheres system.
- `adp_naive.py`:  Runs a naive simulation for the alanine dipeptide system.
- `adp_run.py`: Runs mMSM-explore on the alanine dipeptide system.
- `hp35_run.py`: Runs mMSM-explore on the precomputed HP-35 contact-distance trajectory.

## Usage

Each script can be run independently. Example:

```bash
python runner.py --out output_file_path --runtime_ns 100 [--c path_to_previous_run]
```

Replace `runner.py` with the desired script (e.g., `twosph_run.py`, `adp_run.py`, etc.).  
The `--c` option can be used to specify the output from a previous run to continue from that point.

## Environment Setup

All scripts assume the **base environment** is installed. You can set it up using either `environment-base.yml` (for Conda) or `requirements-base.txt` (for pip), found in the root directory.

- Running the **two spheres system** requires installing the [Integrative Modeling Platform](https://integrativemodeling.org) (`imp`) on top of the base environment.  
- Running the **alanine dipeptide system** requires that [OpenMM](https://openmm.org) (`openmm`) is installed.

## Data requirement

To run `hp35_run.py`, you’ll need the 300 μs HP35 trajectory from D. E. Shaw Research, mapped to native contact distances by Nagel *et al.*  
Download from <https://github.com/moldyn/HP35-DESRES/> (file: `hp35.mindists2.bz2`).

**References**
- Piana, S.; Lindorff-Larsen, K.; Shaw, D. E. *Protein folding kinetics and thermodynamics from atomistic simulation*. **PNAS** (2012) 109(44):17845–17850. <https://doi.org/10.1073/pnas.1201811109>
- Nagel, D.; Sartore, S.; Stock, G. *Selecting features for Markov modeling: A case study on HP35*. **J. Chem. Theory Comput.** (2023) 19(11):3391–3405. <https://doi.org/10.1021/acs.jctc.3c00240>


