# README

This folder contains the test scripts for running the two spheres system and the alanine dipeptide system, as described in our accompanying manuscript (Nitskansky et al., 2025).

## Files

- `run_mmsm.py`: Generic mMSM-explore run script.
- `twosph_naive.py`: Runs a naive simulation for the two spheres system.
- `twosph_run.py`: Runs mMSM-explore on the two spheres system.
- `adp_naive.py`:  Runs a naive simulation for the alanine dipeptide system.
- `adp_run.py`: Runs mMSM-explore on the alanine dipeptide system.

## Usage

Each script can be run independently. Example:

```bash
python runner.py --out output_file_path --runtime_ns 100 [--c path_to_previous_run]
```

Replace `runner.py` with the desired script (e.g., `twosph_run.py`, `adp_run.py`, etc.).  
The `--c` option can be used to specify the output from a previous run to continue from that point.
