# Multiscale Markov State Models

## Overview

This repository contains the implementation of the **Multiscale Markov State Model (mMSM)**, a data structure for 
capturing the hierarchical temporal organization of dynamical systems. The accompanying algorithm, 
**mMSM-explore**, constructs an mMSM from input trajectories of system configurations.

mMSM is designed to:
- Represent system dynamics across **multiple timescales simultaneously**.
- Identify metastabe states across these different timescales.
- Improve the efficiency of **state-space exploration** compared to conventional MSM-based approaches.

While this work primarily demonstrates the method for **molecular dynamics (MD) simulations**, mMSM is in 
principle applicable to other types of simulations (e.g., Monte-Carlo).

For details, see the paper:  
**Title:** *Building Multiscale Markov State Models by Systematic Mapping of Temporal Communities*  
**Author:** Nir Nitskansky, Kessem Clein, Barak Raveh  
*(Manuscript under review)*

## Repository Structure

| Directory/File                | Description                                            |
|-------------------------------|--------------------------------------------------------|
| `mmsm/`                       | Core implementation of the mMSM method                 |
| `implementations/alaninedp/`  | Implementation for the **alanine dipeptide** system    |
| `implementations/twospheres/` | Implementation for the **two-sphere** system           |
| `implementations/hp35/`       | Implementation for the **HP-35** system                |
| `experiments/runners/`        | Scripts for running simulations and generating results |
| `tutorial.ipynb`              | Jupyter Notebook tutorial demonstrating usage          |
| `README.md`                   | This file                                              |

## Quick Start

To get started, install the required dependencies using **either** Conda or pip:

**Using Conda (recommended):**

```
conda env create -f environment-notebook.yml
conda activate mmsm_nb
```

**Using pip:**

```
python -m venv venv
source venv/bin/activate
pip install -r requirements-notebook.txt
```

> **Note:** Some packages (e.g., `openmm`, `leidenalg`) may be easier to install via Conda due to system dependencies.

If you only wish to use the core method (without running the notebook), use the `environment-base.yml` or `requirements-base.txt` instead.

Check out the **Jupyter Notebook tutorial**:  
[tutorial.ipynb](tutorial.ipynb)

## How to Cite

```bibtex
@article{nitskansky-clein-raveh2025,
  author  = {Nir Nitskansky and Kessem Clein and Barak Raveh},
  title   = {Building Multiscale Markov State Models by Systematic Mapping of Temporal Communities},
  note    = {Manuscript under review, 2025}
}



