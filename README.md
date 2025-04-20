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
*(In preparation)*

## Repository Structure

| Directory/File                | Description                                         |
|-------------------------------|-----------------------------------------------------|
| `mmsm/`                       | Core implementation of the mMSM method              |
| `implementations/alaninedp/`  | Implementation for the **alanine dipeptide** system |
| `implementations/twospheres/` | Implementation for the **two-sphere** system        |
| `experiments/runners/`        | Scripts for running simulations and generating results         |
| `tutorial.ipynb`              | Jupyter Notebook tutorial demonstrating usage                    |
| `README.md`                   | This file                                           |

## Quick Start
Check out the **Jupyter Notebook tutorial**:

[tutorial.ipynb](tutorial.ipynb)

## How to Cite

```bibtex
@article{nitskansky-clein-raveh2025,
  author  = {Nir Nitskansky and Kessem Clein and Barak Raveh},
  title   = {Building Multiscale Markov State Models by Systematic Mapping of Temporal Communities},
  note    = {Manuscript in preparation, 2025}
}



