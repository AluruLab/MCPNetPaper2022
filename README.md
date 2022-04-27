# MCPNetPaper2022

This repository contains the scripts used to generate the tables in the paper.


For completeness, it includes the MCPNet code as a git submodule.  To retrieve the code, please call from the source directory

`git submodule update --init --recursive`

Please see https://github.com/AluruLab/MCPNet for more information about building and running the code.

## Script organization

The scripts directory include python, R, and bash shell scripts.

- directory `simulate_with_noise`: contains scripts for generating the simulated yeast data and injecting random noise.

### For MCPNet

- directory `eval_params`:  contains scripts for parameter evaluations for Supplemental Material S1.2

- `mcpnet_simulated_eval.sh`: slurm script to run mcpnet for all simulated yeast, noisy or noise-free, datasets


- `mcpnet_arabidopsis_eval.pbs`: pbs script to run mcpnet for athaliana datasets

- `mcpnet_yeast_eval.pbs` and `mcpnet_yeast_eval_1core.pbs`: pbs scripts to run mcpnet for real yeast dataset.


### For Existing Tools

- directory `ath_json`: contains parameters for evaluating AUPR for all athaliana data sets and all existing tools.

- `athaliana_auc_pr.py`: computes the AUPR for all athaliana data sets and all existing tools.

- `simulated_input.json` and `simulated_roc_pr.py`: computes the AUPR for simulated noisy yeast data for all existing tools.

- `yeast_input.json` and `yeast_ath_roc_pr.py`: computes the AUPR for real yeast and the *development* athaliana datasets for all existing tools.