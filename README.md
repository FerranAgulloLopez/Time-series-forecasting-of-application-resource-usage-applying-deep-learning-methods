# Time series forecasting of application resource usage applying deep learning methods

_This component was created as a result of my Masters thesis in the MAI Masters degree._

## Introduction

This repostiory contains the code for replicating the experiments of the Masters thesis. It also contains the scripts used to create the charts for both datasets.

## Files structure

- app: directory containing the code for replicating the experiments
- configs: directory containing the configurations for every experiment
- data_scripts: directory containing the code for creating the carts for both datasets
- launcer_*.sh: scripts to run experiments on the server

### How to install

- Create an environment with python3.9
- Install the python packages inside the requirements.txt file

### How to run

- To train and evaluate models:
  - variables: 
    - output_path: directory to store the outputs of the experiments
    - config_path: experiment configuration to use
  - command: PYTHONPATH=$PYTHONPATH:/.. python3 app/train_test_model.py --output output_path --config config_path
- To visualize data:
  - variables: 
    - output_path: directory to store the outputs of the process
    - config_path: process configuration to use
  - command: PYTHONPATH=$PYTHONPATH:/.. python3 app/data_visualization.py --output output_path --config config_path
