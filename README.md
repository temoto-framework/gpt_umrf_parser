# Prompt Experiments
contact: slwanna@utexas.edu 

## Setup Instructions

1. Create a conda environment and install packages

```$ conda env create -f environment.yaml --name prompt_experiments```

```$ conda activate prompt_experiments```

```$ pip install -r requirements.txt```

### To Run Experiments

The experiments are located in the src folder. To run a specific experiment, e.g., experiment 2

1. Adjust the sys_config.yaml file to point to the correct experiment configuration file.

2. Run the following command.

```$ python src/experiment_2/ood_test.py --config sys_configs.yaml```
