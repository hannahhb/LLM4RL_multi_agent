# [Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373)

## Purpose
This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper, [Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373). This fork is adding multi agent experiment testing via the Multi Particulate environment from PettingZoo. A research report for the multi-agent implementation details and progress made as of 06/2024 can be found [here](report.pdf). 


## Running experiments
### Setup the LLMs

Download the models from huggingface website (quantised being used in current implementation) and correct the path in planner.py

Models that have been tested to work on MacBook Pro M3 environment:
Vicuna 13B v1.5 GGUF - https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF
Mistral 7B Instruct - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2


### Train and evaluate the models
Any algorithm can be run from the main.py entry point.

To train on a SimpleDoorKey environment,

```bash
python main.py train --task SimpleDoorKey --savedir train --n_itr 4000 --use_teacher_policy
```

can go for ~10k for more comprehensive results. 


For pure RL implementation,

```bash
python main.py train --task SimpleDoorKey --savedir train --n_itr 4000
```


<!--to train with given query result from LLM as teacher,

```bash
python main.py train --task SimpleDoorKey --savedir train --offline_planner
```-->

To evaluate the trained model,

```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval
```

To evaluate the LLM-based teacher baseline,
```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval --eval_teacher
```

## Logging details 
Tensorboard logging is enabled by default for all algorithms. The logger expects that you supply an argument named ```logdir```, containing the root directory you want to store your logfiles

The resulting directory tree would look something like this:
```
log/                         # directory with all of the saved models and tensorboard 
└── ppo                                 # algorithm name
    └── simpledoorkey                   # environment name
        └── save_name                   # unique save name 
            ├── acmodel.pt              # actor and critic network for algo
            ├── events.out.tfevents     # tensorboard binary file
            └── config.json             # readable hyperparameters for this run
```

Using tensorboard makes it easy to compare experiments and resume training later on.

To see live training progress

Run ```$ tensorboard --logdir=log``` then navigate to ```http://localhost:6006/``` in your browser

## Acknowledgements
This work is is adapted from 
 [our work](https://arxiv.org/abs/2311.13373) 
