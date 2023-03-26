# Reinforcement learning with SNN on Frozen Lake
Pre-master project on reinforcement 
learning with spike timing neural networks in discrete observation space.
This project is made for Sofia University "Kliment Ohridski" 
from Borislav Markov.


# Intro
Neurobiology is increasingly gaining momentum in the 
world of artificial intelligence. 
There is a growing body of research into the functioning of 
nerve cells that has led to the creation of biologically valid spike 
timing models of neurons, as well as much knowledge about the 
structural organization and decision-making function of the mammalian brain. 
It has been proven that many decisions are made by the 
method of Reinforcement Learning.

The aim of the undergraduate project is to develop a model of a 
biologically based (spike timing) neural network using 
the NEST Simulator library, which is able to solve the 
optimization task of an agent passing through the known 
FrozenLake environment from the Gym package by means of 
reinforcement learning. The problem has discrete states 
of the environment and 4 possible actions of the agent.

The project includes a brief overview of the field of Spike 
Timing Neural Networks, a description of the theoretical formulation,
Python code using the NEST Simulator library, and an analysis
of the results. In the project, different parameters of the 
biologically similar neurons are tested and the solution is 
illustrated with appropriate visualizations and graphics 
accompanying the training process.

# The FrozenLake environment in the Gym
The Frozen Lake environment is part of the Text Games subpack. 
These are very simple games with visuals like text. 
They have a small number of discrete states and agent control 
is also discrete with a small number of actions. This package 
is designed for Reinforcement Learning training purposes. The 
goal is to compare different solutions in the same environment. 
The following figure shows an example visualization of this environment. 
We have a lake, a rectangular figure graphed into quadrants, for example 
4x4 with the codes of the Latin letters S,F,H,G.

| Visualisation                                               |                                                   Text representation |
|:------------------------------------------------------------|----------------------------------------------------------------------:|
| ![doc/frozen_lake_example.png](doc/frozen_lake_example.png) | ![doc/frozen_lake_example_text.png](doc/frozen_lake_example_text.png) |

Quadrant values are start-S, icy-F, hole-H, target-G. 
A man starts from position "S" and has to reach the final 
goal "G" by crossing icy sections "F". Some of the quadrants 
have “H” holes and there the current attempt fails. The 
reward is given as follows:
 * On reaching the end goal “G” reward of 1 point and the experiment ends.
 * Hitting hole “H” reward “0” and the experiment ends.
 * Hitting an icy section "F" reward "0".

There is a peculiarity of the environment that on icy 
sections the agent can slip and not go in the chosen direction, 
that is, we have slippage. Slippery depends on what settings 
the environment is launched with: is_slippery=False|True. 
Basic information about the environment is given in the following table.


| Action space      | Discrete(4)  |
|-------------------|--------------|
| Observation space | Discrete(16) |
| Python usage      | gym.make("FrozenLake-v0") |

The agent can move in four directions, each coded from 0 to 3 inclusive, namely: left-0, down-1, right-2, up-3.

# Environment

```shell
conda create --name nest-env -c conda-forge nest-simulator jupyterlab seaborn
conda activate nest-env
pip install -U scikit-learn
pip install gym[classic_control]==0.7.0
pip install pyglet==1.5.27
pip install swig
pip install gym[box2d]==0.7.0
```

# Run the application
Make sure you have conda environment 'nest-env'

```shell
conda activate nest-env
cd script/
python actor-critic-frozen-lake-nest.py -e 3x3 -s false -o outputs -c true -n 60
```

Options for `actor-critic-frozen-lake-nest.py` :
* `-h` :prints help message
* `-e <environment>` : This is predefined choice from `3x3` or `4x4`.
* `-s <slippery>`: Slippery, `true` or `false`, default `false`
* `-o <output_folder>` : where to save pictures and text files
* `-c <clean>` : boolean `true` or `false` whether to clean output folder, default `true`.
* `-n <max_number_episodes>` : int, maxmum number of episodes to run, default 60.

