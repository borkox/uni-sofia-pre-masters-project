# Reinforcement learning with SNN on Frozen Lake
Pre-master project on reinforcement 
learning with spike timing neural networks in discrete observation space.
This project is made for Sofia University "Kliment Ohridski" 
from Borislav Markov.

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
python ./actor-critic-frozen-lake-nest.py
```