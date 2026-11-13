This is a repository to develop [Limx Dynamics](https://www.limxdynamics.com/en) training environment with custom goals and enhanced pathfinding.

## Installation steps
We will be following the [recommended way](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) for the Isaac Lab and Isaac Sim installations.

### Virtualenv
```zsh
pyenv virtualenv 3.11 limx_venv
pyenv activate limx_venv
python -m pip install --upgrade pip
```
Do not forget to activate the environment through the following steps.

### Isaac Sim (5.0.0)
Recommended way is to use pip installation:
```zsh
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
```

#### Verification
```zsh
isaacsim
```
The initial run will be downloading extra content so it takes a while.

### Isaac Lab (main)
Clone github repository and run installation script.
```zsh
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

#### Verification
Following command should start an empty world and display a black viewport.
```zsh
python scripts/tutorials/00_sim/create_empty.py
```
Running an example training to see if everything works:
```zsh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```
The training should create `logs/rsl_rl/anymal_c_rough/<timestamp>` in IsaacLab folder. And the `.pt` file can be found under that directory.

### bipedal_locomotion
Running `train.py` script requires `bidepal_locomotion` package.
```zsh
cd exts
pip install -e bipedal_locomotion
```

## Running train.py
Go to your Isaac Lab root directory, then run:
```zsh
./isaaclab.sh -p path/to/tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py --task=Isaac-Limx-WF-Blind-Flat-v0 --headless
```

## Running play.py