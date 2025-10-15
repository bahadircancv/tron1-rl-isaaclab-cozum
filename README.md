This is a repository to develop [Limx Dynamics](https://www.limxdynamics.com/en) training environment with custom goals and enhanced pathfinding.

## Installation steps
We will be following the [recommended way](https://isaac-sim.github.io/IsaacLab/release/2.1.0/source/setup/installation/pip_installation.html) for the Isaac Lab and Isaac Sim installations.

### Virtualenv
```zsh
pyenv virtualenv 3.10 limx4_5_venv
pyenv activate limx4_5_venv
python -m pip install --upgrade pip
```
Do not forget to activate the environment through the following steps.

### Isaac Sim (4.5.0)
Recommended way is to use pip installation:
```zsh
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
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
git checkout release/2.1.0 --force
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

### rsl_rl
```zsh
pip uninstall rsl-rl-lib
cd path/to/tron1-rl-isaaclab-cozum
pip install -e rsl_rl
```

## Running train.py
Go to your Isaac Lab root directory, then run:
```zsh
./isaaclab.sh -p path/to/tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py --task=Isaac-Limx-WF-Blind-Flat-v0 --headless
```

## Running play.py
If `logs` folder where trained model resides in, is located in Isaac Lab directory:
```zsh
 ./isaaclab.sh -p ../tron1-rl-isaaclab-cozum/scripts/rsl_rl/play.py --task=<task_name>
```
If it is in `tron1-rl-isaaclab-cozum`:
```zsh
python scripts/rsl_rl/play.py --task=<task_name>
```
should be used.

### with keyboard
```zsh
 ./isaaclab.sh -p ../tron1-rl-isaaclab-cozum/scripts/rsl_rl/play.py --task=<task_name> --keyboard --num_envs=1
```