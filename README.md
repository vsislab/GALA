# Supplementary materials for "Learning Versatile Quadruped Locomotion via A Single Generalist Policy Network"
## Project page: https://github.com/vsislab/GALA
### Author: Yanyun Chen, Ran Song, Jiapeng Sheng, Xing Fang, Wenhao Tan, Wei Zhang and Yibin Li


### This repository contains
* datas: all data presented in this work.
* env: the Aliengo environments implemented in pybullet gym.
* exp: the learned generalist policy network for Aliengo quadruped robot.
* model: the implemented neural model. 
* rl: the implemented rl algorythm. 
* the utility files.
* eval.py: evaluation of the generalist policy network.
 
### Dependencies
* miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04
* pybullet (https://pypi.org/project/pybullet/)
* torch (https://pytorch.org/)
* gym (https://github.com/openai/gym/)
* all dependencies are listed in requirements.txt
* to install all the dependencies in requirements.txt, run
```bash
  pip install -r requirements.txt
```
* Dockerfile is also available for building the environments

### Notes
* Some paths are hard-coded in _eval_controller.py_ and _aliengo_gym_env.py_. Be careful about them.
* This repository is not maintained anymore. If you have any question, please send emails to yy_chen@mail.sdu.edu.cn.
* The project can only be run after successful installation.

### Running the project
#### Evaluation of the generalist policy network
```bash
python eval.py --render --debug --name policy_dir_name --id id_index --mode mode_index
```
#### Optional parameter configuration
##### render: rendering the environment
##### debug: saving the debug data to the "debug" directory
##### name: the directory name in which the policy network saved 
##### traj: drawing the trajectories of the robot feet
##### max_time: the max time for evaluation till terminate (seconds)
##### video: recording a video for evaluation
##### img: saving the snapshot sequences to image files
##### push: applying external force to the quadruped robot
##### random: evaluating the generalist policy with randomized motor behaviours (commands), make sure that id has beem set to 3
##### id: id for different motor behaviours
- 1: rolling
- 2: recovery
- 3: low speed
- 4: high speed
- 5: lateral walk
- 6: height walk
- 7: width walk
##### mode: evaluating the generalist policy on traversing the training site or evaluation with randomized motor behaviours (commands). 
##### Make sure to set id to 3 (This is only useful for resetting the robot to an appropriate state)
- 1: traversing the training site
- 2: evaluation with randomized motor behaviours
- 0 (or other integer except 1 & 2): evaluation with the selected "id" motor behaviour

#### Example for evaluating the generalist policy in the training site with external force, report the debug data and save the snapshots for 50s
```bash
python eval.py --render --debug --name mc_sp --id 3 --mode 1 --max_time 50 --video --push 
```

#### Example for evaluating the generalist policy with randomized motor behaviours (commands) for 100s
```bash
python eval.py --render --debug --name mc_sp --id 3 --mode 2 --max_time 100 --video
```

#### Example for evaluating the generalist policy with specific motor behaviours (commands)  for 10s
```bash
python eval.py --render --traj --debug --name mc_sp --traj --id 2 --video --mode 0
```

#### Example for evaluating the generalist policy with randomized motor behaviours (commands) and external force
```bash
python eval.py --render --name mc_sp --id 3 --mode 2 --max_time 35 --video --push
```
