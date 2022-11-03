# -*- coding: utf-8 -*-
import argparse
import shutil
# import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import exists, join
import pandas as pd
import torch
from PIL import Image

from env import AliengoGymEnv, AliengoGymEnvWrapper
from env.tasks import load_task_cls
from env.randomizers import ParamRandomizerFromConfig, PushRandomizer
from env.randomizers import TerrainRandomizer, TerrainParam, TerrainType
from env.utils import seed_all
from env.utils.obstacle import create_integrate
from model import load_agent, load_actor, load_critic
from utils import read_param


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='experiment name')
    parser.add_argument('--epoch', type=int, default=1, help='the total eval epochs')
    parser.add_argument('--id', type=int, default=2, help='the project id')
    parser.add_argument('--mode', type=int, default=2, help='running mode')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--push', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max_time', type=float, default=30., help='the time to run policy')
    parser.add_argument('--sim', type=str, default=None, help='sim epoch')
    parser.add_argument('--deploy', type=str, default=None, choices=['ori', 'gnl', 'aux'], help='deploy config')
    parser.add_argument('--recon', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--video', action='store_true', default=False)
    parser.add_argument('--img', action='store_true', default=False)
    parser.add_argument('--traj', action='store_true', default=False)
    parser.add_argument('--timing', action='store_true', default=False)
    parser.add_argument('--filter', action='store_true', default=False)
    args = parser.parse_args()
    args.render = args.render or args.traj or args.video
    return args


def eval(args):
    global critic
    seed_all(args.seed)
    controller_dir = join('exp', args.name)
    model_dir = join(controller_dir, 'model')
    debug_dir = join(controller_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    video_path = join(debug_dir, 'aliengo.mp4') if args.video else None
    timing_path = join(debug_dir, 'timing.json') if args.timing else None
    imgs_dir = join(debug_dir, 'imgs')
    img_dir = join(imgs_dir, 'img')
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)
    img_dir1 = join(imgs_dir, 'img1')
    if os.path.exists(img_dir1):
        shutil.rmtree(img_dir1)
    os.makedirs(img_dir1, exist_ok=True)
    img_dir2 = join(imgs_dir, 'img2')
    if os.path.exists(img_dir2):
        shutil.rmtree(img_dir2)
    os.makedirs(img_dir2, exist_ok=True)
    params = read_param(join(controller_dir, 'cfg.yaml'))
    if args.deploy:
        policy_path = f'deploy_policy_{args.deploy}.pth'
    elif args.sim is not None:
        policy_path = f'all/sim_policy_{args.sim}.pth'
    else:
        policy_path = 'sim_policy.pth'
    policy_path = join(model_dir, policy_path)
    assert exists(policy_path), policy_path
    policy_model = torch.load(policy_path, map_location='cpu')

    project_id = args.id
    project_name, project = None, None
    for k, v in params['playground'].items():
        if k.split(':')[0] == str(project_id):
            project_name, project = k, v
            break
    assert project is not None
    terrain_params = {
        'Flat': TerrainParam(type=TerrainType.Flat, size=(50, 50)),
        'Hill': TerrainParam(type=TerrainType.Hill, size=(30, 30), amplitude=0., roughness=0.00),
    }
    env = AliengoGymEnv(
        terrain_randomizer=TerrainRandomizer(terrain_params[project['terrain_type']]),
        time_step=1 / 1000,
        action_repeat=10,
        seed=0,
        self_collision_enabled=True,
        remove_default_link_damping=True,
        kp=[300, 300, 300],
        kd=[4, 4, 4],
        motor_damping=0.01,
        motor_friction=0.02,
        foot_friction=1.5,
        foot_restitution=0.1,
        terrain_friction=1.3,
        terrain_restitution=0.1,
        observation_noise={'velocity': 0.08, 'rpy': 0.08, 'rpy_rate': 0.3, 'motor_position': 0.01, 'motor_velocity': 1.2},
        latency=0.01,
        max_time=args.max_time,
        param_randomizer=ParamRandomizerFromConfig(enabled=False),
        push_randomizer=PushRandomizer(render=True,
                                       push_strength_ratio=2 if args.push else 0.,
                                       start_step=3020 if args.mode == 1 else 300,
                                       interval_step=500,
                                       duration_step=50,
                                       horizontal_force=120,
                                       vertical_force=50,
                                       ),
        training=False,
        mode=args.mode,
        # on_rack=True,
        render=args.render,
        video_path=video_path,
        timing_path=timing_path,
        draw_foot_traj=args.traj)
    if args.mode == 1:
        create_integrate(env)
    task = load_task_cls(project['task'])(env)
    env = AliengoGymEnvWrapper(env, task, debug=args.debug)

    policy_params = params['policy' if args.deploy is None else 'deploy_policy']
    if env.is_multi_critic_task:
        policy_params.update({
            'task_dim': env.task.task_dim,
            'task_adv_coef': env.task.task_adv_coef
        })
    actor = load_actor(policy_params).eval()
    actor.load_state_dict(policy_model['actor'], strict=args.deploy is None)
    agent = load_agent(actor)
    if args.deploy is None:
        critic = load_critic(policy_params)
        critic.load_state_dict(policy_model['critic'])

    task_dim = env.task.task_dim
    for epo in range(args.epoch):
        result = {'len': 0, 'rew': 0,
                  'robot_encoder': {'privileged_code': [], 'robot_code': [], 'privileged_info': [], 'recon_mu': [], 'recon_sigma': []},
                  'contact_detector': {'real': [], 'fake': []}}
        task_id = []
        critic_value = [list() for _ in range(task_dim)]
        obs = env.reset()
        agent.reset(obs)
        count = 0
        while True:
            act = agent(obs)
            obs_next, rew, done, info = env.step(act)
            if args.img:
                if count % 4 == 0:
                    im = Image.fromarray(env.render(mode="fixed", dyaw=0))
                    im.save(join(img_dir, f'{(int)(count / 2)}.jpeg'))
                    im = Image.fromarray(env.render(mode="followed", dyaw=-135))
                    im.save(join(img_dir1, f'{(int)(count / 2)}.jpeg'))
                    im = Image.fromarray(env.render(mode="followed", dyaw=90))
                    im.save(join(img_dir2, f'{(int)(count / 2)}.jpeg'))
                count += 1
            if env.is_multi_critic_task:
                rew = rew
                if args.deploy is None:
                    task_id.append(info['task_id'])
                    for i in range(task_dim):
                        critic_value[i].append(critic(obs[None], net_id=i)[0][0].detach().cpu().numpy())
            result['rew'] += rew
            result['len'] += 1
            if done:
                env.save_debug_report(join(debug_dir, f'debug.xlsx'))
                break
            obs = obs_next

        print(f"Epoch #{epo}: "
              f"len {result['len']}",
              f"rew {result['rew']:.2f}",
              f"suc {info['success']}",
              f"time_limit {info['time_limit']}",
              f"space_limit {info['space_limit']}",
              f"time {env.time:.1f}/{args.max_time:.0f}",
              sep='  ')
        # if env.is_multi_critic_task:
        #     indexes = []
        #     for i in range(task_dim):
        #         v = critic_value[i]
        #         x = np.arange(len(v))
        #         plt.plot(x, v, c='red')
        #         plt.plot(x, np.ma.masked_where(np.array(task_id) == i, v), c='black')
        #         title = f'local_{i}'
        #         plt.title(title)
        #         # plt.show()
        #         indexes.append(title)
        #     with pd.ExcelWriter(join(debug_dir, 'critic.xlsx')) as f:
        #         pd.DataFrame(np.asarray(critic_value).transpose(), columns=indexes).to_excel(f, 'value', index=False)
    env.close()


if __name__ == '__main__':
    args = parse_args()
    eval(args)
