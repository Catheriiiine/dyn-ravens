# Using new version

import os
import cv2
import pickle
import numpy as np
from environments.environment import Environment, EnvironmentNoRotationsWithHeightmap, ContinuousEnvironment
import multiprocessing as mp
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
from scipy.special import softmax

import pkgutil
import sys
import tempfile
import time

import gym
import numpy as np
from ravens.tasks import cameras
from ravens.tasks.grippers import Spatula
from ravens.utils import pybullet_utils
from ravens.utils import utils
import torch 
from planners import PlannerGD
from models.res_regressor import MPCResRgrNoPool
from utils.utils import fps_np, depth2fgpcd, downsample_pcd, recenter, fps, load_yaml
from ravens.environments.flex_rewards import config_reward_ptcl

# utils
from utils.utils import load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed, pcd2pix, gen_goal_shape, gen_subgoal, gt_rewards, gt_rewards_norm_by_sum, lighten_img, rmbg
from models.gnn_dyn import PropNetDiffDenModel

def predict(env, 
            subgoal,
            model_dy,
            init_pos = None,
            n_mpc=1,
            n_look_ahead=1,
            n_sample=100,
            n_update_iter=100,
            gd_loop=1,
        #   particle_r=0.06,
            particle_num=50,
            mpc_type='GD',
            funnel_dist=None,
            action_seq_mpc_init=None,
            action_label_seq_mpc_init=None,
            time_lim=float('inf'),
            auto_particle_r=False):
    
    
    
    assert type(subgoal) == np.ndarray
    # assert subgoal.shape == (env.screenHeight, env.screenWidth)
    # planner
    if mpc_type == 'GD':
        env.planner = PlannerGD(env.config, env)
    else:
        raise NotImplementedError
    action_lower_lim = action_upper_lim = np.zeros(4) # DEPRECATED, should be of no use
    reward_params = (env.get_cam_extrinsics(), env.get_cam_params(), env.global_scale)

    particle_den_seq = []
    if auto_particle_r:
        res_rgr_folder = env.config['mpc']['res_sel']['model_folder']
        res_rgr_folder = os.path.join('checkpoints/dyn_sweep', res_rgr_folder)
        res_rgr = MPCResRgrNoPool(env.config)
        # res_rgr = MPCResCls(env.config)
        if env.config['mpc']['res_sel']['iter_num'] == -1:
            res_rgr.load_state_dict(torch.load(os.path.join(res_rgr_folder, 'net_best_dy_state_dict.pth')))
        else:
            res_rgr.load_state_dict(torch.load(os.path.join(res_rgr_folder, 'net_dy_iter_%d_state_dict.pth' % env.config['mpc']['res_sel']['iter_num'])))
        res_rgr = res_rgr.cuda()
        
        # construct res_rgr input
        # first channel is the foreground mask
        fg_mask = (env.render()[..., -1] / env.global_scale < 0.599/0.8).astype(np.float32)
        # second channel is the goal mask
        subgoal_mask = (subgoal < 0.5).astype(np.float32)
        particle_num = res_rgr.infer_param(fg_mask, subgoal_mask)
        print('particle_num: %d' % particle_num)
        # particle_r = np.sqrt(1.0 / particle_den)
        # particle_den_seq.append(particle_den)
        particle_den_seq.append(particle_num)
    # particle_den = np.array([1 / (particle_r * particle_r)])

    # return values
    rewards = np.zeros(n_mpc+1)
    # from 5 to 3
    raw_obs = np.zeros((n_mpc+1, env.screenHeight, env.screenWidth, 5))
    # states = np.zeros((n_mpc+1, particle_num, 3))
    states = []
    actions = np.zeros((n_mpc, env.act_dim))
    # states_pred = np.zeros((n_mpc, particle_num, 3))
    states_pred = []
    rew_means = np.zeros((n_mpc, 1, n_update_iter * gd_loop))
    rew_stds = np.zeros((n_mpc, 1, n_update_iter * gd_loop))

    if init_pos is not None:
        env.set_positions(init_pos)
    obs_cur = env.render()
    
    # print("obs_cur")
    # print(obs_cur.shape)
    # print(raw_obs.shape)
    # print("raw_obs[0]")
    # print(raw_obs[0].shape)
    raw_obs[0] = obs_cur
    # note: in world cdn
    obs_cur, particle_r = env.obs2ptcl_fixed_num_batch(obs_cur, particle_num, batch_size=30)
    
    # for i in obs_cur:
    #    for j in i:
    #       j[0] = j[0] * 27 - 13.5
    #       j[1] = j[1] * 27
    # print("obs_cur:", obs_cur[0])
    # print(obs_cur[1])
    # print("obs_cur shape:", obs_cur.shape)
                
    
    particle_den = np.array([1 / (particle_r * particle_r)])[0]
    print('particle_den:', particle_den)
    print('particle_num:', particle_num)
    # obs_cur = env.obs2ptcl(obs_cur, particle_r)
    if action_seq_mpc_init is None:
        action_seq_mpc_init, action_label_seq_mpc_init = env.sample_action(n_mpc)
    subgoal_tensor = torch.from_numpy(subgoal).float().cuda().reshape(env.screenHeight, env.screenWidth)
    subgoal_coor_tensor = torch.flip((subgoal_tensor < 0.5).nonzero(), dims=(1,)).cuda().float()
    subgoal_coor_np, _ = fps_np(subgoal_coor_tensor.detach().cpu().numpy(), min(particle_num * 5, subgoal_coor_tensor.shape[0]))
    subgoal_coor_tensor = torch.from_numpy(subgoal_coor_np).float().cuda()
    # rewards[0] = config_reward_ptcl(torch.from_numpy(obs_cur).float().cuda().reshape(-1, particle_num, 3),
    #                                 subgoal_tensor,
    #                                 cam_params=env.get_cam_params(),
    #                                 goal_coor=subgoal_coor_tensor,
    #                                 normalize=True)[0].item()
    states.append(obs_cur[0])
    total_time = 0.0
    rollout_time = 0.0
    optim_time = 0.0
    iter_num = 0
    for i in range(n_mpc):
        attr_cur = np.zeros((obs_cur.shape[0], particle_num))
        # init_p = env.get_positions()
        # print('mpc iter: {}'.format(i))
        # print('action_seq_mpc_init: {}'.format(action_seq_mpc_init.shape))
        # print('action_label_seq_mpc_init: {}'.format(action_label_seq_mpc_init))
        # print(f'act_Seq:{action_seq_mpc_init[:n_look_ahead]}')
        traj_opt_out = env.planner.trajectory_optimization_ptcl_multi_traj(
                        obs_cur,
                        particle_den,
                        attr_cur,
                        obs_goal=subgoal,
                        model_dy=model_dy,
                        act_seq=action_seq_mpc_init[:n_look_ahead],
                        act_label_seq=action_label_seq_mpc_init[:n_look_ahead] if action_label_seq_mpc_init is not None else None,
                        n_sample=n_sample,
                        n_look_ahead=min(n_look_ahead, n_mpc - i),
                        n_update_iter=n_update_iter,
                        action_lower_lim=action_lower_lim,
                        action_upper_lim=action_upper_lim,
                        use_gpu=True,
                        rollout_best_action_sequence=True,
                        reward_params=reward_params,
                        gd_loop=gd_loop,
                        time_lim=time_lim,)
        action_seq_mpc = traj_opt_out['action_sequence']
        next_r = traj_opt_out['next_r']
        obs_pred = traj_opt_out['observation_sequence'][0]
        rew_mean = traj_opt_out['rew_mean']
        rew_std = traj_opt_out['rew_std']
        iter_num += traj_opt_out['iter_num']
        
        print('mpc_step:', i)
        print('action:', action_seq_mpc[0])
        print('actions:', action_seq_mpc)
        
        # change: comment out step and res_rgr

        # obs_cur = env.step(action_seq_mpc[0])[0]['color'][0]
        # obs_cur = env.step(action_seq_mpc[0])
        # print("obs_cur:", obs_cur)
        # if obs_cur is None:
        #     raise Exception('sim exploded')

        # if auto_particle_r:
        #     # construct res_rgr input
        #     # first channel is the foreground mask
        #     fg_mask = (env.render()[..., -1] / env.global_scale < 0.599/0.8).astype(np.float32)
        #     # second channel is the goal mask
        #     subgoal_mask = (subgoal < 0.5).astype(np.float32)
        #     particle_num = res_rgr.infer_param(fg_mask, subgoal_mask)
        #     # particle_den = np.array([res_rgr(res_rgr_input[None, ...]).item() * 4000.0])
        #     # particle_r = np.sqrt(1.0 / particle_den)
        #     # particle_den_seq.append(particle_den)
        #     particle_den_seq.append(particle_num)

        # obs_cur = env.render()
        
        # raw_obs[i + 1] = obs_cur
        # obs_cur, particle_r = env.obs2ptcl_fixed_num_batch(obs_cur, particle_num, batch_size=30)
        # particle_den = np.array([1 / (particle_r ** 2)])[0]
        # print('particle_den:', particle_den)
        # print('particle_num:', particle_num)
        
        # states.append(obs_cur[0])
        actions[i] = action_seq_mpc[0]

        # subgoal_coor_np, _ = fps_np(subgoal_coor_tensor.detach().cpu().numpy(), min(obs_cur.shape[0] * 5, subgoal_coor_tensor.shape[0]))
        # subgoal_coor_tensor = torch.from_numpy(subgoal_coor_np).float().cuda()
        # rewards[i + 1] = config_reward_ptcl(torch.from_numpy(obs_cur).float().cuda().reshape(-1, particle_num, 3),
        #                                     subgoal_tensor,
        #                                     cam_params=env.get_cam_params(),
        #                                     goal_coor=subgoal_coor_tensor,
        #                                     normalize=True)[0].item()
        total_time += traj_opt_out['times']['total_time']
        rollout_time += traj_opt_out['times']['rollout_time']
        optim_time += traj_opt_out['times']['optim_time']
        states_pred.append(obs_pred)

        rew_means[i] = rew_mean
        rew_stds[i] = rew_std
        # action_seq_mpc_init = action_seq_mpc[1:]

        # print("action_seq_mpc_init.shape[0]: ", action_seq_mpc_init.shape[0])
        
        if action_seq_mpc_init.shape[0] > 1:
            action_seq_mpc_init = np.concatenate((traj_opt_out['action_full'][1:], action_seq_mpc_init[n_look_ahead:]), axis=0)
            
            if action_label_seq_mpc_init is not None:
                action_label_seq_mpc_init = action_label_seq_mpc_init[1:]
                

        # print('rewards: {}'.format(rewards))
        # print()
    return {
            # 'rewards': rewards,
            'raw_obs': raw_obs,
            'states': states,
            'actions': actions,
            # 'states_pred': states_pred,
            # 'rew_means': rew_means,
            # 'rew_stds': rew_stds,
            # 'total_time': total_time,
            # 'rollout_time': rollout_time,
            # 'optim_time': optim_time,
            # 'iter_num': iter_num,
            }

    
    # return subg_output
if __name__ == "__main__":
    predict()
