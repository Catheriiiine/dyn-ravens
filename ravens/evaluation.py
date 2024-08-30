import os
import pickle

from absl import app
from absl import flags
import numpy as np
from ravens import agents
from ravens import dataset
from ravens import tasks
from ravens.environments.environment import Environment
import tensorflow as tf
from visualize_mpc import predict
import os
import pdb

from utils.utils import load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed, pcd2pix, gen_goal_shape, gen_subgoal, gt_rewards, gt_rewards_norm_by_sum, lighten_img, rmbg
from models.gnn_dyn import PropNetDiffDenModel
import cv2
import pickle
from environments.environment import Environment, EnvironmentNoRotationsWithHeightmap, ContinuousEnvironment
import multiprocessing as mp
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
from scipy.special import softmax
from ravens.tasks import cameras
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('assets_root', './assets/', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'hanoi', '')
flags.DEFINE_string('agent', 'transporter', '')
flags.DEFINE_integer('n_demos', 100, '')
flags.DEFINE_integer('n_steps', 40000, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  print(gpus)
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Logical GPUs:", logical_gpus)

  # Configure how much GPU to use (in Gigabytes).
  if FLAGS.gpu_limit is not None:
    mem_limit = 1024 * FLAGS.gpu_limit
    dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
    cfg.set_virtual_device_configuration(gpus[0], dev_cfg)
    

  # Initialize environment and task.
  env = Environment(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
  task = tasks.names[FLAGS.task]()
  task.mode = 'test'

  # Load test dataset.
  ds = dataset.Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-test'))

  # change: add
  config = load_yaml("ravens/config/config.yaml")
    
  model_folder = config['mpc']['model_folder']
  model_iter = config['mpc']['iter_num']
  n_mpc = config['mpc']['n_mpc']
  n_look_ahead = config['mpc']['n_look_ahead']
  n_sample = config['mpc']['n_sample']
  n_update_iter = config['mpc']['n_update_iter']
  gd_loop = config['mpc']['gd_loop']
  mpc_type = config['mpc']['mpc_type']
  task_type = config['mpc']['task']['type']
  
  model_root = 'checkpoints/dyn_sweep/'
  model_folder = os.path.join(model_root, model_folder)
  GNN_single_model = PropNetDiffDenModel(config, True)
  
  if model_iter == -1:
      GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_best.pth'), strict=False)
  else:
      GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_epoch_0_iter_{model_iter}.pth'), strict=False)
  GNN_single_model = GNN_single_model.cuda()
  
  # change: 480, 640 w>h
  screenWidth = 640
  screenHeight = 480

  # Run testing for each training run.
  for train_run in range(FLAGS.n_runs):

    name = f'net_epoch_0_iter_1000-{train_run}'
    
    action_seq_mpc_init = np.load('ravens/init_actions/init_action_'+ str(n_sample) +'.npy')[np.newaxis, ...] # [1, 50, 4]
    action_label_seq_mpc_init = np.zeros(1)

    # Initialize agent.
    np.random.seed(train_run)
    tf.random.set_seed(train_run)
    # agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir)

    # Load trained agent.
    # if FLAGS.n_steps > 0:
    #   agent.load(FLAGS.n_steps)

    # Run testing and save total rewards with last transition info.
    results = []
    
    intrinsics = np.array([
    [450., 0, 320.],
    [0, 450., 240.],
    [0, 0, 1]
    ])
    
    for i in range(ds.n_episodes):
      print(f'Test: {i + 1}/{ds.n_episodes}')
      episode, seed = ds.load(i)
      # goal = episode[-1]
      total_reward = 0
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      obs = env.reset()
      goal = task.goals[-1][2][0][0]
      goal1 = task.goals[-1][2][0]
      # fx, fy, cx, cy = 450, 450, 320, 240
      # goal = ((goal[0] - 0.5) * 27, goal[1] * 27)

      # change: change from world cdn to pixel
      world_coords_2d_simplified = np.array([goal[0], goal[1]])  

      # Convert world coordinates to homogeneous camera coordinates
      camera_coords_homogeneous_simplified = np.array([world_coords_2d_simplified[0], world_coords_2d_simplified[1], 1])

      # Convert to pixel coordinates using the intrinsic matrix
      pixel_coords_simplified = intrinsics @ camera_coords_homogeneous_simplified

      # Normalize to get pixel coordinates
      pixel_coords_simplified = pixel_coords_simplified[:2] / pixel_coords_simplified[2]
      # pixel_coords_simplified = [0, 0]
      # pixel_coords_simplified[0] = goal[0] * fx / goal[2] + cx
      # pixel_coords_simplified[1] = goal[1] * fy / goal[2] + cy
      
      # add subgoal
      if task_type == 'target_control':
        # goal_row = config['mpc']['task']['goal_row']
        # goal_col = config['mpc']['task']['goal_col']
        # goal_r = config['mpc']['task']['goal_r']
        # change: goal to zone
        
        # goal_row = goal[1]*640 + 320
        # goal_col = goal[0]*480 + 120
        # goal_r = 0.06*480
        goal_row = pixel_coords_simplified[0] 
        goal_col = pixel_coords_simplified[1] 
        print("goal: ", goal_row, goal_col)
        goal_r = 81
        rotation = goal1[1]
        subgoal, mask = gen_subgoal(c_row=goal_row,
                                    c_col=goal_col,
                                    r=goal_r,
                                    h=screenHeight,
                                    w=screenWidth,
                                    rotation=rotation)
        goal_img = (mask[..., None]*255).repeat(3, axis=-1).astype(np.uint8)
      elif task_type == 'target_shape':
          goal_char = config['mpc']['task']['target_char']
          subgoal, goal_img = gen_goal_shape(goal_char,
                                              h=screenHeight,
                                              w=screenWidth)
      else:
          raise NotImplementedError
      # print("goal1:", goal1)
      funnel_dist = np.zeros_like(subgoal)
      info = None
      reward = 0
      # print("evaluation:")
      for _ in range(task.max_steps):
        # print("task goals:", task.goals)
        print("goal1:", goal1)
        # env.
        act = predict(env, subgoal, GNN_single_model, n_mpc=1,
                                        n_look_ahead=n_look_ahead,
                                        n_sample=n_sample,
                                        n_update_iter=n_update_iter,
                                        mpc_type=mpc_type,
                                        gd_loop=gd_loop,
                                        particle_num=-1,
                                        funnel_dist=funnel_dist,
                                        action_seq_mpc_init=action_seq_mpc_init,
                                        action_label_seq_mpc_init=action_label_seq_mpc_init,
                                        time_lim=config['mpc']['time_lim'],
                                        auto_particle_r=True,)['actions'][-1]
        # change: scale act
        r = 13.5
        act = [(act[0] + r ) / (2*r), act[1]/(2*r), (act[2] + r ) / (2*r), act[3]/(2*r)]
        # pdb.set_trace()
        print(f"act:{act}")
        obs, reward, done, info = env.step(act)
        total_reward += reward
        print(f'Total Reward: {total_reward} Done: {done}')
        if done:
          break
      results.append((total_reward, info))

      # Save results.
      with tf.io.gfile.GFile(
          os.path.join(FLAGS.root_dir, f'{name}-{FLAGS.n_steps}.pkl'),
          'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
  app.run(main)
