# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment class."""

import os
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


import pybullet as p


PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'
CUBE_URDF_PATH = 'stacking/block.urdf'


class Environment(gym.Env):
  """OpenAI Gym-style environment class."""

  def __init__(self,
               assets_root,
               task=None,
               disp=False,
               shared_memory=False,
               hz=240,
               use_egl=False):
    """Creates OpenAI Gym-style environment with PyBullet.

    Args:
      assets_root: root directory of assets.
      task: the task to use. If None, the user must call set_task for the
        environment to work properly.
      disp: show environment with PyBullet's built-in display viewer.
      shared_memory: run with shared memory.
      hz: PyBullet physics simulation step speed. Set to 480 for deformables.
      use_egl: Whether to use EGL rendering. Only supported on Linux. Should get
        a significant speedup in rendering when using.

    Raises:
      RuntimeError: if pybullet cannot load fileIOPlugin.
    """
    if use_egl and disp:
      raise ValueError('EGL rendering cannot be used with `disp=True`.')

    self.pix_size = 0.003125
    self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
    self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
    self.agent_cams = cameras.RealSenseD415.CONFIG
    self.assets_root = assets_root
    #add ons
    # self.extrinsics = cameras.RealSenseD415.EXTRIN
    self.cam_params = cameras.RealSenseD415.CAM_PARAMS
    self.config = load_yaml("ravens/config/config.yaml")
    self.is_real = False
    self.wkspc_w = self.config['dataset']['wkspc_w']
    self.headless = self.config['dataset']['headless']
    self.obj = self.config['dataset']['obj']
    self.global_scale = self.config['dataset']['global_scale']
    self.cont_motion = self.config['dataset']['cont_motion']
    self.init_pos = self.config['dataset']['init_pos']
    self.robot_type = self.config['dataset']['robot_type']
    self.img_channel = 1
    # change: from 4 to 6
    self.act_dim = 4
    # cvx_region add ons
    self.cvx_region = np.zeros((1,4)) # every row: left, right, bottom, top
    self.cvx_region[0,0] = -6.75
    self.cvx_region[0,1] = 6.75
    self.cvx_region[0,2] = -13.5
    self.cvx_region[0,3] = 13.5
    # change: add a obs mark
    self.obs = None

    color_tuple = [
        gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
        for config in self.agent_cams
    ]
    depth_tuple = [
        gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
        for config in self.agent_cams
    ]
    self.extrinsics = self.agent_cams[0]['extrinsics']
    
    self.observation_space = gym.spaces.Dict({
        'color': gym.spaces.Tuple(color_tuple),
        'depth': gym.spaces.Tuple(depth_tuple),
    })
    # change: expand bounds, * 10 - 10
    self.position_bounds = gym.spaces.Box(
        low=np.array([0.25, -0.5, 0.], dtype=np.float32),
        high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
        shape=(3,),
        dtype=np.float32)
  #  self.position_bounds = gym.spaces.Box(
  #       low=np.array([0.25, -0.5, 0.], dtype=np.float32),
  #       high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
  #       shape=(3,),
  #       dtype=np.float32)
    self.action_space = gym.spaces.Dict({
        'pose0':
            gym.spaces.Tuple(
                (self.position_bounds,
                 gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32))),
        'pose1':
            gym.spaces.Tuple(
                (self.position_bounds,
                 gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)))
    })
    # self.action_space = gym.spaces.Dict({
    #     'pose0':
    #         gym.spaces.Tuple(
    #             (self.position_bounds,
    #              gym.spaces.Box(-10, 10, shape=(4,), dtype=np.float32))),
    #     'pose1':
    #         gym.spaces.Tuple(
    #             (self.position_bounds,
    #              gym.spaces.Box(-10, 10, shape=(4,), dtype=np.float32)))
    # })
    
    # change: from w=h=720 to 480.640 / 480.640
    self.screenHeight = 480
    self.screenWidth = 640
    
    # Start PyBullet.
    disp_option = p.DIRECT
    if disp:
      disp_option = p.GUI
      if shared_memory:
        disp_option = p.SHARED_MEMORY
    client = p.connect(disp_option)
    file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
    if file_io < 0:
      raise RuntimeError('pybullet: cannot load FileIO!')
    if file_io >= 0:
      p.executePluginCommand(
          file_io,
          textArgument=assets_root,
          intArgs=[p.AddFileIOAction],
          physicsClientId=client)

    self._egl_plugin = None
    if use_egl:
      assert sys.platform == 'linux', ('EGL rendering is only supported on '
                                       'Linux.')
      egl = pkgutil.get_loader('eglRenderer')
      if egl:
        self._egl_plugin = p.loadPlugin(egl.get_filename(),
                                        '_eglRendererPlugin')
      else:
        self._egl_plugin = p.loadPlugin('eglRendererPlugin')
      print('EGL renderering enabled.')

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setAdditionalSearchPath(assets_root)
    p.setAdditionalSearchPath(tempfile.gettempdir())
    p.setTimeStep(1. / hz)

    # If using --disp, move default camera closer to the scene.
    if disp:
      # target = p.getDebugVisualizerCamera()[11]
      # p.resetDebugVisualizerCamera(
      #     cameraDistance=1.1,
      #     cameraYaw=90,
      #     cameraPitch=-25,
      #     cameraTargetPosition=target)
      target = [0.5, 0, 0]
      p.resetDebugVisualizerCamera(
          cameraDistance=0.667,
          cameraYaw=90,
          cameraPitch=-89,
          cameraTargetPosition=target)

    if task:
      self.set_task(task)

  @property
  def is_static(self):
    """Return true if objects are no longer moving."""
    v = [np.linalg.norm(p.getBaseVelocity(i)[0])
         for i in self.obj_ids['rigid']]
    return all(np.array(v) < 5e-3)

  def add_object(self, urdf, pose, category='rigid'):
    """List of (fixed, rigid, or deformable) objects in env."""
    fixed_base = 1 if category == 'fixed' else 0
    obj_id = pybullet_utils.load_urdf(
        p,
        os.path.join(self.assets_root, urdf),
        pose[0],
        pose[1],
        useFixedBase=fixed_base)
    self.obj_ids[category].append(obj_id)
    return obj_id

  """cameras"""
  def get_cam_extrinsics(self):
    return self.extrinsics
  
  def get_cam_params(self):
      # return fx, fy, cx, cy
      return self.cam_params 
  #---------------------------------------------------------------------------
  # Standard Gym Functions
  #---------------------------------------------------------------------------

  def seed(self, seed=None):
    self._random = np.random.RandomState(seed)
    return seed

  def reset(self):
    """Performs common reset functionality for all supported tasks."""
    if not self.task:
      raise ValueError('environment task must be set. Call set_task or pass '
                       'the task arg in the environment constructor.')
    self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.8)
    # cvx_region add ons
    self.cvx_region = np.zeros((1,4)) # every row: left, right, bottom, top
    self.cvx_region[0,0] = -6.75
    self.cvx_region[0,1] = 6.75
    self.cvx_region[0,2] = -13.5
    self.cvx_region[0,3] = 13.5

    # Temporarily disable rendering to load scene faster.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                             [0, 0, -0.001])
    pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])
    # change: add a cube
    # print("add cubes")
    # pybullet_utils.load_urdf(p, os.path.join(self.assets_root, CUBE_URDF_PATH), basePosition=[0.25, 0, 0])
    # pybullet_utils.load_urdf(p, os.path.join(self.assets_root, CUBE_URDF_PATH), basePosition=[0, 0.25, 0])
    # Load UR5 robot arm equipped with suction end effector.
    # TODO(andyzeng): add back parallel-jaw grippers.
    self.ur5 = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, UR5_URDF_PATH))
    self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
    self.ee_tip = 10  # Link ID of suction cup.

    # Get revolute joint indices of robot (skip fixed joints).
    n_joints = p.getNumJoints(self.ur5)
    joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
    self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

    # Move robot to home joint configuration.
    for i in range(len(self.joints)):
      p.resetJointState(self.ur5, self.joints[i], self.homej[i])

    # Reset end effector.
    self.ee.release()

    # Reset task.
    self.task.reset(self)

    # Re-enable rendering.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    obs, _, _, _ = self.step()
    return obs

  def step(self, action=None):
    """Execute action with specified primitive.

    Args:
      action: action to execute.

    Returns:
      (obs, reward, done, info) tuple containing MDP step data.
    """
    if action is not None:
      # change: from :2 to :3
      pose0 = np.asarray(action[:2])
      pose1 = np.asarray(action[2:])
      if (pose0 - pose1)[0] == 0:
            pusher_angle = np.pi/2
      else:
          pusher_angle = np.arctan((pose0 - pose1)[1]/(pose0 - pose1)[0])
      orn = np.array([0.0, np.pi, pusher_angle + np.pi/2])
      end_effector_orn = p.getQuaternionFromEuler(orn)
      action = {'pose0' : (pose0, end_effector_orn), 'pose1' : (pose1, end_effector_orn)}

      timeout = self.task.primitive(self.movej, self.movep, self.ee, **action)

      # Exit early if action times out. We still return an observation
      # so that we don't break the Gym API contract.
      if timeout:
        obs = self._get_obs()
        return obs, 0.0, True, self.info

    # Step simulator asynchronously until objects settle.
    while not self.is_static:
      p.stepSimulation()

    # Get task rewards.
    reward, info = self.task.reward() if action is not None else (0, {})
    done = self.task.done()

    # Add ground truth robot state into info.
    info.update(self.info)

    obs = self._get_obs()

    return obs, reward, done, info

  def close(self):
    if self._egl_plugin is not None:
      p.unloadPlugin(self._egl_plugin)
    p.disconnect()

  def render(self, mode='rgb_array'):
    # Render only the color image from the first camera.
    # Only support rgb_array for now.
    if mode != 'rgb_array':
      raise NotImplementedError('Only rgb_array implemented')
    color, _, _ = self.render_camera(self.agent_cams[0])
    return color

  def render_camera(self, config):
    """Render RGB-D image with specified camera configuration."""

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = p.getMatrixFromQuaternion(config['rotation'])
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = config['position'] + lookdir
    focal_len = config['intrinsics'][0]
    znear, zfar = config['zrange']
    viewm = p.computeViewMatrix(config['position'], lookat, updir)
    fovh = (config['image_size'][0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = config['image_size'][1] / config['image_size'][0]
    projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = p.getCameraImage(
        width=config['image_size'][1],
        height=config['image_size'][0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        # Note when use_egl is toggled, this option will not actually use openGL
        # but EGL instead.
        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    # print(f"Depth range: min = {depth_min}, max = {depth_max}")

    # Get color image.
    color_image_size = (config['image_size'][0], config['image_size'][1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    # color = color[:, :, :3]  # remove alpha channel
    if config['noise']:
      color = np.int32(color)
      color += np.int32(self._random.normal(0, 3, config['image_size']))
      color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (config['image_size'][0], config['image_size'][1])
    zbuffer = np.array(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
    depth = (2. * znear * zfar) / depth
    if config['noise']:
      depth += self._random.normal(0, 0.003, depth_image_size)

    # Get segmentation image.
    segm = np.uint8(segm).reshape(depth_image_size)
    # change: add depth channel
    depth1 = np.expand_dims(depth, axis=-1)
    color = np.concatenate((color, depth1), axis=-1)
    return color, depth, segm

  @property
  def info(self):
    """Environment info variable with object poses, dimensions, and colors."""

    # Some tasks create and remove zones, so ignore those IDs.
    # removed_ids = []
    # if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
    #         isinstance(self.task, tasks.names['bag-alone-open'])):
    #   removed_ids.append(self.task.zone_id)

    info = {}  # object id : (position, rotation, dimensions)
    for obj_ids in self.obj_ids.values():
      for obj_id in obj_ids:
        pos, rot = p.getBasePositionAndOrientation(obj_id)
        dim = p.getVisualShapeData(obj_id)[0][3]
        info[obj_id] = (pos, rot, dim)
    return info

  def set_task(self, task):
    task.set_assets_root(self.assets_root)
    self.task = task

  #---------------------------------------------------------------------------
  # Robot Movement Functions
  #---------------------------------------------------------------------------

  def movej(self, targj, speed=0.01, timeout=10):
    """Move UR5 to target joint configuration."""
    t0 = time.time()
    while (time.time() - t0) < timeout:
      currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
      currj = np.array(currj)
      diffj = targj - currj
      # print(f"diffj: {diffj}")
      if all(np.abs(diffj) < 1e-2):
        return False

      # Move with constant velocity
      norm = np.linalg.norm(diffj)
      v = diffj / norm if norm > 0 else 0
      stepj = currj + v * speed
      # stepj = targj
      gains = np.ones(len(self.joints))
      p.setJointMotorControlArray(
          bodyIndex=self.ur5,
          jointIndices=self.joints,
          controlMode=p.POSITION_CONTROL,
          targetPositions=stepj,
          positionGains=gains)
      p.stepSimulation()
    print(f"diffj: {diffj}")
    print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
    return True

  def movep(self, pose, speed=0.01):
    """Move UR5 to target end effector pose."""
    targj = self.solve_ik(pose)
    return self.movej(targj, speed)

  def solve_ik(self, pose):
    """Calculate joint configuration with inverse kinematics."""
    joints = p.calculateInverseKinematics(
        bodyUniqueId=self.ur5,
        endEffectorLinkIndex=self.ee_tip,
        targetPosition=pose[0],
        targetOrientation=pose[1],
        lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
        upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
        jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
        restPoses=np.float32(self.homej).tolist(),
        maxNumIterations=100,
        residualThreshold=1e-5)
    joints = np.float32(joints)
    joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
    return joints

  def _get_obs(self):
    # Get RGB-D camera image observations.
    obs = {'color': (), 'depth': ()}
    for config in self.agent_cams:
      color, depth, _ = self.render_camera(config)
      obs['color'] += (color,)
      obs['depth'] += (depth,)
    if self.obs is not None:
      o = np.array_equal(obs['color'], self.obs['color'])
    else: 
      o = None
    print("obs:", o)
    self.obs = obs
    return obs

  def obs2ptcl_fixed_num_batch(self, obs, particle_num, batch_size):
        assert type(obs) == np.ndarray
        # from 5 to 3
        assert obs.shape[-1] == 5
        assert obs[..., :3].max() <= 255.0
        assert obs[..., :3].min() >= 0.0
        assert obs[..., :3].max() >= 1.0
        print(f"max:{obs[..., -1].max()}")
        # change: ignore 2 assert
        assert obs[..., -1].max() >= 0.7 * self.global_scale
        assert obs[..., -1].max() <= 0.8 * self.global_scale
        depth = obs[..., -1] / self.global_scale
        print("batch_size: ", batch_size)
        print("particle_num", particle_num)
        batch_sampled_ptcl = np.zeros((batch_size, particle_num, 3))
        batch_particle_r = np.zeros((batch_size, ))
        for i in range(batch_size):
            # note: in world cdn
            fgpcd = depth2fgpcd(depth, depth<0.599/0.8, self.get_cam_params())
            fgpcd = downsample_pcd(fgpcd, 0.01)
            sampled_ptcl, particle_r = fps(fgpcd, particle_num)
            # print("sampled_ptcl:", sampled_ptcl.shape)
            # change: Scale the point cloud by 27 and shift the x-coordinates by 0.5
            sampled_ptcl[:, 0] = (sampled_ptcl[:, 0] - 0.5) * 27  # Scale x and shift
            sampled_ptcl[:, 1] = sampled_ptcl[:, 1] * 27        # Scale y
            sampled_ptcl[:, 2] = sampled_ptcl[:, 2] * 27        # Scale z if needed
            
            batch_sampled_ptcl[i] = recenter(fgpcd, sampled_ptcl, r = min(0.02, 0.5 * particle_r))
            batch_particle_r[i] = particle_r
        return batch_sampled_ptcl, batch_particle_r    
  
  def sample_action(self, n):
        # sample one action within feasible space and with corresponding convex region label
        # change: from 4 to 6
        action = -self.wkspc_w + 2 * self.wkspc_w * np.random.rand(n, 1, 4)
        reg_label = np.zeros(n)
        return action, reg_label
  
  def step_subgoal_ptcl(self,
                          subgoal,
                          model_dy,
                          init_pos = None,
                          n_mpc=30,
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
                          auto_particle_r=False,):
        assert type(subgoal) == np.ndarray
        # assert subgoal.shape == (self.screenHeight, self.screenWidth)
        # planner
        if mpc_type == 'GD':
            self.planner = PlannerGD(self.config, self)
        else:
            raise NotImplementedError
        action_lower_lim = action_upper_lim = np.zeros(4) # DEPRECATED, should be of no use
        reward_params = (self.get_cam_extrinsics(), self.get_cam_params(), self.global_scale)

        particle_den_seq = []
        if auto_particle_r:
            res_rgr_folder = self.config['mpc']['res_sel']['model_folder']
            res_rgr_folder = os.path.join('checkpoints/dyn_sweep', res_rgr_folder)
            res_rgr = MPCResRgrNoPool(self.config)
            # res_rgr = MPCResCls(self.config)
            if self.config['mpc']['res_sel']['iter_num'] == -1:
                res_rgr.load_state_dict(torch.load(os.path.join(res_rgr_folder, 'net_best_dy_state_dict.pth')))
            else:
                res_rgr.load_state_dict(torch.load(os.path.join(res_rgr_folder, 'net_dy_iter_%d_state_dict.pth' % self.config['mpc']['res_sel']['iter_num'])))
            res_rgr = res_rgr.cuda()
            
            # construct res_rgr input
            # first channel is the foreground mask
            fg_mask = (self.render()[..., -1] / self.global_scale < 0.599/0.8).astype(np.float32)
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
        raw_obs = np.zeros((n_mpc+1, self.screenHeight, self.screenWidth, 5))
        # states = np.zeros((n_mpc+1, particle_num, 3))
        states = []
        actions = np.zeros((n_mpc, self.act_dim))
        # states_pred = np.zeros((n_mpc, particle_num, 3))
        states_pred = []
        rew_means = np.zeros((n_mpc, 1, n_update_iter * gd_loop))
        rew_stds = np.zeros((n_mpc, 1, n_update_iter * gd_loop))

        if init_pos is not None:
            self.set_positions(init_pos)
        obs_cur = self.render()
        
        # print("obs_cur")
        # print(obs_cur.shape)
        # print(raw_obs.shape)
        # print("raw_obs[0]")
        # print(raw_obs[0].shape)
        raw_obs[0] = obs_cur
        
        obs_cur, particle_r = self.obs2ptcl_fixed_num_batch(obs_cur, particle_num, batch_size=30)
        
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
        # obs_cur = self.obs2ptcl(obs_cur, particle_r)
        if action_seq_mpc_init is None:
            action_seq_mpc_init, action_label_seq_mpc_init = self.sample_action(n_mpc)
        subgoal_tensor = torch.from_numpy(subgoal).float().cuda().reshape(self.screenHeight, self.screenWidth)
        subgoal_coor_tensor = torch.flip((subgoal_tensor < 0.5).nonzero(), dims=(1,)).cuda().float()
        subgoal_coor_np, _ = fps_np(subgoal_coor_tensor.detach().cpu().numpy(), min(particle_num * 5, subgoal_coor_tensor.shape[0]))
        subgoal_coor_tensor = torch.from_numpy(subgoal_coor_np).float().cuda()
        rewards[0] = config_reward_ptcl(torch.from_numpy(obs_cur).float().cuda().reshape(-1, particle_num, 3),
                                        subgoal_tensor,
                                        cam_params=self.get_cam_params(),
                                        goal_coor=subgoal_coor_tensor,
                                        normalize=True)[0].item()
        states.append(obs_cur[0])
        total_time = 0.0
        rollout_time = 0.0
        optim_time = 0.0
        iter_num = 0
        for i in range(n_mpc):
            attr_cur = np.zeros((obs_cur.shape[0], particle_num))
            # init_p = self.get_positions()
            # print('mpc iter: {}'.format(i))
            # print('action_seq_mpc_init: {}'.format(action_seq_mpc_init.shape))
            # print('action_label_seq_mpc_init: {}'.format(action_label_seq_mpc_init))
            # print(f'act_Seq:{action_seq_mpc_init[:n_look_ahead]}')
            traj_opt_out = self.planner.trajectory_optimization_ptcl_multi_traj(
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
            # print('action:', action_seq_mpc[0])
            print('actions:', action_seq_mpc)
            
            obs_cur = self.step(action_seq_mpc[0])[0]['color'][0]
            obs_cur = self.step(action_seq_mpc[0])
            print("obs_cur:", obs_cur)
            if obs_cur is None:
                raise Exception('sim exploded')

            if auto_particle_r:
                # construct res_rgr input
                # first channel is the foreground mask
                fg_mask = (self.render()[..., -1] / self.global_scale < 0.599/0.8).astype(np.float32)
                # second channel is the goal mask
                subgoal_mask = (subgoal < 0.5).astype(np.float32)
                particle_num = res_rgr.infer_param(fg_mask, subgoal_mask)
                # particle_den = np.array([res_rgr(res_rgr_input[None, ...]).item() * 4000.0])
                # particle_r = np.sqrt(1.0 / particle_den)
                # particle_den_seq.append(particle_den)
                particle_den_seq.append(particle_num)

            # obs_cur = self.render()
            # print(f"raw_obs633: {raw_obs[i + 1].shape}")
            # print(f"obs_cur634: {obs_cur.shape}")
            raw_obs[i + 1] = obs_cur
            obs_cur, particle_r = self.obs2ptcl_fixed_num_batch(obs_cur, particle_num, batch_size=30)
            particle_den = np.array([1 / (particle_r ** 2)])[0]
            print('particle_den:', particle_den)
            print('particle_num:', particle_num)
            # obs_cur = self.obs2ptcl(obs_cur, particle_r)
            states.append(obs_cur[0])
            actions[i] = action_seq_mpc[0]
            subgoal_coor_np, _ = fps_np(subgoal_coor_tensor.detach().cpu().numpy(), min(obs_cur.shape[0] * 5, subgoal_coor_tensor.shape[0]))
            subgoal_coor_tensor = torch.from_numpy(subgoal_coor_np).float().cuda()
            rewards[i + 1] = config_reward_ptcl(torch.from_numpy(obs_cur).float().cuda().reshape(-1, particle_num, 3),
                                                subgoal_tensor,
                                                cam_params=self.get_cam_params(),
                                                goal_coor=subgoal_coor_tensor,
                                                normalize=True)[0].item()
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
                    

            print('rewards: {}'.format(rewards))
            print()
        return {'rewards': rewards,
                'raw_obs': raw_obs,
                'states': states,
                'actions': actions,
                'states_pred': states_pred,
                'rew_means': rew_means,
                'rew_stds': rew_stds,
                'total_time': total_time,
                'rollout_time': rollout_time,
                'optim_time': optim_time,
                'iter_num': iter_num,
                'particle_den_seq': particle_den_seq,}

class EnvironmentNoRotationsWithHeightmap(Environment):
  """Environment that disables any rotations and always passes [0, 0, 0, 1]."""

  def __init__(self,
               assets_root,
               task=None,
               disp=False,
               shared_memory=False,
               hz=240):
    super(EnvironmentNoRotationsWithHeightmap,
          self).__init__(assets_root, task, disp, shared_memory, hz)

    heightmap_tuple = [
        gym.spaces.Box(0.0, 20.0, (320, 160, 3), dtype=np.float32),
        gym.spaces.Box(0.0, 20.0, (320, 160), dtype=np.float32),
    ]
    self.observation_space = gym.spaces.Dict({
        'heightmap': gym.spaces.Tuple(heightmap_tuple),
    })
    self.action_space = gym.spaces.Dict({
        'pose0': gym.spaces.Tuple((self.position_bounds,)),
        'pose1': gym.spaces.Tuple((self.position_bounds,))
    })

  def step(self, action=None):
    """Execute action with specified primitive.

    Args:
      action: action to execute.

    Returns:
      (obs, reward, done, info) tuple containing MDP step data.
    """
    if action is not None:
      action = {
          'pose0': (action['pose0'][0], [0., 0., 0., 1.]),
          'pose1': (action['pose1'][0], [0., 0., 0., 1.]),
      }
    return super(EnvironmentNoRotationsWithHeightmap, self).step(action)

  def _get_obs(self):
    obs = {}

    color_depth_obs = {'color': (), 'depth': ()}
    for config in self.agent_cams:
      color, depth, _ = self.render_camera(config)
      color_depth_obs['color'] += (color,)
      color_depth_obs['depth'] += (depth,)
    cmap, hmap = utils.get_fused_heightmap(color_depth_obs, self.agent_cams,
                                           self.task.bounds, pix_size=0.003125)
    obs['heightmap'] = (cmap, hmap)
    return obs


class ContinuousEnvironment(Environment):
  """A continuous environment."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Redefine action space, assuming it's a suction-based task. We'll override
    # it in `reset()` if that is not the case.
    self.position_bounds = gym.spaces.Box(
        low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        shape=(3,),
        dtype=np.float32
    )
    self.action_space = gym.spaces.Dict({
        'move_cmd':
            gym.spaces.Tuple(
                (self.position_bounds,
                 gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
        'suction_cmd': gym.spaces.Discrete(2),  # Binary 0-1.
        'acts_left': gym.spaces.Discrete(1000),
    })

  def set_task(self, task):
    super().set_task(task)

    # Redefine the action-space in case it is a pushing task. At this point, the
    # ee has been instantiated.
    if self.task.ee == Spatula:
      self.action_space = gym.spaces.Dict({
          'move_cmd':
              gym.spaces.Tuple(
                  (self.position_bounds,
                   gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
          'slowdown_cmd': gym.spaces.Discrete(2),  # Binary 0-1.
          'acts_left': gym.spaces.Discrete(1000),
      })

  def get_ee_pose(self):
    return p.getLinkState(self.ur5, self.ee_tip)[0:2]

  def step(self, action=None):
    if action is not None:
      # change: from :2 to :3
      pose0 = np.asarray(action[:2])
      pose1 = np.asarray(action[2:])
      if (pose0 - pose1)[0] == 0:
            pusher_angle = np.pi/2
      else:
          pusher_angle = np.arctan((pose0 - pose1)[1]/(pose0 - pose1)[0])
      orn = np.array([0.0, np.pi, pusher_angle + np.pi/2])
      end_effector_orn = p.getQuaternionFromEuler(orn)
      action = {'pose0' : (pose0, end_effector_orn), 'pose1' : (pose1, end_effector_orn)}

      timeout = self.task.primitive(self.movej, self.movep, self.ee, action)

      # Exit early if action times out. We still return an observation
      # so that we don't break the Gym API contract.
      if timeout:
        obs = self._get_obs()
        return obs, 0.0, True, self.info

    # Step simulator asynchronously until objects settle.
    while not self.is_static:
      p.stepSimulation()

    # Get task rewards.
    reward, info = self.task.reward() if action is not None else (0, {})
    task_done = self.task.done()
    if action is not None:
      done = task_done and action['acts_left'] == 0
    else:
      done = task_done

    # Add ground truth robot state into info.
    info.update(self.info)

    obs = self._get_obs()

    return obs, reward, done, info
