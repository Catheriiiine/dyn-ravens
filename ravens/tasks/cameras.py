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

"""Camera configs."""

import numpy as np
import pybullet as p

# def get_rotation_matrix(quaternion):
#     """Convert quaternion to rotation matrix."""
#     return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

# def create_transformation_matrix(position, rotation):
#     """Create a 4x4 transformation matrix from position and rotation."""
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = get_rotation_matrix(rotation)
#     transformation_matrix[:3, 3] = position
#     return transformation_matrix

def create_rotation_matrix(pitch, yaw, roll):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    return R

def create_view_matrix(position, rotation):
    pitch, yaw, roll = rotation
    R = create_rotation_matrix(pitch, yaw, roll)
    
    # Extract the front, up, and right vectors from the rotation matrix
    front = R @ np.array([0, 0, -1])
    up = R @ np.array([0, 1, 0])
    right = np.cross(up, front)
    
    # Normalize the vectors (optional but recommended)
    front /= np.linalg.norm(front)
    up /= np.linalg.norm(up)
    right /= np.linalg.norm(right)
    
    # print("here")
    # Create the view matrix
    view_matrix = np.eye(4)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up
    view_matrix[2, :3] = -front
    view_matrix[0, 3] = -np.dot(right, position)
    view_matrix[1, 3] = -np.dot(up, position)
    view_matrix[2, 3] = np.dot(front, position)
    
    return view_matrix


class RealSenseD415():
  """Default configuration with 3 RealSense RGB-D cameras."""

  # Mimic RealSense D415 RGB-D camera parameters.
  # change: 480, 640
  image_size = (480, 640)
  intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)
  cam_params = [450, 450, 320, 240]
  # Set default camera poses.
  front_position = (1., 0, 0.75)
  front_rotation1 = (np.pi / 4, np.pi, -np.pi / 2)
  front_rotation = p.getQuaternionFromEuler(front_rotation1)
  left_position = (0, 0.5, 0.75)
  left_rotation1 = (np.pi / 4.5, np.pi, np.pi / 4)
  left_rotation = p.getQuaternionFromEuler(left_rotation1)
  right_position = (0, -0.5, 0.75)
  right_rotation1 = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
  right_rotation = p.getQuaternionFromEuler(right_rotation1)
  top_position = (0.5, 0, 0.667)
  top_rotation1 = (0, np.pi, -np.pi / 2)
  top_rotation = p.getQuaternionFromEuler(top_rotation1)

  extrinsics_front = create_view_matrix(front_position, front_rotation1)
  extrinsics_left = create_view_matrix(left_position, left_rotation1)
  extrinsics_right = create_view_matrix(right_position, right_rotation1)
  extrinsics_top = create_view_matrix(top_position, top_rotation1)
#   front_matrix = create_transformation_matrix(front_position, front_rotation)
#   left_matrix = create_transformation_matrix(left_position, left_rotation)
#   right_matrix = create_transformation_matrix(right_position, right_rotation)
  
# #   extrinsics = np.stack([front_matrix, left_matrix, right_matrix])
#   extrinsics = [[0, -0.5, 0.75], [], [1., 0, 0.75], []]
  # Default camera configs.
  CONFIG = [{
     'image_size': image_size,
      'intrinsics': intrinsics,
      'extrinsics': extrinsics_top,
      'position': top_position,
      'rotation': top_rotation,
      'zrange': (0.01, 20.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'extrinsics': extrinsics_front,
      'position': front_position,
      'rotation': front_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'extrinsics': extrinsics_left,
      'position': left_position,
      'rotation': left_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'extrinsics': extrinsics_right,
      'position': right_position,
      'rotation': right_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, ]
  
  INTRIN = intrinsics
  CAM_PARAMS = cam_params


class Oracle():
  """Top-down noiseless image used only by the oracle demonstrator."""

  # Near-orthographic projection.
  # change: 640, 480
  image_size = (480, 640)
  intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
  position = (0.5, 0, 1000.)
  rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

  # Camera config.
  CONFIG = [{
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': position,
      'rotation': rotation,
      'zrange': (999.7, 1001.),
      'noise': False
  }]
