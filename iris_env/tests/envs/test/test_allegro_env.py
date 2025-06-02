from envs.allegro_env import MjSimulator
import torch
from utils import torch_transform
import numpy as np


class param:
    def __init__(self):
        self.model_path_ = 'envs/sources/xml/allegro_cube_env.xml'
        target_position = torch.tensor([-0.03, -0.00, 0.023])

        target_axis_angle = -1.0 * torch.pi / 2 * torch.tensor([.0, .0, 1.0])
        target_quaternion = torch_transform.axis_angle_to_quaternion(target_axis_angle)
        self.target_p_ = target_position.numpy()
        self.target_q_ = target_quaternion.numpy()

        obj_pose = np.array([-0.03, 0.0, 0.023, 1, 0, 0, 0])
        joint_position = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])

        self.mj_qpos_position_ = np.hstack((joint_position, obj_pose))
        self.frame_skip_ = int(50)


env_param = param()
env = MjSimulator(env_param)
jpos_cmd = 0.01 * np.ones(16)

# get fintertip position
x_ftps = env.get_fingertips_position()
print(x_ftps[0:3])
print(x_ftps[3:6])
print(x_ftps[6:9])
print(x_ftps[9:12])
# print(env.data_.body('th_proximal').xpos)

print('-------')
qjoint = env.get_jpos()
print(env.fftp_pos_fd_fn(qjoint[0:4]))
print(env.mftp_pos_fd_fn(qjoint[4:8]))
print(env.rftp_pos_fd_fn(qjoint[8:12]))
print(env.thtp_pos_fd_fn(qjoint[12:16]))
# print(env.get_jpos())
