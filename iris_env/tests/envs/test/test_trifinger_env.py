from envs.trifinger_env import MjSimulator
import torch
from utils import torch_transform
import numpy as np
from envs.trifinger_fkin import f0tip_pos_fd_fn, f120tip_pos_fd_fn, f240tip_pos_fd_fn


class param:
    def __init__(self):
        self.model_path_ = 'envs/sources/xml/env_trifinger_foambrick.xml'
        target_position = torch.tensor([-0.03, -0.00, 0.023])

        target_axis_angle = -1.0 * torch.pi / 2 * torch.tensor([.0, .0, 1.0])
        target_quaternion = torch_transform.axis_angle_to_quaternion(target_axis_angle)
        self.target_p_ = target_position.numpy()
        self.target_q_ = target_quaternion.numpy()

        self.init_obj_qpos_ = np.array([-0.1, 0.1, 0.023, 1, 0, 0, 0])
        self.init_robot_qpos_ = np.array([
            0.0, -.6, -1.1,
            0.0, -.6, -1.1,
            0.0, -.6, -1.1
        ])

        self.jc_kp_ = 10  # the recommended range [5000~50000]
        self.jc_damping_ = 0.1  # the recommended range [2]

        self.frame_skip_ = int(50)
        self.n_qvel_ = 15


env_param = param()
env = MjSimulator(env_param)
cmd = 0.01 * np.ones(9)
env.step(cmd)

f0tip_x = f0tip_pos_fd_fn(env.get_finger_jpos()[0:3])
f120tip_x = f120tip_pos_fd_fn(env.get_finger_jpos()[3:6])
f240tip_x = f240tip_pos_fd_fn(env.get_finger_jpos()[6:9])
print('0-------')
print(f0tip_x)
print(f120tip_x)
print(f240tip_x)

ftips_x = env.get_fingertip_position()
print(env.data_.site('fingertip_0').xpos)
print(env.data_.site('fingertip_120').xpos)
print(env.data_.site('fingertip_240').xpos)
