from envs.allegro_env_mjx import BatchedAllegroEnv
from envs import rotations
import numpy as np
import jax
import time


class env_param_jax:

    def __init__(self):
        self.model_path_ = 'envs/xmls/env_allegro_bowl.xml'

        # system dimensions:
        self.n_robot_qpos_ = 16
        self.n_qpos_ = 23
        self.n_qvel_ = 22
        self.n_cmd_ = 16
        self.batch_size = 4096

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100)

        self.init_robot_qpos_ = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])

        # random init and target pose for object
        init_obj_xy = np.array([-0.03, -0.01]) + 0.005 * np.random.randn(2)
        init_obj_pos = np.hstack([init_obj_xy, 0.03])
        init_yaw_angle = np.pi * np.random.rand(1) - np.pi / 2
        init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([init_yaw_angle, 0, 0]))
        self.init_obj_qpos_ = np.hstack((init_obj_pos, init_obj_quat_rand))

        self.target_p_ = np.array([-0.02, -0.01, 0.03])
        yaw_angle = init_yaw_angle + np.random.choice([np.pi / 2, -np.pi / 2])
        self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

# init environment
param = env_param_jax()
env = BatchedAllegroEnv(param)

reset = jax.jit(env.reset_env)
step = jax.jit(env.step)

for idx in range(2):

    t = time.perf_counter()
    mjx_data = reset()
    print(f"reset time: {time.perf_counter() - t}")
