import mujoco
import mujoco.viewer
import pathlib
import numpy as np
import time
import casadi as cs
import jax
import numpy as np

import envs.rotations as rot
import envs.allegro_fkin as allegro_fk

from mujoco import mjx
from jax import numpy as jp


class AllegroEnv():
    def __init__(self, param):

        self.param_ = param

        # system dimensions:
        self.n_robot_qpos_ = 16
        self.n_qpos_ = 23
        self.n_qvel_ = 22
        self.n_cmd_ = 16

        # low-level control loop
        self.frame_skip_ = int(50)

        # init model data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)

        #place MuJoCo model and data on the GPU device using MJX 
        self.mjx_model = mjx.put_model(self.model_)
        self.mjx_data = mjx.put_data(self.model_, self.data_)

        self.fingertip_names_ = ['ftp_0', 'ftp_1', 'ftp_2', 'ftp_3']

        self.test_ft1_cmd = np.zeros(3)
        self.keyboard_sensitivity = 0.1
        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)

        # initialize cost function
        self.init_cost_fns()

    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            elif chr(keycode) == 'ĉ':
                self.test_ft1_cmd[1] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'Ĉ':
                self.test_ft1_cmd[1] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'ć':
                self.test_ft1_cmd[0] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'Ć':
                self.test_ft1_cmd[0] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'O':
                self.test_ft1_cmd[2] += 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'P':
                self.test_ft1_cmd[2] -= 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'R':
                self.test_ft1_cmd = np.array([0.0, 0.0, 0.0])
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True

    def reset_env(self):
        self.data_.qpos[:] = np.hstack((self.param_.init_robot_qpos_, self.param_.init_obj_qpos_))
        self.data_.qvel[:] = np.zeros(22)

        mujoco.mj_forward(self.model_, self.data_)

    def step(self, jpos_cmd):
        curr_jpos = self.get_jpos()
        target_jpos = (curr_jpos + jpos_cmd)
        for i in range(self.frame_skip_):
            self.data_.ctrl = target_jpos
            mujoco.mj_step(self.model_, self.data_)
            self.viewer_.sync()
            # print('error = ', np.linalg.norm(target_jpos - self.get_jpos()))

    def reset_fingers_qpos(self):
        for iter in range(self.param_.frame_skip_):
            self.data_.ctrl = self.param_.init_robot_qpos_
            mujoco.mj_step(self.model_, self.data_)
            time.sleep(0.001)
            self.viewer_.sync()

    def get_state(self):
        obj_pos = self.data_.qpos.flatten().copy()[-7:]
        robot_pos = self.data_.qpos.flatten().copy()[0:16]
        return np.concatenate((obj_pos, robot_pos))

    def get_jpos(self):
        return self.data_.qpos.flatten().copy()[0:16]

    def get_fingertips_position(self):
        fts_pos = []
        for ft_name in self.fingertip_names_:
            fts_pos.append(self.data_.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()

    def set_goal(self, goal_pos=None, goal_quat=None):
        # goal_id = mujoco.mj_name2id(self.model_, mujoco.mjtObj.mjOBJ_GEOM, 'goal')
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
            # self.model_.geom_pos[goal_id]=goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
            # self.model_.geom_quat[goal_id] = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass

    # ---------------------------------------------------------------------------------------------
    #      cost functions
    # ---------------------------------------------------------------------------------------------
    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        obj_pose = x[0:7]
        ff_qpos = x[7:11]
        mf_qpos = x[11:15]
        rf_qpos = x[15:19]
        tm_qpos = x[19:23]

        # forward kinematics to compute the position of fingertip
        ftp_1_position = allegro_fk.fftp_pos_fd_fn(ff_qpos)
        ftp_2_position = allegro_fk.mftp_pos_fd_fn(mf_qpos)
        ftp_3_position = allegro_fk.rftp_pos_fd_fn(rf_qpos)
        ftp_4_position = allegro_fk.thtp_pos_fd_fn(tm_qpos)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        contact_cost = (
                cs.sumsqr(obj_pose[0:3] - ftp_1_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_2_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_3_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_4_position)
        )

        # control cost
        control_cost = cs.sumsqr(u)

        # cost params
        cost_param = cs.vvcat([target_position, target_quaternion])

        # base cost
        base_cost = 1 * contact_cost
        final_cost = 100 * position_cost + 5.0 * quaternion_cost

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 0.1 * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn

