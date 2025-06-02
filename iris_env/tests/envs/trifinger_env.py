import mujoco
import mujoco.viewer
import numpy as np
import casadi as cs

import envs.trifinger_fkin as trifinger_fkin
from envs import rotations


class TriFingerEnv():
    def __init__(self, param):
        self.param_ = param

        # system dimensions:
        self.n_qpos_ = 16
        self.n_qvel_ = 15
        self.n_cmd_ = 9

        # internal joint controller for each finger
        self.jc_kp_ = 10
        self.jc_damping_ = 0.05

        # low level control steps
        self.frame_skip_ = int(20)

        # init model data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)
        self.ft_names_ = ['fingertip_0', 'fingertip_120', 'fingertip_240']

        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_mj_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)

    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True

    def reset_mj_env(self):
        self.data_.qpos[:] = np.copy(np.concatenate((self.param_.init_obj_qpos_, self.param_.init_robot_qpos_)))
        self.data_.qvel[:] = np.copy(np.array(self.n_qvel_ * [0]))
        mujoco.mj_forward(self.model_, self.data_)

    def step(self, cmd):
        finger_jpos = self.get_finger_jpos()
        target_jpos = (finger_jpos + cmd).copy()

        # run the OCS controller
        for _ in range(self.frame_skip_):
            e_jpos = target_jpos - self.get_finger_jpos()
            e_jvel = self.data_.qvel[6:]
            torque = self.jc_kp_ * (e_jpos) - self.jc_damping_ * e_jvel
            self.data_.ctrl[:] = torque + self.data_.qfrc_bias[6:]
            mujoco.mj_step(self.model_, self.data_, nstep=1)
            self.viewer_.sync()

        mujoco.mj_forward(self.model_, self.data_)

    def get_fingertip_position(self):
        fts_pos = []
        for ft_name in self.ft_names_:
            fts_pos.append(self.data_.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()

    def get_state(self):
        mujoco.mj_forward(self.model_, self.data_)
        return self.data_.qpos.flatten().copy()

    def get_finger_jpos(self):
        mujoco.mj_forward(self.model_, self.data_)
        return self.data_.qpos.flatten()[7:].copy()

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass

    # ---------------------------------------------------------------------------------------------
    #      cost function for MPC
    # ---------------------------------------------------------------------------------------------

    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        obj_pose = x[0:7]
        f0_qpos = x[7:10]
        f120_qpos = x[10:13]
        f240_qpos = x[13:16]

        # forward kinematics to compute the position of fingertip
        ftp_1_position = trifinger_fkin.f0tip_pos_fd_fn(f0_qpos)
        ftp_2_position = trifinger_fkin.f120tip_pos_fd_fn(f120_qpos)
        ftp_3_position = trifinger_fkin.f240tip_pos_fd_fn(f240_qpos)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        contact_cost = (
                cs.sumsqr(obj_pose[0:3] - ftp_1_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_2_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_3_position)
        )

        # grasp cost
        obj_dirmat = rotations.quat2dcm_fn(x[3:7])
        obj_v0 = obj_dirmat.T @ (ftp_1_position - x[0:3])
        obj_v1 = obj_dirmat.T @ (ftp_2_position - x[0:3])
        obj_v2 = obj_dirmat.T @ (ftp_3_position - x[0:3])
        grasp_closure = cs.sumsqr(obj_v0 / cs.norm_2(obj_v0) + obj_v1 / cs.norm_2(obj_v1) + obj_v2 / cs.norm_2(obj_v2))

        # control cost
        control_cost = cs.sumsqr(u)

        # cost params
        cost_param = cs.vvcat([target_position, target_quaternion])

        # base cost
        base_cost = 0.5 * contact_cost + 0.05 * grasp_closure
        final_cost = 500 * position_cost + 5.0 * quaternion_cost

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 10 * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn
