import mujoco 
import jax

from mujoco import mjx
from jax import numpy as jnp 
import envs.allegro_fkin_jax as allegro_fk

class BatchedAllegroEnv:
    def __init__(self, param):
        # Initialize parameters and fetch batch size
        self.param_ = param
        self.batch_size = param.batch_size

        # system dimensions:
        self.n_robot_qpos_ = 16
        self.n_qpos_ = 23
        self.n_qvel_ = 22
        self.n_cmd_ = 16

        # low-level control loop
        self.frame_skip_ = int(50)

        # Initialize model
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        # In __init__ method:
        self.mjx_model_ = mjx.put_model(self.model_)
        # Generate model batches
        self.mjx_model_batch = jax.tree_map(
            lambda x: jnp.tile(x[None], (self.batch_size,) + (1,) * (x.ndim)),
            self.mjx_model_
        )
        # Create a batch of environment
        self.mjx_data_batch = jax.vmap(mjx.make_data)(self.mjx_model_batch)

        self.fingertip_names_ = ['ftp_0', 'ftp_1', 'ftp_2', 'ftp_3']

        self.test_ft1_cmd = jnp.zeros(3)
        self.keyboard_sensitivity = 0.1
        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.goal_pos = jnp.zeros((self.batch_size, 3))
        self.goal_quat = jnp.zeros((self.batch_size, 4))

        self.set_goal(self.param_.target_p_, self.param_.target_q_)

        self.reset_env()

        self.init_cost_fns()

    # Remains untouched
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
                self.test_ft1_cmd = jnp.array([0.0, 0.0, 0.0])
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True



    def reset_env(self):
        init_qpos = jnp.tile(
            jnp.concatenate([self.param_.init_robot_qpos_, self.param_.init_obj_qpos_]),
            (self.batch_size, 1))

        init_qvel = jnp.zeros((self.batch_size, self.n_qvel_))
        
        self.mjx_data_batch = jax.vmap(lambda d, qp, qv: d.replace(qpos=qp, qvel=qv))(
            self.mjx_data_batch, init_qpos, init_qvel)
        
        self.mjx_data_batch = jax.vmap(mjx.forward)(self.mjx_model_, self.mjx_data_batch)

        return self.mjx_data_batch
    

    def get_jpos(self):
        return self.mjx_data_batch.qpos[:, :self.n_robot_qpos_]
        

    def get_state(self):
        obj_pos = self.mjx_data_batch.qpos[:, -7:]
        robot_pos = self.mjx_data_batch.qpos[:, :16]
        return jnp.concatenate([obj_pos, robot_pos], axis=1)

    

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            # Ensure goal_pos is 2D: (batch_size, 3)
            self.goal_pos = jnp.atleast_2d(goal_pos)
            if self.goal_pos.shape[0] == 1:
                self.goal_pos = jnp.tile(self.goal_pos, (self.batch_size, 1))
        if goal_quat is not None:
            # Ensure goal_quat is 2D: (batch_size, 4)
            self.goal_quat = jnp.atleast_2d(goal_quat)
            if self.goal_quat.shape[0] == 1:
                self.goal_quat = jnp.tile(self.goal_quat, (self.batch_size, 1))
        
        # Update goal position in the MJX model
        def update_goal_pos(model, pos):
            body_id = mjx.name2id(model, 1, b'goal')  # 1 is the constant for mjOBJ_BODY in MJX
            return model.replace(body_pos=model.body_pos.at[body_id].set(pos))
        
        # Update goal orientation in the MJX model
        def update_goal_quat(model, quat):
            body_id = mjx.name2id(model, 1, b'goal')  # 1 is the constant for mjOBJ_BODY in MJX
            return model.replace(body_quat=model.body_quat.at[body_id].set(quat))
        
        # Apply updates to all models in the batch
        self.mjx_model_batch = jax.vmap(update_goal_pos)(self.mjx_model_batch, self.goal_pos)
        self.mjx_model_batch = jax.vmap(update_goal_quat)(self.mjx_model_batch, self.goal_quat)
        
        # Forward the simulation to apply changes
        self.mjx_data_batch = jax.vmap(mjx.forward)(self.mjx_model_batch, self.mjx_data_batch)


    def step(self, jpos_cmd_batch):
        # Get current joint positions for all batch elements
        curr_jpos_batch = self.get_jpos_batch()
        
        # Calculate target joint positions for all batch elements
        target_jpos_batch = curr_jpos_batch + jpos_cmd_batch

        # Define a single step function
        def single_step(model, data, target_jpos):
            data = data.replace(ctrl=target_jpos)
            return mjx.step(model, data)

        # Create a batched version of the single step function
        batched_single_step = jax.vmap(single_step, in_axes=(None, 0, 0))

        # Perform frame_skip steps
        def body_fun(_, loop_carry):
            return batched_single_step(self.mjx_model_, loop_carry, target_jpos_batch)

        self.mjx_data_batch = jax.lax.fori_loop(
            0, self.frame_skip_, body_fun, self.mjx_data_batch
        )

        return self.mjx_data_batch


    def compute_path_cost(self, x, u):
        obj_pose, robot_qpos = x[:, :7], x[:, 7:]
        ff_qpos, mf_qpos, rf_qpos, tm_qpos = robot_qpos[:, :4], robot_qpos[:, 4:8], robot_qpos[:, 8:12], robot_qpos[:, 12:]
        
        # Compute fingertip positions
        ftp_1_position = jax.vmap(allegro_fk.fftp_pos_fd_fn)(ff_qpos)
        ftp_2_position = jax.vmap(allegro_fk.mftp_pos_fd_fn)(mf_qpos)
        ftp_3_position = jax.vmap(allegro_fk.rftp_pos_fd_fn)(rf_qpos)
        ftp_4_position = jax.vmap(allegro_fk.thtp_pos_fd_fn)(tm_qpos)
        
        # Compute costs
        position_cost = jnp.sum((obj_pose[:, :3] - self.goal_pos)**2, axis=1)
        quaternion_cost = 1 - jnp.sum(obj_pose[:, 3:7] * self.goal_quat, axis=1)**2
        contact_cost = (
            jnp.sum((obj_pose[:, :3] - ftp_1_position)**2, axis=1) +
            jnp.sum((obj_pose[:, :3] - ftp_2_position)**2, axis=1) +
            jnp.sum((obj_pose[:, :3] - ftp_3_position)**2, axis=1) +
            jnp.sum((obj_pose[:, :3] - ftp_4_position)**2, axis=1)
        )
        control_cost = jnp.sum(u**2, axis=1)
        
        return contact_cost + 0.1 * control_cost


    def compute_final_cost(self, x):
        obj_pose = x[:, :7]
        
        position_cost = jnp.sum((obj_pose[:, :3] - self.goal_pos)**2, axis=1)
        quaternion_cost = 1 - jnp.sum(obj_pose[:, 3:7] * self.goal_quat, axis=1)**2
        
        return 1000 * position_cost + 50 * quaternion_cost

    def init_cost_fns(self):
        self.path_cost_fn = self.compute_path_cost
        self.final_cost_fn = self.compute_final_cost


    