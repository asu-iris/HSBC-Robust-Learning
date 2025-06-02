from typing import Tuple

from brax.mjx.base import State
from brax.mjx import pipeline
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco


class AllegroOnly:
    def __init__(self, param):
        
        self.param=param
        
        # frame_skip
        self.n_frames=50
        
        # robot dim
        self.dim_robot_q=16
        
        # model path and model
        self.model_path = 'envs/allegro/models/allegro_right_hand.xml'
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        
        # brax syle system
        self.sys= mjcf.load_model(self.mj_model)
        
        
    def reset(self, qpos: jax.Array, qvel: jax.Array) -> State:
        return pipeline.init(self.sys, qpos, qvel)
    
    
    def step(self, state: State, action: jax.Array) -> State:
        
        # Calculate target joint positions
        target_qpos = state.q[0:self.dim_robot_q]+action

        def f(x, _):
            data = x.replace(ctrl=action)
            return (
                pipeline.step(self.sys, data),
                None,
            )
        return jax.lax.scan(f, state, (), self.n_frames)[0]
    


        
        
        
        
    