import subprocess
import os
import mujoco
from mujoco import mjx
import jax 


if subprocess.run('nvidia-smi').returncode:
    raise RuntimeError('NVIDIA driver is not installed')

# Configure MuJoCo to use the glfw rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'glfw'

try:
  print('Checking that the installation succeeded:')
  import mujoco
  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n')

print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

model_path_ = 'envs/xmls/env_allegro_bowl.xml'

mj_model = mujoco.MjModel.from_xml_path(model_path_)
mj_data = mujoco.MjData(mj_model)
# renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())



