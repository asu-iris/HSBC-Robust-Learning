# ==============================================================================
"""tests for rollout function."""

import concurrent.futures
import threading

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco import rollout
import numpy as np

# -------------------------- models used for testing ---------------------------

TEST_XML = r"""
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <geom type="box" pos="1 0 0" size=".1 .1 .1"/>
      <site name="site" pos="1 0 0"/>
    </body>∏
  </worldbody>
  <actuator>
    <general joint="pitch" gainprm="100"/>
    <general joint="yaw" dyntype="filter" dynprm="1" gainprm="100"/>
  </actuator>
  <sensor>
    <accelerometer site="site"/>
  </sensor>
</mujoco>
"""

TEST_XML_NO_SENSORS = r"""
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <geom type="box" pos="1 0 0" size=".1 .1 .1"/>
      <site name="site" pos="1 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="pitch" gainprm="100"/>
    <general joint="yaw" dyntype="filter" dynprm="1" gainprm="100"/>
  </actuator>
</mujoco>
"""

TEST_XML_NO_ACTUATORS = r"""
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <geom type="box" pos="1 0 0" size=".1 .1 .1"/>
      <site name="site" pos="1 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer site="site"/>
  </sensor>
</mujoco>
"""

TEST_XML_MOCAP = r"""
<mujoco>
  <worldbody>
    <body name="1" mocap="true">
    </body>
    <body name="2" mocap="true">
    </body>
  </worldbody>
  <sensor>
    <framepos objtype="xbody" objname="1"/>
    <framequat objtype="xbody" objname="1"/>
  </sensor>
</mujoco>
"""

TEST_XML_EMPTY = r"""
<mujoco>
</mujoco>
"""

TEST_XML_DIVERGE = r"""
<mujoco>
  <option>
    <flag gravity="disable"/>
  </option>

  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 -.3" euler="30 45 90">
      <freejoint/>
      <geom type="box" size=".1 .2 .4"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="non-diverging" qpos="0 0 .5 1 0 0 0"/>
  </keyframe>
</mujoco>
"""

ALL_MODELS = {'TEST_XML': TEST_XML,
              'TEST_XML_NO_SENSORS': TEST_XML_NO_SENSORS,
              'TEST_XML_NO_ACTUATORS': TEST_XML_NO_ACTUATORS,
              'TEST_XML_EMPTY': TEST_XML_EMPTY}

# ------------------------------ tests -----------------------------------------


class MuJoCoRolloutTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(42)

  # ----------------------------- test basic operation

  @parameterized.parameters(ALL_MODELS.keys())
  def test_single_step(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    initial_state = np.random.randn(nstate)
    control = np.random.randn(model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)


  @parameterized.parameters(ALL_MODELS.keys())
  def test_one_rollout(self, model_name):
    nstep = 3  # number of timesteps

    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    initial_state = np.random.randn(nstate)
    control = np.random.randn(nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)


  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_step(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = np.random.randn(nroll, nstate)
    control = np.random.randn(nroll, nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    mujoco.mj_resetData(model, data)


  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nroll_initial_state(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = np.random.randn(nroll, nstate)
    control = np.random.randn(nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)

    mujoco.mj_resetData(model, data)
    control = np.tile(control, (nroll, 1, 1))

  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nroll_control(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = np.random.randn(nstate)
    control = np.random.randn(nroll, nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)



  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nroll_warmstart(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = np.random.randn(nstate)
    control = np.random.randn(nstep, model.nu)
    initial_warmstart = np.tile(data.qacc_warmstart.copy(), (nroll, 1))
    state, sensordata = rollout.rollout(model, data, initial_state, control,
                                        initial_warmstart=initial_warmstart)



  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nroll_state(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = np.random.randn(nstate)
    control = np.random.randn(nstep, model.nu)
    state = np.empty((nroll, nstep, nstate))
    state, sensordata = rollout.rollout(model, data, initial_state, control,
                                        state=state)



  @parameterized.parameters(ALL_MODELS.keys())
  def test_infer_nroll_sensordata(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 5  # number of rollouts
    nstep = 1  # number of steps

    initial_state = np.random.randn(nstate)
    control = np.random.randn(nstep, model.nu)
    sensordata = np.empty((nroll, nstep, model.nsensordata))
    state, sensordata = rollout.rollout(model, data, initial_state, control,
                                        sensordata=sensordata)



  @parameterized.parameters(ALL_MODELS.keys())
  def test_one_rollout_fixed_ctrl(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 1  # number of rollouts
    nstep = 3  # number of steps

    initial_state = np.random.randn(nstate)
    control = np.random.randn(model.nu)
    state = np.empty((nroll, nstep, nstate))
    sensordata = np.empty((nroll, nstep, model.nsensordata))
    rollout.rollout(model, data, initial_state, control,
                    state=state, sensordata=sensordata)



  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_rollout(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 2  # number of initial states
    nstep = 3  # number of timesteps

    initial_state = np.random.randn(nroll, nstate)
    control = np.random.randn(nroll, nstep, model.nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)


  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_model(self, model_name):
    nroll = 3  # number of initial states and models
    nstep = 3  # number of timesteps

    spec = mujoco.MjSpec.from_string(ALL_MODELS[model_name])

    if len(spec.bodies) > 1:
      model = []
      for i in range(nroll):
        body = spec.bodies[1]
        assert body.name != 'world'
        body.pos = body.pos + i
        model.append(spec.compile())
    else:
      model = [spec.compile() for i in range(nroll)]

    nstate = mujoco.mj_stateSize(model[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model[0])

    initial_state = np.random.randn(nroll, nstate)
    control = np.random.randn(nroll, nstep, model[0].nu)
    state, sensordata = rollout.rollout(model, data, initial_state, control)



  @parameterized.parameters(ALL_MODELS.keys())
  def test_multi_rollout_fixed_ctrl_infer_from_output(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 2  # number of rollouts
    nstep = 3  # number of timesteps

    initial_state = np.random.randn(nroll, nstate)
    control = np.random.randn(nroll, 1, model.nu)
    state = np.empty((nroll, nstep, nstate))
    state, sensordata = rollout.rollout(model, data, initial_state, control,
                                        state=state)


  @parameterized.parameters(ALL_MODELS.keys())
  def test_py_rollout_generalized_control(self, model_name):
    model = mujoco.MjModel.from_xml_string(ALL_MODELS[model_name])
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)

    nroll = 4  # number of rollouts
    nstep = 3  # number of timesteps

    initial_state = np.random.randn(nroll, nstate)

    control_spec = (mujoco.mjtState.mjSTATE_CTRL |
                    mujoco.mjtState.mjSTATE_QFRC_APPLIED |
                    mujoco.mjtState.mjSTATE_XFRC_APPLIED)
    ncontrol = mujoco.mj_stateSize(model, control_spec)
    control = np.random.randn(nroll, nstep, ncontrol)

    state, sensordata = rollout.rollout(model, data, initial_state, control,
                                        control_spec=control_spec)



  # ----------------------------- test threaded operation

  def test_threading(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    num_workers = 32
    nroll = 10000
    nstep = 5
    initial_state = np.random.randn(nroll, nstate)
    state = np.empty((nroll, nstep, nstate))
    sensordata = np.empty((nroll, nstep, model.nsensordata))
    control = np.random.randn(nroll, nstep, model.nu)

    thread_local = threading.local()

    def thread_initializer():
      thread_local.data = mujoco.MjData(model)

    model_list = [model] * nroll
    def call_rollout(initial_state, control, state, sensordata):
      rollout.rollout(model_list, thread_local.data, initial_state, control,
                      skip_checks=True,
                      nstep=nstep, state=state, sensordata=sensordata)

    n = nroll // num_workers  # integer division
    chunks = []  # a list of tuples, one per worker
    for i in range(num_workers-1):
      chunks.append((initial_state[i*n:(i+1)*n],
                     control[i*n:(i+1)*n],
                     state[i*n:(i+1)*n],
                     sensordata[i*n:(i+1)*n]))

    # last chunk, absorbing the remainder:
    chunks.append((initial_state[(num_workers-1)*n:],
                   control[(num_workers-1)*n:],
                   state[(num_workers-1)*n:],
                   sensordata[(num_workers-1)*n:]))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers, initializer=thread_initializer) as executor:
      futures = []
      for chunk in chunks:
        futures.append(executor.submit(call_rollout, *chunk))
      for future in concurrent.futures.as_completed(futures):
        future.result()




  # ---------------------------- test advanced operation




if __name__ == '__main__':
  absltest.main()