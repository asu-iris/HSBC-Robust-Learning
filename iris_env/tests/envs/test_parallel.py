import mujoco
import concurrent.futures
import threading
import numpy as np
from mujoco import rollout


# model = mujoco.MjModel.from_xml_path('./envs/xmls/env_allegro_bowl.xml')
model = mujoco.MjModel.from_xml_path('./envs/xmls/env_allegro_airplane.xml')

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

# with concurrent.futures.ThreadPoolExecutor(
#     max_workers=num_workers, initializer=thread_initializer) as executor:
#     futures = []
#     for chunk in chunks:
#     futures.append(executor.submit(call_rollout, *chunk))
#     for future in concurrent.futures.as_completed(futures):
#     future.result()

