import mujoco
import concurrent.futures
import threading
import threading
import numpy as np
from mujoco import rollout
import time



# model = mujoco.MjModel.from_xml_path('/home/wjin/projects/iris-env/tests/envs/xmls/humanoid.xml')
# model = mujoco.MjModel.from_xml_path('/home/wjin/projects/iris-env/dexhand/allegro/models/allegro_right_hand.xml')
model = mujoco.MjModel.from_xml_path('/home/wjin/projects/iris-env/dexhand/allegro/models/allegro_bunny.xml')
data = mujoco.MjData(model)

nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

nroll = 500
nstep = 20

# single thread testing
initial_state = np.random.randn(nroll, nstate)
control = 0.01*np.random.randn(nroll, nstep, model.nu)

st=time.time()
state, sensordata = rollout.rollout(model, data, initial_state, control)
print('500 rollouts, each with 10 time steps, run in single thread,  total time:', time.time()-st)
    



# multiple thread testing
num_workers = 64
state = np.empty((nroll, nstep, nstate))

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

st=time.time()
with concurrent.futures.ThreadPoolExecutor(
    max_workers=num_workers, initializer=thread_initializer) as executor:
    futures = []
    for chunk in chunks:
        futures.append(executor.submit(call_rollout, *chunk))
    for future in concurrent.futures.as_completed(futures):
        future.result()
print('500 rollouts, each with 10 time steps, run in 64 threads, total time:', time.time()-st)