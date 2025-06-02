## IRIS physics environments (lab use only)

This repo is mainly for GPU-accelerated simulation. We are trying to standardize the APIs of each  environment and make it lightweight and flexible to be able to included in  sampling-based model predictive control (MPC). 

### Structure

- `./gym_like/`: includes the environments commonly seen in reinforcement learning. The following environments should work.
    - [x] Cartpole
    - [x] Ant
    - [x] Half_cheetah

- `./dexhand/` includes the environments of dexterous manpulation. The following environments should work.
    - [x] Allegro_object (fast)


If you contribute new environment, please follow the API style defined in [allegro.py](./dexhand/allegro/allegro.py), and put it in "gym",  "dexhand", or "quadruped" folder. 


While I am still trying to figure out what's the optimal API for our above purpose, but the style in [allegro.py](./dexhand/allegro/allegro.py) looks quite flexible for now.



### Tricks to significantly speed up!
Here are some tricks i found very useful to speed up your mjx simulation speed:
* If you use sampling-based MPC, you can set the simulation timestep (MJX/MuJoCo default is 0.002s, 500Hz) of your MPC planner to a larger one while your env is the default one. Example is the line 52: `planner=AllegroObject(param, timestep=0.01)` in [allegro_object_test.py](./dexhand/allegro/allegro_object_test.py). This means that your MPC horizon can be set shorter and thus MPC faster!

* In your xml file, add those options

    ```  
    <option cone="pyramidal" ls_iterations="5">
    </option>

    <custom>
        <numeric data="15" name="max_contact_points"/>
        <numeric data="15" name="max_geom_pairs"/>
    </custom>
    ```
    This options enforce the max contact count at each simulation step, and limit the per-step solver search iteraction.

* Modify your xml file to use primitive shapes for geom collision (`type="box"` collision geom is quite time-consuming for contact-rich environment). But do this with causion because your primitive collision geom should be similar to your actual geom. 

For example, by applying the above tricks, i can speep up the [allegro_object_test.py](./dexhand/allegro/allegro_object_test.py)

```
single environment 10 frameskips (i.e., 10 simulation steps) from 0.03s --> 0.01s
512 environments 10 frameskips (i.e., 10 simulation steps) from 0.1s --> 0.015s
```