# HSBC-Robust-Learning
Published Version of HSBC.

Packages Needed:
- mujoco (with mjx)
- brax
- jax
- pytorch

To run cartpole, walker and humanoid:

```
python test_cartpole/Cartpole_Robust_Align.py --dir {save_dir} --device {} --err {error rate} --rounds {batch_num} --opt Adam --dense True --model_num 16 --freq {checkpoint_freq}

python test_walker/Walker_Robust_Align.py --dir {save_dir} --device {} --err {error rate} --rounds {batch_num} --opt Adam --dense True --model_num 16 --freq {checkpoint_freq}

python test_humanoid/Humanoid_Robust_Align.py --dir {save_dir} --device {} --err {error rate} --rounds {batch_num} --opt Adam --dense True --model_num 16 --freq {checkpoint_freq}
```

To run dexterous manipulation:
```
python test_allegro/Allegro_Robust_Align_multitarget.py  --dir {save_dir} --obj {cube/bunny} --device {} --err {error rate} --rounds {batch_num} --opt Adam --dense True --model_num 16 --freq {checkpoint_freq}
```

To run go2-standup:
```
python test_go2/Go2_Robust_Align_free.py  --dir {save_dir} --device 1 --err {error rate} --rounds {batch_num} --opt Adam --dense True --model_num 16 --freq {checkpoint_freq}
```