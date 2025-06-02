#export CUDA_VISIBLE_DEVICES='3'
# python test_cartpole/Cartpole_Robust_Align.py --dir Data/Cartpole/RA/error_2/run_0 --device 0 --err 2 --rounds 51 --opt Adam --dense True --model_num 16 --freq 2
# python test_cartpole/Cartpole_Robust_Align.py --dir Data/Cartpole/RA/error_2/run_1 --device 0 --err 2 --rounds 51 --opt Adam --dense True --model_num 16 --freq 2
# python test_cartpole/Cartpole_Robust_Align.py --dir Data/Cartpole/RA/error_2/run_2 --device 0 --err 2 --rounds 51 --opt Adam --dense True --model_num 16 --freq 2
# python test_cartpole/Cartpole_Robust_Align.py --dir Data/Cartpole/RA/error_2/run_3 --device 0 --err 2 --rounds 51 --opt Adam --dense True --model_num 16 --freq 2
# python test_cartpole/Cartpole_Robust_Align.py --dir Data/Cartpole/RA/error_2/run_4 --device 0 --err 2 --rounds 51 --opt Adam --dense True --model_num 16 --freq 2

python test_cartpole/Cartpole_Recorded.py --dir Data/Human_fb_confirm/Cartpole_2 --device 3 --err 4 --rounds 11  --dense True --model_num 16 --freq 1
python test_cartpole/Cartpole_Recorded.py --dir Data/Human_fb_confirm/Cartpole_3 --device 3 --err 4 --rounds 11  --dense True --model_num 16 --freq 1
python test_cartpole/Cartpole_Recorded.py --dir Data/Human_fb_confirm/Cartpole_4 --device 3 --err 4 --rounds 11  --dense True --model_num 16 --freq 1
python test_cartpole/Cartpole_Recorded.py --dir Data/Human_fb_confirm/Cartpole_5 --device 3 --err 4 --rounds 11  --dense True --model_num 16 --freq 1
python test_cartpole/Cartpole_Recorded.py --dir Data/Human_fb_confirm/Cartpole_6 --device 3 --err 4 --rounds 11  --dense True --model_num 16 --freq 1
python test_cartpole/Cartpole_Recorded.py --dir Data/Human_fb_confirm/Cartpole_7 --device 3 --err 4 --rounds 11  --dense True --model_num 16 --freq 1