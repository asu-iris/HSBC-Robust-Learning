# python tests/Humanoid_Robust_Align.py  --dir Data/Humanoid/RA/error_0/run_0 --device 1 --err 0 --opt Adam --dense True --model_num 8
# python tests/Humanoid_Robust_Align.py  --dir Data/Humanoid/RA/error_2/run_0 --device 1 --err 2 --opt Adam --dense True --model_num 8
# python tests/Humanoid_Robust_Align.py  --dir Data/Humanoid/RA/error_0/run_4_minimal --device 2 --err 0 --opt Adam --dense True --model_num 16
# python tests/Humanoid_Robust_Align.py  --dir Data/Humanoid/RA/error_0/run_6_minimal --device 2 --err 0 --opt Adam --dense True --model_num 16
#python tests/Humanoid_Robust_Align.py  --dir Data/Humanoid/RA/error_0/run_4_16 --device 0 --err 0 --opt Adam --dense True --model_num 16
export CUDA_VISIBLE_DEVICES='0'
python test_humanoid/Humanoid_Robust_Align.py  --dir Data/Humanoid_New/RA/error_1/run_0 --device 0 --err 1 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
python test_humanoid/Humanoid_Robust_Align.py  --dir Data/Humanoid_New/RA/error_1/run_1 --device 0 --err 1 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
python test_humanoid/Humanoid_Robust_Align.py  --dir Data/Humanoid_New/RA/error_1/run_2 --device 0 --err 1 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
python test_humanoid/Humanoid_Robust_Align.py  --dir Data/Humanoid_New/RA/error_1/run_3 --device 0 --err 1 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
python test_humanoid/Humanoid_Robust_Align.py  --dir Data/Humanoid_New/RA/error_1/run_4 --device 0 --err 1 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5