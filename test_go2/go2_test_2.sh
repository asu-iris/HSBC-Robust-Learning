export CUDA_VISIBLE_DEVICES='2'
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_0/run_2 --device 2 --err 0 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_0/run_3 --device 2 --err 0 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_0/run_4 --device 2 --err 0 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5