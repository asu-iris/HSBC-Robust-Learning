export CUDA_VISIBLE_DEVICES='1'
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_2/run_0 --device 1 --err 2 --rounds 151 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_2/run_1 --device 1 --err 2 --rounds 151 --opt Adam --dense True --model_num 16 --freq 5

# python test_go2/Go2_Robust_Align.py  --dir Data/Go2/RA/error_0/run_1 --device 0 --err 0 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
# python test_go2/Go2_Robust_Align.py  --dir Data/Go2/RA/error_0/run_2 --device 0 --err 0 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
# python test_go2/Go2_Robust_Align.py  --dir Data/Go2/RA/error_0/run_3 --device 0 --err 0 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5
# python test_go2/Go2_Robust_Align.py  --dir Data/Go2/RA/error_0/run_4 --device 0 --err 0 --rounds 101 --opt Adam --dense True --model_num 16 --freq 5