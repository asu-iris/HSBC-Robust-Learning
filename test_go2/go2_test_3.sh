export CUDA_VISIBLE_DEVICES='3'
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_1/run_0 --device 3 --err 1 --rounds 81 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_1/run_1 --device 3 --err 1 --rounds 81 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_1/run_2 --device 3 --err 1 --rounds 81 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_1/run_3 --device 3 --err 1 --rounds 81 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_1/run_4 --device 3 --err 1 --rounds 81 --opt Adam --dense True --model_num 16 --freq 5

python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_3/run_0 --device 3 --err 3 --rounds 121 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_3/run_1 --device 3 --err 3 --rounds 121 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_3/run_2 --device 3 --err 3 --rounds 121 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_3/run_3 --device 3 --err 3 --rounds 121 --opt Adam --dense True --model_num 16 --freq 5
python test_go2/Go2_Robust_Align_free.py  --dir Data/Go2_big/RA/error_3/run_4 --device 3 --err 3 --rounds 121 --opt Adam --dense True --model_num 16 --freq 5