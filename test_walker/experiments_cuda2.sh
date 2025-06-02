# python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_2/run_1 --device 2 --err 2 --opt Adam
# python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_2/run_2 --device 2 --err 2 --opt Adam
# python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_2/run_3_dense --device 1 --err 2 --opt Adam --dense True
# python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_2/run_4_dense --device 1 --err 2 --opt Adam --dense True

python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_0/run_1_dense --device 2 --err 0 --opt Adam --dense True
python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_2/run_5_dense --device 2 --err 2 --opt Adam --dense True
python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_2/run_6_mcmc --device 2 --err 2 --opt MCMC --dense True
python tests/Walker_Robust_Align.py --dir Data/Walker_Robust/RA/error_2/run_7_mcmc --device 2 --err 2 --opt MCMC --dense True  