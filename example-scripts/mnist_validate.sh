#!/bin/sh
python ../validate.py --data "mnist" --solver "rk4" --kinetic-energy 0.01 --jacobian-norm2 0.01 --alpha 1e-5 --test_solver dopri5 --test_atol 1e-5 --test_rtol 1e-5 --step_size 0.25 --chkpt "/HPS/CNF/work/ffjord-rnode/experiments/mnist/example/checkpt.pth"
