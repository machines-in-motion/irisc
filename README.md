# iRiSC: Iterative Risk Sensitive Optimal Control for Nonlinear Systems with Imperfect Observations

This repository contains the python implementation for the iterative risk sensitive solvers accounting for process and measurement uncertainties

## How To Run a Demo

The scripts available under `acc_2021/pneumatic_hopper` and `acc_2021/point_cliff` should be sufficient to get started, here are few tips for the **Pneumatic Hopper**

- run `acc/pneumatic_hopper/solve_ddp.py`, this generates a plan and stores it in `acc/pneumatic_hopper/solutions`. 
- similarly you can select the **sensitivity** parameter in  `acc/pneumatic_hopper/config_pneumatic_hopper.py` and run `acc/pneumatic_hopper/solve_firisk.py` to obtain and store the risk sensitive solutions. 
- once solutions are obtained, simulation with estimation and noise can be run with either `simulate_ilqg.py` or `simulate_risk_partial.py`  



## Dependencies

- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [Crocoddyl](https://github.com/loco-3d/crocoddyl)






