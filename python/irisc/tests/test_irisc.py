
""" Things I want to test 
1. allocateData dimensions 
2. backward pass for Horizon(T) step and (T-1) iteration 
3. estimation update 
4. simple line search 
5. gaps computation 
"""

import numpy as np 
import os, sys
src_path = os.path.abspath('../') # append library directory without packaging 
sys.path.append(src_path)

import crocoddyl 
from test_factory import create_point_cliff_models
from solvers.irisc import RiskSensitiveSolver 