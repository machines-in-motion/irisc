import numpy 
import os, sys
src_path = os.path.abspath('../') # append library directory without packaging 
sys.path.append(src_path)
from utils.action_models import point_cliff 
import crocoddyl 

def create_point_cliff_models(dt, horizon): 
    """ creates model of cliff point for testing purposes """
    cliff_diff_running =  point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, dt) 
    models = [cliff_running]*(horizon-1) + [cliff_terminal]
    return models 


