""" tests derivatives of differential models against finite differencing """
import numpy as np 
import os, sys
src_path = os.path.abspath('../') # append library directory without packaging 
sys.path.append(src_path)

import crocoddyl 
from test_factory import create_point_cliff_models
from utils.finite_differences import action_model_derivatives

def test_point_cliff_derivatives(): 
    dt = 0.01 
    horizon = 1  
    models = create_point_cliff_models(dt, horizon)
    cost_numdiff = []
    dynamics_numdiff  = []
    datas = []
    for m in models:
        datas += [crocoddyl.IntegratedActionDataEuler(m)]
        cost_numdiff += [action_model_derivatives.CostNumDiff(m)]
        dynamics_numdiff += [action_model_derivatives.DynamicsNumDiff(m)]


    for i, m in enumerate(models): 
        # if i == 0:
        #     continue
        x = m.state.rand() # generates a random state 
        u = np.random.rand(m.nu) # generates a random control 
        # calculate derivatives of IntegratedActionModel 
        m.calc(datas[i], x,u) 
        m.calcDiff(datas[i], x,u)
        # COST TESTING
        # cost_numdiff[i].calcLx(x,u)
        # np.testing.assert_almost_equal(datas[i].Lx, cost_numdiff[i].Lx , decimal=7, err_msg="Error in Lx for model %s"%(i+1))
        # cost_numdiff[i].calcLu(x,u)
        # np.testing.assert_almost_equal(datas[i].Lu, cost_numdiff[i].Lu , decimal=7, err_msg="Error in Lu for model  %s"%(i+1))
        # cost_numdiff[i].calcLxx(x,u)
        # np.testing.assert_almost_equal(datas[i].Lxx, cost_numdiff[i].Lxx , decimal=7, err_msg="Error in Lxx for model %s"%(i+1))
        # cost_numdiff[i].calcLuu(x,u)
        # np.testing.assert_almost_equal(datas[i].Luu, cost_numdiff[i].Luu , decimal=7, err_msg="Error in Luu for model %s"%(i+1))
        # cost_numdiff[i].calcLxu(x,u)
        # np.testing.assert_almost_equal(datas[i].Lxu, cost_numdiff[i].Lxu , decimal=7, err_msg="Error in Lxu for model %s"%(i+1))
        # DYNAMICS TESTING 
        dynamics_numdiff[i].calcFx(x,u) 
        np.testing.assert_almost_equal(datas[i].Fx, dynamics_numdiff[i].Fx , decimal=7, err_msg="Error in Fx for model %s"%(i+1))
        dynamics_numdiff[i].calcFu(x,u) 
        np.testing.assert_almost_equal(datas[i].Fu, dynamics_numdiff[i].Fu , decimal=7, err_msg="Error in Fu for model %s"%(i+1))

if __name__ == "__main__":
    test_point_cliff_derivatives() 