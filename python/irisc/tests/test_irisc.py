
""" Things I want to test 
1. allocateData dimensions 
2. backward pass for Horizon(T) step and (T-1) iteration 
3. estimation update 
4. simple line search 
5. gaps computation 
"""
import numpy as np 
import os, sys
src_path = os.path.abspath('../') 
sys.path.append(src_path)

from test_factory import create_point_cliff_models

from utils.uncertainty import measurement_models, process_models, problem_uncertainty
from solvers import irisc
import crocoddyl 

def test_initialization(): 
    dt = 0.01 
    horizon = 3
    models = create_point_cliff_models(dt, horizon) 

    x0 = np.zeros(4)
    initial_covariance = 1.e-4 * np.eye(4)
    uncertainty_models = []

    for i, m in enumerate(models[:-1]):
        # loop only over running models 
        process_noise = 1.e-5*np.eye(m.state.ndx)
        p_model = process_models.FullStateProcess(m, process_noise) 
        measurement_noise = 1.e-3*np.eye(m.state.ndx)
        m_model = measurement_models.FullStateMeasurement(m, measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]

 
    problem = crocoddyl.ShootingProblem(x0, models[:-1], models[-1])
    pUncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, uncertainty_models)
    sensitivity = 0.01 
    risk_solver = irisc.RiskSensitiveSolver(problem, pUncertainty, sensitivity, True)
    # system approximations 
    assert len(risk_solver.A) == horizon-1
    assert len(risk_solver.B) == horizon-1
    assert len(risk_solver.Omega) == horizon-1
    assert len(risk_solver.H) == horizon-1
    assert len(risk_solver.Gamma) == horizon-1
    # cost approximations 
    assert len(risk_solver.Q) == horizon
    assert len(risk_solver.q) == horizon
    assert len(risk_solver.S) == horizon-1
    assert len(risk_solver.R) == horizon-1
    assert len(risk_solver.r) == horizon-1
    # backward pass 
    assert len(risk_solver.M) == horizon
    assert len(risk_solver.kff) == horizon-1
    assert len(risk_solver.Kfb) == horizon-1
    assert len(risk_solver.V) == horizon
    assert len(risk_solver.v) == horizon
    # forward estimation 
    assert len(risk_solver.xhat) == horizon
    assert len(risk_solver.G) == horizon-1
    assert len(risk_solver.P) == horizon
    # recoupling 
    assert len(risk_solver.A) == horizon-1
    assert len(risk_solver.A) == horizon-1 



if __name__ == "__main__": 
    test_initialization()