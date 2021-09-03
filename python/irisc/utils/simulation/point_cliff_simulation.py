""" creates a simulation with process and measurement noise implemented """


class PointCliffSimulator:
    def __init__(self, opt_dt, sim_dt, shooting_problem, problem_uncertainty):
        self.problem = shooting_problem
        self.uncertainty = problem_uncertainty
        self.horizon = self.problem.T 
        self.problem_dt = None 
        self.sim_dt = None 
        self.n_integration_steps = int(self.problem_dt/self.sim_dt)


    def step(self, x, u, pNoise=False, mNoise=False): 

        for _ in self.n_integration_steps: 
            pass 

        if pNoise: 
            pass 

        if mNoise: 
            pass 

        return x_next , y_next  