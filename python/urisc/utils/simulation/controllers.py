import numpy as np 


class AbstractController:
    def __init__(self, action_models):
        self.action_models = action_models
        self.states = []
        for m in self.action_models: 
            self.states += [m.state]
            

class DDPController(AbstractController):
    def __init__(self, action_models):
        super().__init__(action_models)










class UscentedRiskController(AbstractController):
    def __init__(self, action_models):
        super().__init__(action_models)