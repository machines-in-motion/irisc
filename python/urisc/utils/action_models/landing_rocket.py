""" here we will have two different dynamics 
Nominal for Optimization 
Sloch for actual simulation 
model inspired from "Direct Policy Optimization Using Deterministic Sampling and Collocation" by Howell et. al. 
link: https://ieeexplore.ieee.org/abstract/document/9387078 
"""



import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 

class NominalRocket2D:
    def __init__(self) -> None:
        pass


class SlochRocket2D:
    def __init__(self) -> None:
        pass



class DifferentialActionModelRocket2D(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, t, T, isTerminal=False):
        pass 