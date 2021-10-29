""" Creates a class that can construct multiple control problems for the robot Solo12 """ 


import numpy as np 
import crocoddyl 
import pinocchio as pin 
import os, sys, time 
src_path = os.path.abspath('../../../')
sys.path.append(src_path)
from utils.uncertainty import measurement_models, process_models, problem_uncertainty



class Solo12Gaits:
    def __init__(self, robot, contact_names):
        self.robot = robot 
        self.rmodel = self.robot.model 
        self.rdata = self.rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.contact_names = contact_names 
        self.contact_ids = [self.rmodel.getFrameId(n) for n in self.contact_names] 
        # Defining the friction coefficient and normal
        self.mu = 0.3
        self.nsurf = np.eye(3) 
        self.baumgarte = np.array([0., 50.]) # pd gains to stabalize the contact 

    def createBalanceProblem(self, balanceConf): 
        """ creates a balance problem around x0 """
        loco3dModels = []
        uncertaintyModels = []
        pin.framesForwardKinematics(self.rmodel, self.rdata, balanceConf.x0[:self.state.nq])
        comPos = pin.centerOfMass(self.rmodel, self.rdata, balanceConf.x0[:self.state.nq])
        for t in range(balanceConf.horizon+1):
            costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
            # CoM cost 
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comPos, self.actuation.nu)
            comCost = crocoddyl.CostModelResidual(self.state, crocoddyl.ActivationModelQuad(3), comResidual)
            # comTrack = crocoddyl.CostModelCoMPosition(self.state, comPos, self.actuation.nu)
            costModel.addCost("comTrack", comCost, 1.e+5)

            # contact model and cost 
            contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu) 
            for i, frame_id in enumerate(self.contact_ids): 
                footRef = self.rdata.oMf[frame_id].translation
                # Contact Constraint 
                footContactModel = crocoddyl.ContactModel3D(self.state, frame_id, footRef, self.actuation.nu, 
                                                                self.baumgarte)
                contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", footContactModel, True)
                # friction cone  Cost
                cone_rotation = self.rdata.oMf[frame_id].rotation.T.dot(self.nsurf)
                frictionCone = crocoddyl.FrictionCone(cone_rotation, self.mu, 4, True)#, 0., 1000.)
                bounds = crocoddyl.ActivationBounds(frictionCone.lb, frictionCone.ub)
                frictionResiduals = crocoddyl.ResidualModelContactFrictionCone(self.state, frame_id, frictionCone, self.actuation.nu)
                frictionCost = crocoddyl.CostModelResidual(self.state,
                crocoddyl.ActivationModelQuadraticBarrier(bounds), frictionResiduals) 
                costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCost, 1.e+1) 
               

            stateWeights =[1.e-1] * 3 + [1.e-1] * 3 + [1.e-2, 1.e-2, 1.e-2] * 4 # (self.rmodel.nv - 6)
            stateWeights += [1.e-1] * 6 + [1.e-2] * (self.rmodel.nv - 6)
            # 

            stateResiduals = crocoddyl.ResidualModelState(self.state, balanceConf.x0, self.actuation.nu)
            stateCost = crocoddyl.CostModelResidual(self.state,
                        crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights)**2),
                        stateResiduals)
            costModel.addCost("stateReg", stateCost, 1.)
            if t == balanceConf.horizon:
                pass 
            else:
                controlResiduals = crocoddyl.ResidualModelControl(self.state, np.zeros(self.actuation.nu))
                controlCost = crocoddyl.CostModelResidual(self.state, crocoddyl.ActivationModelQuad(self.actuation.nu),controlResiduals)     
                costModel.addCost("ctrlReg", controlCost, 1.e-3)
            # # differential ocp model 
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, 
            self.actuation, contactModel, costModel, 0., True) 
            loco3dModels += [crocoddyl.IntegratedActionModelEuler(dmodel, balanceConf.timeStep)]
            # # 
            # """ Creating the Measurement Models """
            if t == 0:
                pass 
            else:
                if balanceConf.measurementModel is not None:
                    if balanceConf.measurementModel=="FullStateUniform":
                        p_model = process_models.FullStateProcess(loco3dModels[-1], np.sqrt(balanceConf.timeStep)*balanceConf.process_noise) 
                        m_model = measurement_models.FullStateMeasurement(loco3dModels[-1], np.sqrt(balanceConf.timeStep)*balanceConf.measurement_noise)
                        uncertaintyModels += [problem_uncertainty.UncertaintyModel(p_model, m_model)]
                    else:
                        raise BaseException("Measurement Model Not Recognized")
        return loco3dModels, uncertaintyModels