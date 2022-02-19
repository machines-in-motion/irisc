""" this model is the same one as section 8.1.1 in the book titled: Modern Robotics by Lynch 

we will start with no contact force, just moving the manipulator 
then we will figure out the rest 
"""


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 
DELTA = 1.e-6 # numerical differentiation step 

class TwoLinkManipulator:
    def __init__(self):
        self.g = -9.81 
        self.l1 = 1. 
        self.m1 = 2.
        self.l2 = 1. 
        self.m2 = 2.

        self.nq = 2
        self.nv = 2 
        self.nx = self.nq + self.nv 
        self.ndx = 2*self.nv  
        self.nu = 2 

    def position_kinematics(self, x):
        p1, p2 = np.zeros(2), np.zeros(2) 
        p1[0] = self.l1 * np.cos(x[0])
        p1[1] = self.l1 * np.sin(x[0])
        p2[0] = p1[0] + self.l2 * np.cos(x[0] + x[1])
        p2[1] = p1[1] + self.l2 * np.sin(x[0] + x[1])
        return p1, p2 

    def velocity_kinematics(self, x):
        dp1, dp2 = np.zeros(2), np.zeros(2)
        dp1[0] = -self.l1 * np.sin(x[0])*x[2]
        dp1[1] = self.l1 * np.cos(x[0])*x[2]
        dp2[0] = dp1[0] - self.l2*np.sin(x[0] + x[1])*(x[2]+ x[3])
        dp2[1] = dp1[1] + self.l2*np.cos(x[0] + x[1])*(x[2]+ x[3])
        return dp1, dp2 

    def mass_matrix(self, x):
        l1s = self.l1**2 
        l2s = self.l2**2 
        l1l2c2 = self.l1*self.l2*np.cos(x[1])

        m = np.zeros((2,2)) 
        m[0,0] = self.m1*l1s + self.m2*(l1s + 2*l1l2c2 + l2s)
        m[0,1] = self.m2*(l1l2c2 + l2s)
        m[1,0] = m[0,1]
        m[1,1] = self.m2 * l2s
        return m   
 
    def coriolis_forces(self, x):
        f = np.array([ -self.m2*self.l1*self.l2*np.sin(x[1])*(2*x[2]*x[3] + x[3]**2), 
                        self.m2*self.l1*self.l2*np.sin(x[1])*(x[2]**2) ])
        return f  

    def gravity_vector(self, x):
        f = np.array([
           (self.m1 + self.m2)*self.l1*self.g*np.cos(x[0]) + self.m2*self.g*self.l2*np.cos(x[0]+x[1]), 
           self.m2*self.g*self.l2*np.cos(x[0]+x[1]) 
        ]) 
        return f 

    def inv_mass_matrix(self, x):
        m = self.mass_matrix(x)
        det_m = (m[0,0]*m[1,1]) - (m[0,1]*m[1,0])
        inv_det = 1/det_m
        minv = np.zeros((2,2))
        minv[0,0] = m[1,1]
        minv[0,1] = -m[0,1]
        minv[1,0] = -m[1,0]
        minv[1,1] = m[0,0]
        return inv_det*minv 

    def nle(self, x):
        return self.coriolis_forces(x) + self.gravity_vector(x)

    def ee_jacobian(self, x):
        J = np.zeros((2,2))
        J[0,0] = - self.l1 * np.sin(x[0]) - self.l2 * np.sin(x[0]+ x[1])
        J[0,1] = - self.l2 * np.sin(x[0]+ x[1])
        J[1,0] = self.l1 * np.cos(x[0]) + self.l2 * np.cos(x[0]+ x[1])
        J[1,1] = self.l2 * np.cos(x[0]+ x[1])
        return J 

    def nonlinear_dynamics(self, x, u):
        acc = self.inv_mass_matrix(x).dot(u - self.nle(x))
        return acc

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u """
        dfdx = np.zeros([self.nv ,self.ndx])
        dfdu = np.zeros([self.nv ,self.nu])

        dx = np.zeros(self.ndx)
        for i in range(self.ndx):
            dx[i] = DELTA
            acc1 = self.nonlinear_dynamics(x+dx, u)
            acc2 = self.nonlinear_dynamics(x-dx, u)
            dfdx[:,i] = acc2 - acc1
            dx[i] = 0. 
        dfdx *= 1./(2.*DELTA)

        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = DELTA
            acc1 = self.nonlinear_dynamics(x, u + du)
            acc2 = self.nonlinear_dynamics(x, u - du)
            dfdu[:,i] = acc2 - acc1
            du[i] = 0. 
        dfdu *= 1./(2.*DELTA)
        return dfdx, dfdu 

    def discrete_dynamics(self, x, u, dt):
        """ computes state transitions for a given dt """
        dv = self.nonlinear_dynamics(x,u) 
        qnext = x[:2] + dt*x[2:] + .5*dv*dt**2 
        vnext = x[2:] + dt*dv 
        return qnext, vnext 


class DifferentialActionTwoLinkManipulator(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, xref=None, isTerminal=False):
        self.dynamics = TwoLinkManipulator()
        state =  crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, self.dynamics.nu, self.dynamics.ndx)
        self.isTerminal = isTerminal
        if xref is None:
            self.xref = np.zeros(self.state.nx)
        else:
            self.xref = xref 
        #
        self.w1 = 1.e-2*np.eye(self.state.ndx) # state reg 
        self.w1[2,2], self.w1[3,3] = 1.e+3, 41.e+3 
        self.w2 = 1.e+2*np.eye(self.nu) # control reg 
        self.w3 = 1.e+2*np.eye(self.state.ndx) # ee pos & vel terminal 
        self.w4 = 2*np.eye(self.state.ndx) # ee pos & vel running 
       
    def _running_cost(self, x, u): 
        _, ee_p = self.dynamics.position_kinematics(x)
        _, ee_dp = self.dynamics.velocity_kinematics(x)
        ee_state = np.hstack([ee_p, ee_dp])
        cost = .5*(ee_state-self.xref).T.dot(self.w4).dot(ee_state-self.xref)
        cost += .5*(x).T.dot(self.w1).dot(x)
        cost += .5*u.T.dot(self.w2).dot(u)
        return cost 
    
    def _terminal_cost(self, x):
        _, ee_p = self.dynamics.position_kinematics(x)
        _, ee_dp = self.dynamics.velocity_kinematics(x)
        ee_state = np.hstack([ee_p, ee_dp])
        cost = .5*(ee_state-self.xref).T.dot(self.w3).dot(ee_state-self.xref)
        return cost 

    def terminal_cost_derivatives(self, x):
        """ compute the gradient and hessian wrt the state x[th1, th2, dth1, dth2] 
        """

        Lx = np.zeros([self.state.ndx])
        Lxx = np.zeros([self.state.ndx, self.state.ndx])

        dx = np.zeros(self.state.ndx)
        for i in range(self.state.ndx):
            dx[i] = DELTA
            xnew = self.state.integrate(x, dx)
            cost1 = self._terminal_cost(xnew)
            xnew = self.state.integrate(x, -dx)
            cost2 = self._terminal_cost(xnew)
            Lx[i] = cost1 - cost2 
            dx[i] = 0. 
        Lx *= 1./(2.*DELTA) 

        dxi = np.zeros(self.state.ndx)
        dxj = np.zeros(self.state.ndx)

        for i in range(self.state.ndx):
            for j in range(self.state.ndx): 
                dxi[i] = DELTA
                dxj[j] = DELTA
                xnew = self.state.integrate(x, dxi + dxj)
                cost1 = self._terminal_cost(xnew)
                xnew = self.state.integrate(x, dxi - dxj)
                cost2 = self._terminal_cost(xnew)
                xnew = self.state.integrate(x, -dxi + dxj)
                cost3 = self._terminal_cost(xnew)
                xnew = self.state.integrate(x, -dxi - dxj)
                cost4 = self._terminal_cost(xnew)
                Lxx[i,j] = cost1 - cost2 - cost3 + cost4 
                dxi[i] = 0.
                dxj[j] = 0.
        Lxx = .5*(Lxx + Lxx.T)
        Lxx *= 1./(4.*DELTA*DELTA)
        return Lx, Lxx 

    def running_cost_state_derivative(self, x, u):
        """ compute the gradient and hessian wrt the state x[th1, th2, dth1, dth2] 
        """

        Lx = np.zeros([self.state.ndx])
        Lxx = np.zeros([self.state.ndx, self.state.ndx])

        dx = np.zeros(self.state.ndx)
        for i in range(self.state.ndx):
            dx[i] = DELTA
            xnew = self.state.integrate(x, dx)
            cost1 = self._running_cost(xnew, u)
            xnew = self.state.integrate(x, -dx)
            cost2 = self._running_cost(xnew, u)
            Lx[i] = cost1 - cost2 
            dx[i] = 0. 
        Lx *= 1./(2.*DELTA) 

        dxi = np.zeros(self.state.ndx)
        dxj = np.zeros(self.state.ndx)

        for i in range(self.state.ndx):
            for j in range(self.state.ndx): 
                dxi[i] = DELTA
                dxj[j] = DELTA
                xnew = self.state.integrate(x, dxi + dxj)
                cost1 = self._running_cost(xnew, u)
                xnew = self.state.integrate(x, dxi - dxj)
                cost2 = self._running_cost(xnew, u)
                xnew = self.state.integrate(x, -dxi + dxj)
                cost3 = self._running_cost(xnew, u)
                xnew = self.state.integrate(x, -dxi - dxj)
                cost4 = self._running_cost(xnew, u)
                Lxx[i,j] = cost1 - cost2 - cost3 + cost4 
                dxi[i] = 0.
                dxj[j] = 0.
        Lxx = .5*(Lxx + Lxx.T)
        Lxx *= 1./(4.*DELTA*DELTA)
        return Lx, Lxx 

    def calc(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        #
        if self.isTerminal: 
            data.cost = self._terminal_cost(x) 
            data.xout = np.zeros(self.state.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xout[:] = self.dynamics.nonlinear_dynamics(x,u)


    def calcDiff(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        #
        Fx = np.zeros([self.state.nv,self.state.ndx]) 
        Fu = np.zeros([self.state.nv,self.nu])
        Lx = np.zeros([self.state.ndx])
        Lu = np.zeros([self.nu])
        Lxx = np.zeros([self.state.ndx, self.state.ndx])
        Luu = np.zeros([self.nu, self.nu])
        Lxu = np.zeros([self.state.ndx, self.nu])
        # COST DERIVATIVES 
        if self.isTerminal:
            Lx[:] , Lxx[:,:] = self.terminal_cost_derivatives(x) 
        else:
            Lx[:], Lxx[:,:] =  self.running_cost_state_derivative(x,u)
            Lu[:] =  self.w2.dot(u)
            Luu[:,:] = self.w2.copy()

            # dynamics derivatives 
            Fx[:,:], Fu[:,:] = self.dynamics.derivatives(x,u)
        # COPY TO DATA 
        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


class TwoLinkManipulatorViz:
    def __init__(self):
        pass


if __name__ =="__main__":
    model = TwoLinkManipulator() 
    print(" Checking Forward Kinematics ".center(LINE_WIDTH,"="))
    