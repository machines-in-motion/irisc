import numpy as np 
import matplotlib.pyplot as plt 


from robot_properties_solo.solo12wrapper import Solo12Config

if __name__ == "__main__": 
    robot_config = Solo12Config() 
    controlled_joints = []
    for leg in ["FL", "FR", "HL", "HR"]:
        controlled_joints += [leg + "_HAA", leg + "_HFE", leg + "_KFE"]
    q_names = ['x', 'y', 'z','r','p','y'] + controlled_joints
    v_names = ['vx', 'vy', 'vz','wx','wy','wz'] + ['v_'+n for n in controlled_joints]
    state_names = q_names + v_names

    dt = 1.e-3 
    jnt_limits = [ 0.9, 1.45,  2.8,  0.9, 1.45,  2.8,  0.9, 1.45,  2.8,  0.9, 1.45,  2.8]

    reference = np.load("trot_plan.npz")
    # momentum_plan = reference["mom_opt"]
    # forces_plan = reference["F_opt"]
    # ik_com_plan = reference["ik_com_opt"]
    # ik_momentum_plan = reference["ik_mom_opt"]
    # xs_plan = reference["xs"]
    # us_plan = reference["us"]
    # contact_plan = reference["cnt_plan"]
    q_plan = reference["q"]
    f_plan = reference["f"]
    print(q_plan.shape)
    print(f_plan.shape)

    # for i in reference.keys():
    #     if i == "cnt_plan" or i == "F_opt":
    #         continue
    #     plt.figure(i)
    #     for k in range(reference[i].shape[1]):
    #         plt.plot(np.arange(reference[i].shape[0]), reference[i][:,k])
    
    # plt.show()
    horizon = 2500 
    id_time = dt * np.arange(0,horizon)
    plt.figure("Joint Positions",figsize=(20,15))
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.plot(id_time, horizon*[jnt_limits[i]], 'k--')
        plt.plot(id_time, horizon*[-jnt_limits[i]], 'k--')
        plt.plot(id_time, q_plan[:,7+i], label="des_"+state_names[6+i])
        plt.legend()
    plt.show()