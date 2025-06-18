import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

with open('data_dict.pickle', 'rb') as f:
    loaded_dict = pickle.load(f)

dt = 0.02
stop_state_log = 1000

if __name__ == "__main__":
    base_local = np.array([0.04576, 0.00014, -0.16398])

    abad_L_base_local = np.array([55.56e-3, 105e-3, -260.2e-3])
    hip_L_abad_local = np.array([-0.077, 0.0205, 0])
    knee_L_hip_local = np.array([-0.1500, -0.02050, -0.25981])
    foot_L_knee_local = np.array([0.150, 0, -0.25981])

    abad_R_base_local = np.array([55.56e-3, -105e-3, -260.2e-3])
    hip_R_abad_local = np.array([-0.077, -0.0205, 0])
    knee_R_hip_local = np.array([-0.1500, 0.02050, -0.25981])
    foot_R_knee_local = np.array([0.150, 0, -0.25981])

    plt.figure(figsize=(20, 4))
    for i in range(stop_state_log):
        plt.clf()
        base_pos= np.array([loaded_dict['base_pos_x'][i], loaded_dict['base_pos_y'][i], loaded_dict['base_pos_z'][i]])
        base_quat = np.array([loaded_dict['base_quat_0'][i], loaded_dict['base_quat_1'][i], loaded_dict['base_quat_2'][i], loaded_dict['base_quat_3'][i]])


        # base
        plt.plot(base_pos[0], base_pos[2], 'go')

        ###############################################################################################################
        # abad_L_Joint
        R_base = Rotation.from_quat(base_quat).as_matrix()
        abad_left_joint_global = R_base @ abad_L_base_local + base_pos

        plt.plot([base_pos[0], abad_left_joint_global[0]],
                 [base_pos[2], abad_left_joint_global[2]], 'b')

        # hip_L_left
        theta_abad = loaded_dict['dof_pos_0'][i]
        R_abad = np.array([
            [1, 0, 0],
            [0, np.cos(theta_abad), -np.sin(theta_abad)],
            [0, np.sin(theta_abad), np.cos(theta_abad)]
        ])
        hip_left_global = R_abad @ hip_L_abad_local + abad_left_joint_global

        plt.plot([abad_left_joint_global[0], hip_left_global[0]],
                 [abad_left_joint_global[2], hip_left_global[2]], 'b-')

        # knee_L_left
        theta_hip = loaded_dict['dof_pos_1'][i]
        R_hip = np.array([
            [np.cos(theta_hip), 0, np.sin(theta_hip)],
            [0, 1, 0],
            [-np.sin(theta_hip), 0, np.cos(theta_hip)]
        ])
        knee_left_global = R_hip @ knee_L_hip_local + hip_left_global

        plt.plot([hip_left_global[0], knee_left_global[0]],
                 [hip_left_global[2], knee_left_global[2]], 'b--')

        # foot_L_left
        theta_knee = loaded_dict['dof_pos_2'][i]
        R_knee = np.array([
            [np.cos(theta_knee), 0, -np.sin(theta_knee)],
            [0, 1, 0],
            [np.sin(theta_knee), 0, np.cos(theta_knee)]
        ])
        foot_left_global = R_knee @ foot_L_knee_local + knee_left_global

        plt.plot([knee_left_global[0], foot_left_global[0]],
                 [knee_left_global[2], foot_left_global[2]], 'b--')

        # real left foot
        left_foot = np.array([loaded_dict['foot_pos_1_x'][i],
                             loaded_dict['foot_pos_1_y'][i],
                             loaded_dict['foot_pos_1_z'][i]])
        plt.scatter(left_foot[0], left_foot[2], c='b')

        ###############################################################################################################
        # abad_R_Joint
        R_base = Rotation.from_quat(base_quat).as_matrix()
        abad_right_joint_global = R_base @ abad_R_base_local + base_pos

        plt.plot([base_pos[0], abad_right_joint_global[0]],
                 [base_pos[2], abad_right_joint_global[2]], 'r')

        # hip_R_right
        theta_abad = -loaded_dict['dof_pos_3'][i]
        R_abad = np.array([
            [1, 0, 0],
            [0, np.cos(theta_abad), -np.sin(theta_abad)],
            [0, np.sin(theta_abad), np.cos(theta_abad)]
        ])
        hip_right_global = R_abad @ hip_R_abad_local + abad_right_joint_global

        plt.plot([abad_right_joint_global[0], hip_right_global[0]],
                 [abad_right_joint_global[2], hip_right_global[2]], 'r-')

        # knee_R_right
        theta_hip = -loaded_dict['dof_pos_4'][i]
        R_hip = np.array([
            [np.cos(theta_hip), 0, np.sin(theta_hip)],
            [0, 1, 0],
            [-np.sin(theta_hip), 0, np.cos(theta_hip)]
        ])
        knee_right_global = R_hip @ knee_R_hip_local + hip_right_global

        plt.plot([hip_right_global[0], knee_right_global[0]],
                 [hip_right_global[2], knee_right_global[2]], 'r--')

        # foot_R_right
        theta_knee = -loaded_dict['dof_pos_5'][i]
        R_knee = np.array([
            [np.cos(theta_knee), 0, -np.sin(theta_knee)],
            [0, 1, 0],
            [np.sin(theta_knee), 0, np.cos(theta_knee)]
        ])
        foot_right_global = R_knee @ foot_R_knee_local + knee_right_global

        plt.plot([knee_right_global[0], foot_right_global[0]],
                 [knee_right_global[2], foot_right_global[2]], 'r--')

        # real right foot
        right_foot = np.array([loaded_dict['foot_pos_2_x'][i],
                               loaded_dict['foot_pos_2_y'][i],
                               loaded_dict['foot_pos_2_z'][i]])
        plt.scatter(right_foot[0], right_foot[2], c='r')

        ###############################################################################################################

        plt.plot([-0.5, 10], [0, 0], 'k--')


        plt.xlim([-0.5, 10])
        plt.ylim([-0.1, 1])

        plt.tight_layout()
        plt.show()
        plt.pause(dt)