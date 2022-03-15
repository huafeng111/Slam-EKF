import numpy as np
from pr3_utils import *
import sys

def LM(timesteps,fea_num,params):
    eye = np.eye(3)
    Lm= {}
    # set up the covariance and mean for each part
    Lm['covariance2'] = np.ones((3, 3, fea_num))
    Lm['trajectory2'] = np.ones((4, fea_num, timesteps))
    Lm['mean2'] = np.zeros((4, fea_num))
    Lm['mean2'].fill(None)
    for i in range(fea_num):
        Lm['covariance2'][:, :, i] =eye*params
    Lm['mean'] = np.zeros((4, fea_num))
    Lm['mean'].fill(None)
    Lm['trajectory'] = np.zeros((4, fea_num, timesteps))
    Lm['covariance'] = np.zeros((3, 3, fea_num))
    p_val=eye * params
    for i in range(fea_num):
        Lm['covariance'][:, :, i] = p_val
    return Lm
def matrix_K2(value):
    return np.block([[matrix_K(value[3:, np.newaxis]),
          -value[:3, np.newaxis]],[np.zeros((1, 4))]])
def Vehicle(params ,step):
    vehicle = {}
    eye,eye6=np.eye(4),np.eye(6)
    vehicle['mean2'] = eye
    vehicle['mean'] =eye
    vehicle['covariance2'] = eye6 * params
    vehicle['covariance'] = eye6 * params
    # trajectory route record
    vehicle['trajectory2'] = np.zeros((4, 4, step))
    vehicle['trajectory'] = np.zeros((4, 4, step))
    return vehicle

def matrix_K(value):
    return np.array([[0, -value[2], value[1]], [value[2], 0, -value[0]],
                         [-value[1], value[0], 0]])
def matrix_G(i):
    return (np.array([[1, 0, -i[0] / i[2], 0], [0, 1, -i[1] / i[2], 0],
                            [0, 0, 0, 1],[0, 0, -i[3] / i[2], 1]]))/i[2]

def EKF_update1(k_value,Matrix,curr_LM,eye2,eye1,f):
    mid_cov=k_value.T @ np.linalg.inv(k_value \
    @LM['covariance'][:, :, i]@ k_value.T + eye1)
    val_ = LM['covariance'][:, :, i] @ mid_cov
    div_c=(curr_LM / curr_LM[2])
    data = Matrix @ div_c
    back_=val_ @ (f - data)
    LM['mean'][:, i] = LM['mean'][:, i]+eye2.T @ back_
    front_val = (np.eye(3) - val_ @ k_value)
    cov_val = LM['covariance'][:, :, i]
    LM['covariance'][:, :, i] = front_val @ cov_val
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
def EKF_visual_predict(coe, params,vehicle, params_me, k, me, ):
    eye = np.eye(3)
    comb_hat = np.vstack((np.hstack((matrix_K(me), k.reshape(3, 1))), np.zeros((1, 4))))
    create_mk=matrix_K(me)
    create_vk = matrix_K(k)
    row_zero=np.zeros((3, 3))
    blockhat = np.block([[ create_mk,  create_vk],
                           [row_zero,  create_mk]])
    vehicle['mean2'] = expm(coe * comb_hat) @ vehicle['mean2']
    back_f=vehicle['covariance2'] @ np.transpose(expm(coe * blockhat))
    pre_f=expm(coe * blockhat)
    zero_arr = np.zeros((3, 3))
    # set up the movement noise
    nois_ = np.block([[eye * params, zero_arr], [zero_arr, eye * params_me]])
    vehicle['covariance2'] = nois_+pre_f @ back_f

def EKFupdate(k, b, imu_T_cam, params,vehicle,LM, features):
    # covariance for measurement noise
    eye2 = np.eye(3, 4)
    time = features.shape[1]
    # define the stero camera matrix
    Matrix =  np.array([[k[0, 0], 0, k[0, 2], 0], # C is the calibration matrix
                  [0, k[1, 1], k[1, 2], 0], [k[0, 0], 0, k[0, 2], -k[0, 1] * b],# baseline
                  [0, k[1, 1], k[1, 2], 0]])
    eye1 = params * np.eye(4)
    for i in range(time):
        f = features[:, i][:]
        if np.all(f !=-1):
            boolean=np.isnan(LM['mean'][:, i])
            v_fr = W_to_T(vehicle['mean'])
            l_p = (b * K[0, 0]) / (f[0] - f[2])
            coords = np.hstack((l_p * np.linalg.inv(K) @ np.hstack((f[:2], 1)), 1))
            if (np.all(boolean)):
                i_tc=np.linalg.inv(imu_T_cam)@ coords
                init=v_fr @ i_tc
                LM['mean'][:, i] =init
                continue
            ctw = imu_T_cam @ vehicle['mean']
            media=ctw @ eye2.T
            curr_LM = ctw @ LM['mean'][:, i]
            front=Matrix @ matrix_G(curr_LM)
            k_value =  front @ media
            # EKF update process
            EKF_update1(k_value,Matrix,curr_LM,eye2,eye1,f)

def Visual_EKF_update2(Matrix,Lm,front,eye2,f,remove_,eye1):

    val_ = Lm['covariance2'][:, :, i] @ front.T @ np.linalg.inv(
        front @ Lm['covariance2'][:, :, i] @ front.T + eye1)
    b=eye2.T @ val_ @ (f - remove_)
    eye3=np.eye(3)
    zero_blo=np.zeros((1, 6))
    front_val= np.abs(np.eye(3) - val_ @ front)
    Lm['covariance2'][:, :, i] =  front_val @ Lm['covariance2'][:, :, i]
    Lm['mean2'][:, i] = b+Lm['mean2'][:, i]
    LM_t = vehicle['mean2'] @ Lm['mean2'][:, i]
    sub_=-matrix_K(LM_t[:3])
    out_m=matrix_G(imu_T_cam @ LM_t)
    block_matr=np.block([[eye3, sub_],[zero_blo]])
    comb=imu_T_cam @ block_matr
    matrix_k = Matrix @ out_m @ comb
    return matrix_k

def EKF_predict(coe, params, vehicle,params_me, k, me):
    eye = np.eye(3)
    comb_hat = np.vstack((np.hstack((matrix_K(me), k.reshape(3, 1))), np.zeros((1, 4))))
    create_mk = matrix_K(me)
    create_vk = matrix_K(k)
    row_zero = np.zeros((3, 3))
    blockhat = np.block([[create_mk, create_vk],
                         [row_zero, create_mk]])
    vehicle['mean'] = expm(coe * comb_hat) @ vehicle['mean']
    back_f = vehicle['covariance'] @ np.transpose(expm(coe * blockhat))
    pre_f = expm(coe * blockhat)
    zero_arr = np.zeros((3, 3))
    # set up the movement noise
    nois_ = np.block([[eye * params, zero_arr], [zero_arr, eye * params_me]])
    vehicle['covariance'] = nois_ + pre_f @ back_f
# inverse pose  to world frame

def W_to_T(mean):
    shape_ = mean[:3, 3].reshape(3, 1)
    mean_T = np.transpose(mean[:3, :3])
    dot_m_s=-np.dot(mean_T, shape_)
    h_space=np.hstack((mean_T, dot_m_s))
    stack = np.vstack((h_space, np.array([0, 0, 0, 1])))
    return stack

def Visual_EKF_update(Matrix,Lm,front,eye2,f,remove_,eye1):
    back_val = Lm['covariance2'][:, :, i] @ front.T
    back_f=f - remove_
    eye3=np.eye(3)
    zero_=np.zeros((1, 6))
    front_=Lm['covariance2'][:, :, i]
    b = Lm['mean2'][:, i]
    val_ = front_ @ front.T @ np.linalg.inv(eye1+front @ back_val)
    res_=(np.eye(3) - val_  @ front)
    Lm['covariance2'][:, :, i] = res_ @ Lm['covariance2'][:, :, i]
    Lm['mean2'][:, i] = eye2.T @ val_ @ back_f+b
    LM_t = vehicle['mean2'] @ Lm['mean2'][:, i]
    sub_=-matrix_K(LM_t[:3])
    block_matr=np.block([[eye3,  sub_], [zero_]])
    front_=Matrix @ matrix_G(imu_T_cam @ LM_t)
    matrix_k = front_ @ imu_T_cam @ block_matr
    return matrix_k

def Visual_Inertial_SLAM(Lm,  imu_T_cam,features, K,params, b,vehicle):
    eye1,eye2  = params * np.eye(4),np.eye(3, 4),
    # Define the stero camera matrix
    Matrix = np.array([[K[0, 0], 0, K[0, 2], 0],  # C is the calibration matrix
                  [0, K[1, 1], K[1, 2], 0], [K[0, 0], 0, K[0, 2], -K[0, 0] * b],  # baseline
                  [0,K[1, 1], K[1, 2], 0]])
    time=features.shape[1]
    for i in range(time):
        f= features[:, i][:]
        if np.all(f!= -1): # process current timestep
            # current landmark is in the camera frame.
            # if landmark not present in the previous frame need to initialize the landmark
            boolean_sy=np.isnan(Lm['mean2'][:, i])
            v_fr = W_to_T(vehicle['mean2'])
            l_p = (b * K[0, 0]) / (f[0] - f[2])
            coords = np.hstack((l_p * np.linalg.inv(K) @ np.hstack((f[:2], 1)), 1))
            if (np.all(boolean_sy)):
                i_tc = np.linalg.inv(imu_T_cam) @ coords
                comb_=v_fr @ i_tc
                LM['mean2'][:, i] = comb_
                continue
            # if landmark is already in current time stamp
            ctw = imu_T_cam @ vehicle['mean2']
            curr_LM = ctw @ LM['mean2'][:, i]
            remove_ = Matrix @ ( curr_LM / curr_LM [2])  # remove depth information and map to pixels
            media = ctw @ eye2.T
            front = Matrix  @ matrix_G( curr_LM ) @ media
            #Visual EKF update
            matrix_k=Visual_EKF_update(Matrix, Lm, front, eye2, f, remove_, eye1)
            # Inertial EKF update
            cov1=vehicle['covariance2'] @ matrix_k.T
            cov2=np.linalg.inv(matrix_k @ vehicle['covariance2'] @ matrix_k.T + eye1)
            val_ =  cov1 @ cov2
            val_di=val_ @ (f- remove_)
            vehicle['mean2'] = expm(matrix_K2(val_di)) @ vehicle['mean2']
            vehicle['covariance2'] = (np.eye(6) - val_ @ matrix_k) @ vehicle['covariance2']

if __name__ == '__main__':
    filename = "./data/03.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    LM = LM(t.shape[1], features.shape[1], 0.02)
    vehicle = Vehicle(0.02,t.shape[1])
    for i in range(1,t.shape[1]):
        print("The",i,"loop")
        d = np.abs(t[0, i - 1]-t[0, i] )
    	# (a) IMU Localization via EKF Prediction
        d=-d
        EKF_predict( d, 0.00003, vehicle,0.0003, linear_velocity[:, i], angular_velocity[:, i])
        EKF_visual_predict(d, 0.00003, vehicle,0.0003, linear_velocity[:, i], angular_velocity[:, i])
        # set up different value for each way
        mean_=vehicle['mean']
        vehicle['trajectory'][:, :, i] = W_to_T(mean_)
        LM['trajectory'][:, :, i - 1] = LM['mean'][:]
        mean_2 = vehicle['mean']
        vehicle['trajectory2'][:, :, i] = W_to_T(mean_2)
        LM['trajectory2'][:, :, i - 1] = LM['mean2'][:]
    	# (b) Landmark Mapping via EKF Update
        EKFupdate(K, b,imu_T_cam, 4000,vehicle, LM, features[:, :, i])
    	# (c) Visual-Inertial SLAM
        Visual_Inertial_SLAM(LM,  imu_T_cam,features[:, :, i], K,2800, b,vehicle)

    # You can use the function below to visualize the robot pose over time
    visualize_trajectory_2d(vehicle['trajectory'],LM['mean'], path_name = filename[7:-4], show_ori = True)
    sys.exit(0)
