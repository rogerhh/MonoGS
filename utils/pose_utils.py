import numpy as np
import torch
import lietorch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inverse(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
    T_inv[:3, :3] = R.t()
    T_inv[:3, 3] = -R.t() @ t
    return T_inv

def inverse_t(T):
    return -T[:3, :3].t() @ T[:3, 3]


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta,
                     camera.cam_rot_delta], axis=0)
    T_w2c = camera.T
    new_w2c = lietorch.SE3.exp(tau).matrix() @ T_w2c
    converged = (tau**2).sum() < (converged_threshold**2)
    camera.T = new_w2c
    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    
    return converged

def relative_pose_error(P1_gt, P2_gt, P1, P2):
    dP_gt = P1_gt.inverse() @ P2_gt
    dP = P1.inverse() @ P2

    return pose_diff(dP_gt, dP)


def pose_diff(P1, P2):
    T1, T2 = P1[:3, 3], P2[:3, 3]
    trans_diff = torch.norm(T1 - T2)

    R1, R2 = P1[:3, :3], P2[:3, :3]
    dR = R1 @ R2.transpose(0, 1)

    tr = dR.trace()
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta)

    return trans_diff, angle
    

def trans_diff(T1, T2):
    return torch.norm(T1[:3, 3] - T2[:3, 3])

def angle_diff(R1, R2):
    R = R1.float() @ R2.float().transpose(0, 1)
    tr = R.trace()
    tr = torch.clamp(tr, -1.0, 3.0)
    angle = torch.acos((tr - 1.0) / 2.0)
    return angle
