import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from copy import deepcopy, copy
import math
import os
import json
import cv2
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import yaml
from munch import munchify

from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera
from gaussian_splatting.gaussian_renderer import render
from utils.config_utils import load_config
from utils.pose_utils import update_pose, trans_diff, angle_diff, pose_diff, relative_pose_error
from utils.slam_utils import get_loss_tracking, get_loss_tracking_per_pixel, get_median_depth, HuberLoss, huber_loss
from processing.utils import load_data
from utils.configs import cuda_device

class TempCamera:
    def __init__(self, viewpoint):
        # Copy the viewpoint's relevant data
        self.T = viewpoint.T.detach().clone()
        self.cam_rot_delta = viewpoint.cam_rot_delta.detach().clone()
        self.cam_trans_delta = viewpoint.cam_trans_delta.detach().clone()
        self.exposure_a = viewpoint.exposure_a.detach().clone()
        self.exposure_b = viewpoint.exposure_b.detach().clone()

    def assign(self, viewpoint):
        viewpoint.T = self.T.detach().clone()
        viewpoint.cam_rot_delta.data.copy_(self.cam_rot_delta)
        viewpoint.cam_trans_delta.data.copy_(self.cam_trans_delta)
        viewpoint.exposure_a.data.copy_(self.exposure_a)
        viewpoint.exposure_b.data.copy_(self.exposure_b)

    def step(self, x):
        self.cam_trans_delta.data += x[:3]
        self.cam_rot_delta.data += x[3:6]
        self.exposure_a.data += x[6]
        self.exposure_b.data += x[7]

def save_torch_image(image, path):
    image_np = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image_np)

def gen_default_forward_sketch_args():
    forward_sketch_args = {"sketch_mode": 0, 
                           "repeat_dim": 0,
                           "stack_dim": 0,
                           "sketch_dim": 0, 
                           "sketch_indices": None, 
                           "rand_indices": None, 
                           "sketch_dtau": None, 
                           "sketch_dexposure": None, 
                           "chunk_size": None, 
                           "rand_weights": None, 
                           "rand_indices_row": None, 
                           "rand_indices_col": None, }
    return forward_sketch_args

def gen_forward_sketch_args(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim):

    n = tau_len + exposure_len
    m = height * width
    d = stack_dim * sketch_dim
    chunk_size = m // d

    # These are static tensors so can initialize them here
    sketch_index_offsets = (torch.arange(stack_dim, dtype=torch.int32, device=cuda_device) * (m)).repeat_interleave(chunk_size*d // stack_dim)
    sketch_index_vals = torch.arange(sketch_dim, dtype=torch.int32, device=cuda_device).repeat_interleave(chunk_size).repeat(stack_dim)
    sketch_index_indices = torch.arange(chunk_size*d, dtype=torch.int32, device=cuda_device)

    rand_flat_indices = torch.empty((repeat_dim, chunk_size*d), dtype=torch.int32, device=cuda_device)
    
    sketch_indices = torch.ones((repeat_dim, stack_dim*height*width), dtype=torch.int32, device=cuda_device) * (-1)
    rand_indices_row = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=cuda_device)
    rand_indices_col = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=cuda_device)
    
    streams = [torch.cuda.Stream(device=cuda_device) for _ in range(repeat_dim)]
    
    for i in range(repeat_dim):
        with torch.cuda.stream(streams[i]):
            rand_flat_indices[i] = torch.randperm(m, dtype=torch.int32, device=cuda_device)[:(chunk_size*d)]
            sketch_indices[i, rand_flat_indices[i, sketch_index_indices] + sketch_index_offsets] = sketch_index_vals
            rand_indices_row_i = rand_flat_indices[i] // width
            rand_indices_col_i = rand_flat_indices[i] % width
            
            rand_indices_row[i] = rand_indices_row_i.view(stack_dim, sketch_dim, chunk_size)
            rand_indices_col[i] = rand_indices_col_i.view(stack_dim, sketch_dim, chunk_size)
     
    torch.cuda.synchronize()
    sketch_indices = sketch_indices.reshape((repeat_dim, stack_dim, height, width))
    rand_indices = (rand_indices_row, rand_indices_col)
    

    rand_weights = torch.randint(0, 2, size=(repeat_dim, height, width), dtype=torch.float32, device=cuda_device) * 2 - 1

    sketch_dtau = torch.empty((stack_dim, sketch_dim, tau_len), dtype=torch.float32, device=cuda_device, requires_grad=True)
    sketch_dexposure = torch.empty((stack_dim, sketch_dim, exposure_len), dtype=torch.float32, device=cuda_device, requires_grad=True)

    forward_sketch_args = {"sketch_mode": 1,
                           "repeat_dim": repeat_dim,
                           "stack_dim": stack_dim,
                           "sketch_dim": sketch_dim,
                           "sketch_indices": sketch_indices,
                           "rand_indices": rand_indices,
                           "sketch_dtau": sketch_dtau,
                           "sketch_dexposure": sketch_dexposure,
                           "chunk_size": chunk_size,
                           "rand_weights": rand_weights,
                           "rand_indices_row": rand_indices_row,
                           "rand_indices_col": rand_indices_col, }
    return forward_sketch_args


def gen_forward_sketch_args_kaczmarz(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, p):

    n = tau_len + exposure_len
    m = height * width
    d = stack_dim * sketch_dim
    # For randomized Kaczmarz, each sketched row corresponds to one pixel
    chunk_size = 1

    # These are static tensors so can initialize them here
    sketch_index_offsets = (torch.arange(stack_dim, dtype=torch.int32, device=cuda_device) * (m)).repeat_interleave(chunk_size*d // stack_dim)
    sketch_index_vals = torch.arange(sketch_dim, dtype=torch.int32, device=cuda_device).repeat_interleave(chunk_size).repeat(stack_dim)
    sketch_index_indices = torch.arange(chunk_size*d, dtype=torch.int32, device=cuda_device)

    
    rand_flat_indices = torch.empty((repeat_dim, chunk_size*d), dtype=torch.int32, device=cuda_device)
    sketch_indices = torch.ones((repeat_dim, stack_dim*height*width), dtype=torch.int32, device=cuda_device) * (-1)
    rand_indices_row = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=cuda_device)
    rand_indices_col = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=cuda_device)
    
    streams = [torch.cuda.Stream(device=cuda_device) for _ in range(repeat_dim)]
    
    for i in range(repeat_dim):
        with torch.cuda.stream(streams[i]):
            # Pick d random indices from the range [0, m) using distribution p
            rand_flat_indices[i] = torch.multinomial(p, num_samples=d, replacement=False)
            sketch_indices[i, rand_flat_indices[i, sketch_index_indices] + sketch_index_offsets] = sketch_index_vals
            rand_indices_row_i = rand_flat_indices[i] // width
            rand_indices_col_i = rand_flat_indices[i] % width
            
            rand_indices_row[i] = rand_indices_row_i.view(stack_dim, sketch_dim, chunk_size)
            rand_indices_col[i] = rand_indices_col_i.view(stack_dim, sketch_dim, chunk_size)
     
    torch.cuda.synchronize()
    sketch_indices = sketch_indices.reshape((repeat_dim, stack_dim, height, width))
    rand_indices = (rand_indices_row, rand_indices_col)

    rand_weights = torch.ones((repeat_dim, height, width), dtype=torch.float32, device=cuda_device)
    sketch_dtau = torch.empty((stack_dim, sketch_dim, tau_len), dtype=torch.float32, device=cuda_device, requires_grad=True)
    sketch_dexposure = torch.empty((stack_dim, sketch_dim, exposure_len), dtype=torch.float32, device=cuda_device, requires_grad=True)

    forward_sketch_args = {"sketch_mode": 1,
                           "repeat_dim": repeat_dim,
                           "stack_dim": stack_dim,
                           "sketch_dim": sketch_dim,
                           "sketch_indices": sketch_indices,
                           "rand_indices": rand_indices,
                           "sketch_dtau": sketch_dtau,
                           "sketch_dexposure": sketch_dexposure,
                           "chunk_size": chunk_size,
                           "rand_weights": rand_weights,
                           "rand_indices_row": rand_indices_row,
                           "rand_indices_col": rand_indices_col, }
    return forward_sketch_args

def get_sketched_system(args):
    config, height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, viewpoint, gaussians, pipeline_params, background = args

    forward_sketch_args = gen_forward_sketch_args(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim)
    render_pkg = render(
        viewpoint, gaussians, pipeline_params, background, forward_sketch_args=forward_sketch_args,
    )
    image, depth, opacity = (
        render_pkg["render"],
        render_pkg["depth"],
        render_pkg["opacity"],
    )

    loss_tracking_img = get_loss_tracking_per_pixel(
        config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
    )
    loss_tracking_img = HuberLoss.apply(loss_tracking_img, 0.1)

    loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
    l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
    # trans_error, angle_error = relative_pose_error(prev.T_gt, viewpoint.T_gt, prev.T, viewpoint.T)

    sketch_dim_per_iter = stack_dim * sketch_dim
    print("sketch_dim_per_iter = ", sketch_dim_per_iter)
    loss_tracking_img = loss_tracking_img.sum(dim=0) / (m / sketch_dim_per_iter)

    rand_weights = forward_sketch_args["rand_weights"]
    rand_indices_row = forward_sketch_args["rand_indices_row"]
    rand_indices_col = forward_sketch_args["rand_indices_col"]

    weighted_loss_tracking_img = loss_tracking_img * rand_weights
    SJ = torch.empty((repeat_dim, stack_dim, sketch_dim, n), device=cuda_device)
    batch_indices = torch.arange(repeat_dim).view(-1, 1, 1, 1)
    Sf = weighted_loss_tracking_img[batch_indices, rand_indices_row, rand_indices_col].sum(dim=-1)

    for i in range(repeat_dim):

        viewpoint.cam_rot_delta.grad = None
        viewpoint.cam_trans_delta.grad = None
        viewpoint.exposure_a.grad = None
        viewpoint.exposure_b.grad = None
        forward_sketch_args["sketch_dtau"].grad = None
        forward_sketch_args["sketch_dexposure"].grad = None

        weighted_loss_tracking_img[i].backward(gradient=torch.ones_like(weighted_loss_tracking_img[i]), retain_graph=True)
        SJ_i = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=2)
        SJ[i] = SJ_i

    with torch.no_grad():
        SJ = SJ.reshape((-1, n)) / math.sqrt(repeat_dim)
        Sf = Sf.flatten() / math.sqrt(repeat_dim)

    return SJ, Sf

def gen_forward_sketch_args_sequential(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, start_idx):

    n = tau_len + exposure_len
    m = height * width
    d = stack_dim * sketch_dim
    # For randomized Kaczmarz, each sketched row corresponds to one pixel
    chunk_size = 1

    # These are static tensors so can initialize them here
    sketch_index_offsets = (torch.arange(stack_dim, dtype=torch.int32, device=cuda_device) * (m)).repeat_interleave(chunk_size*d // stack_dim)
    sketch_index_vals = torch.arange(sketch_dim, dtype=torch.int32, device=cuda_device).repeat_interleave(chunk_size).repeat(stack_dim)
    sketch_index_indices = torch.arange(chunk_size*d, dtype=torch.int32, device=cuda_device)

    torch.cuda.synchronize()
    
    rand_flat_indices = torch.empty((repeat_dim, chunk_size*d), dtype=torch.int32, device=cuda_device)
    sketch_indices = torch.ones((repeat_dim, stack_dim*height*width), dtype=torch.int32, device=cuda_device) * (-1)
    rand_indices_row = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=cuda_device)
    rand_indices_col = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=cuda_device)
    
    streams = [torch.cuda.Stream(device=cuda_device) for _ in range(repeat_dim)]
    
    for i in range(repeat_dim):
        with torch.cuda.stream(streams[i]):
            # Pick d sequential indices from the range [0, m) starting from start_idx
            # Modulo m to wrap around in case start_idx + d exceeds m
            rand_flat_indices[i] = torch.arange(start_idx, start_idx + d, dtype=torch.int32, device=cuda_device) % m
            torch.cuda.synchronize()
            sketch_indices[i, rand_flat_indices[i, sketch_index_indices] + sketch_index_offsets] = sketch_index_vals
            rand_indices_row_i = rand_flat_indices[i] // width
            rand_indices_col_i = rand_flat_indices[i] % width
            torch.cuda.synchronize()
            
            rand_indices_row[i] = rand_indices_row_i.view(stack_dim, sketch_dim, chunk_size)
            rand_indices_col[i] = rand_indices_col_i.view(stack_dim, sketch_dim, chunk_size)
            torch.cuda.synchronize()
     
    torch.cuda.synchronize()
    sketch_indices = sketch_indices.reshape((repeat_dim, stack_dim, height, width))
    rand_indices = (rand_indices_row, rand_indices_col)

    rand_weights = torch.ones((repeat_dim, height, width), dtype=torch.float32, device=cuda_device)
    sketch_dtau = torch.empty((stack_dim, sketch_dim, tau_len), dtype=torch.float32, device=cuda_device, requires_grad=True)
    sketch_dexposure = torch.empty((stack_dim, sketch_dim, exposure_len), dtype=torch.float32, device=cuda_device, requires_grad=True)
    torch.cuda.synchronize()

    forward_sketch_args = {"sketch_mode": 1,
                           "repeat_dim": repeat_dim,
                           "stack_dim": stack_dim,
                           "sketch_dim": sketch_dim,
                           "sketch_indices": sketch_indices,
                           "rand_indices": rand_indices,
                           "sketch_dtau": sketch_dtau,
                           "sketch_dexposure": sketch_dexposure,
                           "chunk_size": chunk_size,
                           "rand_weights": rand_weights,
                           "rand_indices_row": rand_indices_row,
                           "rand_indices_col": rand_indices_col, }
    return forward_sketch_args

def get_full_system(args):
    config, height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, viewpoint, gaussians, pipeline_params, background = args

    n = tau_len + exposure_len
    m = height * width
    d = stack_dim * sketch_dim

    sketch_dim_per_iter = stack_dim * sketch_dim
    J = torch.empty((m, n), device=cuda_device)
    f = torch.empty((m), device=cuda_device)
    for start_row in range(0, m, sketch_dim_per_iter):
        torch.cuda.synchronize()

        forward_sketch_args = gen_forward_sketch_args_sequential(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, start_row)
        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, forward_sketch_args=forward_sketch_args,
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )

        loss_tracking_img = get_loss_tracking_per_pixel(
            config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
        )
        loss_tracking_img = HuberLoss.apply(loss_tracking_img, 0.1)

        loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
        l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
        # trans_error, angle_error = relative_pose_error(prev.T_gt, viewpoint.T_gt, prev.T, viewpoint.T)

        sketch_dim_per_iter = stack_dim * sketch_dim
        loss_tracking_img = loss_tracking_img.sum(dim=0) / (m / sketch_dim_per_iter)

        rand_weights = forward_sketch_args["rand_weights"]
        rand_indices_row = forward_sketch_args["rand_indices_row"]
        rand_indices_col = forward_sketch_args["rand_indices_col"]

        weighted_loss_tracking_img = loss_tracking_img * rand_weights
        SJ = torch.empty((repeat_dim, stack_dim, sketch_dim, n), device=cuda_device)
        batch_indices = torch.arange(repeat_dim).view(-1, 1, 1, 1)
        Sf = weighted_loss_tracking_img[batch_indices, rand_indices_row, rand_indices_col].sum(dim=-1)

        for i in range(repeat_dim):

            viewpoint.cam_rot_delta.grad = None
            viewpoint.cam_trans_delta.grad = None
            viewpoint.exposure_a.grad = None
            viewpoint.exposure_b.grad = None
            forward_sketch_args["sketch_dtau"].grad = None
            forward_sketch_args["sketch_dexposure"].grad = None

            weighted_loss_tracking_img[i].backward(gradient=torch.ones_like(weighted_loss_tracking_img[i]), retain_graph=True)
            SJ_i = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=2)
            SJ[i] = SJ_i

        with torch.no_grad():
            SJ = SJ.reshape((-1, n)) / math.sqrt(repeat_dim)
            Sf = Sf.flatten() / math.sqrt(repeat_dim)

        num_sketch_row = min(sketch_dim_per_iter, m - start_row)
        J[start_row:start_row + num_sketch_row] = SJ[0:num_sketch_row]
        f[start_row:start_row + num_sketch_row] = Sf[0:num_sketch_row]
        torch.cuda.synchronize()


    return J, f

def left_singular_vectors(A):
    U, s, V = torch.linalg.svd(A, full_matrices=False)
    return U

def row_norm_squared(M):
    return torch.linalg.norm(M, axis=1) ** 2


def sample_rows(A, b, num_samples, mode):
    if mode == "row-norm-squared":
        row_norms = row_norm_squared(A)
        p = row_norms / torch.sum(row_norms)
    elif mode == "uniform":
        n = A.shape[0]
        p = torch.ones(n, device=cuda_device) / n
    elif mode == "leverage-score":
        U = left_singular_vectors(A)
        row_norms = row_norm_squared(U)
        p = row_norms / torch.sum(row_norms)
    else:
        raise ValueError(f"Invalid mode {mode}")
    n = A.shape[0]
    indices = torch.multinomial(p, num_samples=num_samples, replacement=True)
    A_tilde = A[indices, :] / torch.sqrt(num_samples * p[indices, None])
    b_tilde = b[indices] / torch.sqrt(num_samples * p[indices])
    return A_tilde, b_tilde

def project(A, b, d, mode="gaussian", p=8):
    m, n = A.shape
    if mode == "gaussian":
        S = torch.randn((d, m), device=cuda_device)
        p = d
        print(f"p = {p}")
    elif mode == "sparse-count-sketch":
        indices = torch.stack([torch.randperm(d, device=cuda_device)[:p] for _ in range(m)], dim=1)
        weights = torch.randint(0, 2, size=(p, m), dtype=torch.float32, device=cuda_device) * 2 - 1
        S = torch.zeros((d, m), device=cuda_device)
        S[indices, torch.arange(m, device=cuda_device)] = weights
    elif mode == "very-sparse-count-sketch":
        # Generate a m-vector, each entry is [0, d)
        indices = torch.randint(0, d, size=(m,), device=cuda_device)
        weights = torch.randint(0, 2, size=(m,), dtype=torch.float32, device=cuda_device) * 2 - 1
        S = torch.zeros((d, m), device=cuda_device)
        S[indices, torch.arange(m, device=cuda_device)] = weights
        p = 1

    S = S / math.sqrt(float(p))


    SA = S @ A
    Sb = S @ b
    return SA, Sb




def get_sketched_problem(A, b, d, mode="sample", submode="row-norm-squared"):
    if mode == "sample":
        A_tilde, b_tilde = sample_rows(A, b, d, submode)
    elif mode == "project":
        A_tilde, b_tilde = project(A, b, d, submode)

    return A_tilde, b_tilde


def solve_sketched(A, b, d=256, x_true=None, sketch_mode1="sample", sketch_mode2="leverage-score", solve_mode="sketch-and-solve"):

    if solve_mode == "sketch-and-solve":
        SA, Sb = get_sketched_problem(A, b, d, sketch_mode1, sketch_mode2)
        x = torch.linalg.lstsq(SA, Sb, rcond=None)[0]
    elif solve_mode == "sketch-and-iterate":
        x = torch.zeros((A.shape[1]), device=cuda_device)
        r = b
        for _ in range(50):
            r = r - A @ x
            SA, Sr = get_sketched_problem(A, r, d, sketch_mode1, sketch_mode2)
            # u = torch.linalg.lstsq(SA, Sr, rcond=None)[0]
            H_tilde = SA.T @ SA
            u = torch.linalg.solve(H_tilde, A.T @ r)
            x = x + u

            loss = A @ x - b
            print(f"loss = {loss.norm().item()}")

    x_diff = x - x_true
    loss_true = A @ x_true - b
    loss_sketched = A @ x - b
    print(f"loss_true = {loss_true.norm().item()}, loss_sketched = {loss_sketched.norm().item()}, x_diff = {x_diff.norm().item()}")

    return x

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--frame", type=int)

    args = parser.parse_args(sys.argv[1:])

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    live_mode = config["Dataset"]["type"] == "realsense"
    monocular = config["Dataset"]["sensor_type"] == "monocular"
    use_spherical_harmonics = config["Training"]["spherical_harmonics"]
    use_gui = config["Results"]["use_gui"]
    config["Training"]["monocular"] = monocular

    gaussians = GaussianModel(0, config)
    gaussians.load_ply(f"experiment/data/frame{args.frame:06d}_gaussians.ply")

    load_dict = torch.load(f"experiment/data/frame{args.frame:06d}_params.pth")
    viewpoint = load_dict["viewpoint"]
    pipeline_params = load_dict["pipeline_params"]
    background = load_dict["background"]
    device = background.device

    opt_params = []
    opt_params.append(
        {
            "params": [viewpoint.cam_rot_delta],
            "lr": config["Training"]["lr"]["cam_rot_delta"] * 0.1,
            "name": "rot_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.cam_trans_delta],
            "lr": config["Training"]["lr"]["cam_trans_delta"] * 0.1,
            "name": "trans_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_a],
            "lr": config["Training"]["lr"]["exposure_a"] * 0.1,
            "name": "exposure_a_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_b],
            "lr": config["Training"]["lr"]["exposure_b"] * 0.1,
            "name": "exposure_b_{}".format(viewpoint.uid),
        }
    )
        
    orig_viewpoint = TempCamera(viewpoint)

    default_forward_sketch_args = gen_default_forward_sketch_args()

    num_iter = 100

    best_loss_scalar = 1e10

    if True:
        # Randomized Kaczmarz method
        repeat_dim = 1
        stack_dim = 1
        sketch_dim = 64

        initial_lambda = 0.000
        second_order_converged_threshold = config["Training"]["RGN"]["second_order"]["converged_threshold"]

        lambda_ = initial_lambda

        # Run Randomized LS optimization
        height, width = viewpoint.image_height, viewpoint.image_width
        tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
        exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
        m = height * width
        n = tau_len + exposure_len
        d = stack_dim * sketch_dim
        chunk_size = 1

        orig_viewpoint.assign(viewpoint)
        pose_optimizer = torch.optim.Adam(opt_params)

        p_vector = torch.ones(m, device=cuda_device) / m

        default_forward_sketch_args = gen_default_forward_sketch_args()

        outer_losses = []

        for tracking_itr in range(20):
            render_pkg = render(
                viewpoint, gaussians, pipeline_params, background, forward_sketch_args=default_forward_sketch_args,
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                config, image, depth, opacity, viewpoint, forward_sketch_args=default_forward_sketch_args,
            )
            loss_tracking_img = HuberLoss.apply(loss_tracking_img, 0.1)

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
            l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
            # trans_error, angle_error = relative_pose_error(prev.T_gt, viewpoint.T_gt, prev.T, viewpoint.T)
            print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, l2 loss = {l2_loss.item():.4f}")

            outer_losses.append(loss_tracking_scalar.item())

            x = torch.randn(n, device=cuda_device) * 0

            baseline_repeat_dim = 1
            baseline_stack_dim = 64
            baseline_sketch_dim = 64
            sketch_args = (config, height, width, tau_len, exposure_len, baseline_repeat_dim, baseline_stack_dim, baseline_sketch_dim, viewpoint, gaussians, pipeline_params, background)

            J, f = get_full_system(sketch_args)

            x_true = torch.linalg.pinv(J) @ (-f)

            d = 128
            sketch_mode1 = "sample"
            sketch_mode2 = "row-norm-squared"
            solve_mode = "sketch-and-solve"
            x = solve_sketched(J, -f, d=d, x_true=x_true, sketch_mode1=sketch_mode1, sketch_mode2=sketch_mode2, solve_mode=solve_mode)

            new_viewpoint_params = TempCamera(viewpoint)
            new_viewpoint_params.step(x)
            second_order_converged = x.norm() < second_order_converged_threshold

            new_viewpoint_params.assign(viewpoint)
            update_pose(viewpoint)

        # Save the outer losses to an npy file
        print(f"Saving outer losses to experiment/data/frame{args.frame:06d}_outer_losses_{sketch_mode1}_{sketch_mode2}_d-{d}.npy")
        np.save(f"experiment/data/frame{args.frame:06d}_outer_losses_{sketch_mode1}_{sketch_mode2}_d-{d}.npy", outer_losses)


