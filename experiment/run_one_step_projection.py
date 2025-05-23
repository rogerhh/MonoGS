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

def solve_projection_ls(A, b, x_true=None):
    m, n = A.shape
    x = torch.zeros((n), device=cuda_device)
    z = b - A @ x

    A_frob_sq = torch.sum(A ** 2)
    row_probs = torch.norm(A, dim=1) ** 2 / A_frob_sq
    col_probs = torch.norm(A, dim=0) ** 2 / A_frob_sq
    # row_probs = torch.ones(m, device=cuda_device) / m
    # col_probs = torch.ones(n, device=cuda_device) / n
    print(f"row_probs = {row_probs}, col_probs = {col_probs}")

    losses = []
    errors = []

    for _ in range(max_iters):
        col_block = torch.multinomial(col_probs, block_size, replacement=True)
        A_col_block = A[:, col_block] # / torch.sqrt(block_size * col_probs[col_block])

        z = z - A_col_block @ (torch.linalg.pinv(A_col_block) @ z)
        
        row_block = torch.multinomial(row_probs, block_size, replacement=True)
        # print(f"col_block = {col_block}, row_block = {row_block}")
        A_row_block = A[row_block, :] / torch.sqrt(block_size * row_probs[row_block].unsqueeze(1))
        b_row_block = b[row_block] / torch.sqrt(block_size * row_probs[row_block])
        z_row_block = z[row_block] / torch.sqrt(block_size * row_probs[row_block])
        residual = b_row_block - z_row_block - A_row_block @ x
        x = x + torch.linalg.pinv(A_row_block) @ residual

        loss1 = torch.norm(A @ x + z - b) ** 2
        loss2 = torch.norm(A.T @ z) ** 2
        x_diff_norm = torch.norm(x - x_true) if x_true is not None else None
        print(f"Inner loss: loss1 = {loss1:.6f}, loss2 = {loss2:.6f}, x_diff_norm = {x_diff_norm:.6f}")

        losses.append(torch.norm(A @ x - b).item())
        errors.append(x_diff_norm.item() if x_true is not None else None)

    # u_z = 0.5
    # u_x = 0.5

    # for _ in range(max_iters):
    #     col_index = torch.multinomial(col_probs, 1, replacement=True)
    #     A_col = A[:, col_index]
    #     z = z - u_z * ((A_col.T @ z) / (torch.norm(A_col) ** 2 + 1e-10) * A_col).squeeze(1)
    #     print(f"col_index = {col_index}")

    #     row_index = torch.multinomial(row_probs, 1, replacement=True)
    #     A_row = A[row_index, :]
    #     b_row = b[row_index]
    #     z_row = z[row_index]
    #     residual = b_row - z_row - A_row @ x
    #     x = x + u_x * (residual / (torch.norm(A_row) ** 2 + 1e-10) * A_row).squeeze(0)
    #     x_diff_norm = torch.norm(x - x_true) if x_true is not None else None
    #     print(f"Inner loss: x_diff_norm = {x_diff_norm:.6f}")

    print(f"x = {x}")
    print(f"x_true = {x_true}")
    print(f"loss = {torch.norm(A @ x - b)}")
    print(f"opt loss = {torch.norm(A @ x_true - b)}")

    return x, losses, errors

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

        for tracking_itr in range(10):
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

            x = torch.randn(n, device=cuda_device) * 0

            baseline_repeat_dim = 1
            baseline_stack_dim = 64
            baseline_sketch_dim = 64
            sketch_args = (config, height, width, tau_len, exposure_len, baseline_repeat_dim, baseline_stack_dim, baseline_sketch_dim, viewpoint, gaussians, pipeline_params, background)

            J, f = get_full_system(sketch_args)

            x_true = torch.linalg.pinv(J) @ (-f)

            max_rek_iters = 1000
            block_size = 2
            x, inner_loss, inner_error = solve_projection_ls(J, -f, x_true=x_true)

            if tracking_itr == 0:
                min_inner_loss = torch.norm(J @ x_true + f).item()

                data = {
                    "min_inner_loss": min_inner_loss,
                    "inner_losses": inner_losses,
                    "inner_errors": inner_errors,
                }

                # Save the data to a npy file
                np.save(f"experiment/data/frame{args.frame:06d}_rek_inner_losses_block-size-{block_size}.npy", data)
                print(f"Saved inner losses and errors data to experiment/data/frame{args.frame:06d}_rek_inner_losses_block-size-{block_size}.npy")

            new_viewpoint_params = TempCamera(viewpoint)
            new_viewpoint_params.step(x)
            second_order_converged = x.norm() < second_order_converged_threshold

            new_viewpoint_params.assign(viewpoint)
            update_pose(viewpoint)


        # Save the image
        save_torch_image(image, f"experiment/data/frame{args.frame:06d}_final_image_Adam.png")
        save_torch_image(viewpoint.original_image, f"experiment/data/frame{args.frame:06d}_original_image.png")
        # Randomized Kaczmarz method
