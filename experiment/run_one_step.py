import os
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

def gen_forward_sketch_args(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, sketch_index_offsets, sketch_index_vals, sketch_index_indices):

    n = tau_len + exposure_len
    m = height * width
    d = stack_dim * sketch_dim

    chunk_size = m // d
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

def gen_forward_sketch_args_kaczmarz(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, sketch_index_offsets, sketch_index_vals, sketch_index_indices, p):

    n = tau_len + exposure_len
    m = height * width
    d = stack_dim * sketch_dim
    # For randomized Kaczmarz, each sketched row corresponds to one pixel
    chunk_size = 1

    
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

    if False:
        # First run some iterations of first order optimization
        pose_optimizer = torch.optim.SGD(opt_params, momentum=0.9)
        for tracking_itr in range(num_iter):
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

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)

            pose_optimizer.zero_grad()
            loss_tracking_scalar.backward()

            for group in opt_params:
                for param in group["params"]:
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_norm_(param, 1.0)

            pose_optimizer.step()

            update_pose(viewpoint)

        # Save the image
        save_torch_image(image, f"experiment/data/frame{args.frame:06d}_final_image_SGD.png")
        save_torch_image(viewpoint.original_image, f"experiment/data/frame{args.frame:06d}_original_image.png")

        # Run Adam optimization
        orig_viewpoint.assign(viewpoint)
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(num_iter):
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

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)

            pose_optimizer.zero_grad()
            loss_tracking_scalar.backward()

            pose_optimizer.step()

            update_pose(viewpoint)

        # Save the image
        save_torch_image(image, f"experiment/data/frame{args.frame:06d}_final_image_Adam.png")
        save_torch_image(viewpoint.original_image, f"experiment/data/frame{args.frame:06d}_original_image.png")

    if True:
        # First run some iterations of first order optimization
        pose_optimizer = torch.optim.SGD(opt_params, momentum=0.9)
        for tracking_itr in range(num_iter):
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

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)

            pose_optimizer.zero_grad()
            loss_tracking_scalar.backward()

            for group in opt_params:
                for param in group["params"]:
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_norm_(param, 1.0)

            pose_optimizer.step()

            update_pose(viewpoint)

        # Save the image
        save_torch_image(image, f"experiment/data/frame{args.frame:06d}_final_image_SGD.png")
        save_torch_image(viewpoint.original_image, f"experiment/data/frame{args.frame:06d}_original_image.png")

        # Run Adam optimization
        orig_viewpoint.assign(viewpoint)
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(num_iter):
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

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)

            pose_optimizer.zero_grad()
            loss_tracking_scalar.backward()

            pose_optimizer.step()

            update_pose(viewpoint)

        # Save the image
        save_torch_image(image, f"experiment/data/frame{args.frame:06d}_final_image_randLS.png")
        save_torch_image(viewpoint.original_image, f"experiment/data/frame{args.frame:06d}_original_image.png")

    if True:
        repeat_dim = 1
        stack_dim = 2
        sketch_dim = 64

        initial_lambda = config["Training"]["RGN"]["second_order"]["initial_lambda"]
        second_order_converged_threshold = config["Training"]["RGN"]["second_order"]["converged_threshold"]

        lambda_ = initial_lambda

        # Run Randomized LS optimization
        height, width = viewpoint.image_height, viewpoint.image_width
        tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
        exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
        m = height * width
        n = tau_len + exposure_len
        d = stack_dim * sketch_dim
        chunk_size = m // d

        # These are static tensors so can initialize them here
        sketch_index_offsets = (torch.arange(stack_dim, dtype=torch.int32, device=cuda_device) * (m)).repeat_interleave(chunk_size*d // stack_dim)
        sketch_index_vals = torch.arange(sketch_dim, dtype=torch.int32, device=cuda_device).repeat_interleave(chunk_size).repeat(stack_dim)
        sketch_index_indices = torch.arange(chunk_size*d, dtype=torch.int32, device=cuda_device)

        orig_viewpoint.assign(viewpoint)
        pose_optimizer = torch.optim.Adam(opt_params)

        for tracking_itr in range(10):
            forward_sketch_args = gen_forward_sketch_args(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, sketch_index_offsets, sketch_index_vals, sketch_index_indices)
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
            print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, l2 loss = {l2_loss.item():.4f}")

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)

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

                print(f"SJ norm = {SJ.norm()}, Sf norm = {Sf.norm()}")

                damped_SJ = torch.cat((SJ, torch.eye(n, device=cuda_device) * math.sqrt(lambda_)), dim=0)
                damped_Sf = torch.cat((Sf, torch.zeros(n, device=cuda_device)), dim=0)

                second_order_ls_solve_start = time.time()

                x = torch.linalg.lstsq(damped_SJ, -damped_Sf).solution

                torch.cuda.synchronize()
                second_order_ls_solve_end = time.time()

                # DEBUG
                distortion = math.sqrt(n / (repeat_dim * d))
                gamma = (1 + distortion) / (1 - distortion)
                min_sigma = torch.linalg.svdvals(damped_SJ)[-1]
                residual = torch.norm(Sf - SJ @ x) + math.sqrt(lambda_) * torch.norm(x)
                upperbound = residual * gamma * math.sqrt(gamma ** 2 - 1) / min_sigma
                # DEBUG END

                new_viewpoint_params = TempCamera(viewpoint)
                new_viewpoint_params.step(x)
                second_order_converged = x.norm() < second_order_converged_threshold

                new_viewpoint_params.assign(viewpoint)
                update_pose(viewpoint)


        # Save the image
        save_torch_image(image, f"experiment/data/frame{args.frame:06d}_final_image_randLS.png")
        save_torch_image(viewpoint.original_image, f"experiment/data/frame{args.frame:06d}_original_image.png")

    if False:
        # Randomized Kaczmarz method
        repeat_dim = 1
        stack_dim = config["Training"]["RGN"]["second_order"]["stack_dim"]
        sketch_dim = config["Training"]["RGN"]["second_order"]["sketch_dim"]

        initial_lambda = config["Training"]["RGN"]["second_order"]["initial_lambda"]
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

        # These are static tensors so can initialize them here
        sketch_index_offsets = (torch.arange(stack_dim, dtype=torch.int32, device=cuda_device) * (m)).repeat_interleave(chunk_size*d // stack_dim)
        sketch_index_vals = torch.arange(sketch_dim, dtype=torch.int32, device=cuda_device).repeat_interleave(chunk_size).repeat(stack_dim)
        sketch_index_indices = torch.arange(chunk_size*d, dtype=torch.int32, device=cuda_device)

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

            for inner_itr in range(20):
                forward_sketch_args = gen_forward_sketch_args_kaczmarz(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim, sketch_index_offsets, sketch_index_vals, sketch_index_indices, p_vector)
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

                if loss_tracking_scalar < best_loss_scalar:
                    best_loss_scalar = loss_tracking_scalar
                    best_viewpoint_params = TempCamera(viewpoint)

                sketch_dim_per_iter = stack_dim * sketch_dim
                loss_tracking_img = loss_tracking_img.sum(dim=0) / (m / sketch_dim_per_iter)
                loss_vector = loss_tracking_img.flatten()

                rand_weights = forward_sketch_args["rand_weights"]
                rand_indices_row = forward_sketch_args["rand_indices_row"]
                rand_indices_col = forward_sketch_args["rand_indices_col"]

                weighted_loss_tracking_img = loss_tracking_img * rand_weights
                A_block = torch.empty((repeat_dim, stack_dim, sketch_dim, n), device=cuda_device)
                batch_indices = torch.arange(repeat_dim).view(-1, 1, 1, 1)
                b_block = weighted_loss_tracking_img[batch_indices, rand_indices_row, rand_indices_col].sum(dim=-1)

                # TODO: divide by probability

                for i in range(repeat_dim):

                    viewpoint.cam_rot_delta.grad = None
                    viewpoint.cam_trans_delta.grad = None
                    viewpoint.exposure_a.grad = None
                    viewpoint.exposure_b.grad = None
                    forward_sketch_args["sketch_dtau"].grad = None
                    forward_sketch_args["sketch_dexposure"].grad = None

                    weighted_loss_tracking_img[i].backward(gradient=torch.ones_like(weighted_loss_tracking_img[i]), retain_graph=True)
                    A_block_i = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=2)
                    A_block[i] = A_block_i

                with torch.no_grad():
                    SJ = SJ.reshape((-1, n)) / math.sqrt(repeat_dim)
                    Sf = Sf.flatten() / math.sqrt(repeat_dim)

                    damped_SJ = torch.cat((SJ, torch.eye(n, device=cuda_device) * math.sqrt(lambda_)), dim=0)
                    damped_Sf = torch.cat((Sf, torch.zeros(n, device=cuda_device)), dim=0)

                    second_order_ls_solve_start = time.time()

                    x = torch.linalg.lstsq(damped_SJ, -damped_Sf).solution

                    torch.cuda.synchronize()
                    second_order_ls_solve_end = time.time()

                    # DEBUG
                    distortion = math.sqrt(n / (repeat_dim * d))
                    gamma = (1 + distortion) / (1 - distortion)
                    min_sigma = torch.linalg.svdvals(damped_SJ)[-1]
                    residual = torch.norm(Sf - SJ @ x) + math.sqrt(lambda_) * torch.norm(x)
                    upperbound = residual * gamma * math.sqrt(gamma ** 2 - 1) / min_sigma
                    # DEBUG END

                    new_viewpoint_params = TempCamera(viewpoint)
                    new_viewpoint_params.step(x)
                    second_order_converged = x.norm() < second_order_converged_threshold

                    new_viewpoint_params.assign(viewpoint)
                    update_pose(viewpoint)


        # Save the image
        save_torch_image(image, f"experiment/data/frame{args.frame:06d}_final_image_Adam.png")
        save_torch_image(viewpoint.original_image, f"experiment/data/frame{args.frame:06d}_original_image.png")
