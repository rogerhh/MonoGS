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
        outer_losses = []
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

            outer_losses.append(loss_tracking_scalar.item())

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

        # Save the outer losses to an npy file
        np.save(f"experiment/data/frame{args.frame:06d}_outer_losses_Adam.npy", outer_losses)
