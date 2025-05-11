import time

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

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gui import gui_utils
from utils.camera_utils import Camera, CameraMsg
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
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
        # viewpoint.T = self.T
        # viewpoint.cam_rot_delta = nn.Parameter(self.cam_rot_delta)
        # viewpoint.cam_trans_delta = nn.Parameter(self.cam_trans_delta)
        # viewpoint.exposure_a = nn.Parameter(self.exposure_a)
        # viewpoint.exposure_b = nn.Parameter(self.exposure_b)

    def step(self, x):
        self.cam_trans_delta.data += x[:3]
        self.cam_rot_delta.data += x[3:6]
        self.exposure_a.data += x[6]
        self.exposure_b.data += x[7]

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = cuda_device
        self.pause = False

        # LM Parameters
        self.use_huber = self.config["Training"]["RGN"]["use_huber"]
        self.huber_delta = self.config["Training"]["RGN"]["huber_delta"]
        self.first_order_max_iter = self.config["Training"]["RGN"]["first_order"]["max_iter"]
        self.first_order_fast_iter = self.config["Training"]["RGN"]["first_order"]["fast_iter"]
        self.first_order_num_backward_gaussians = self.config["Training"]["RGN"]["first_order"]["num_backward_gaussians"]
        self.first_order_num_pixels = self.config["Training"]["RGN"]["first_order"]["num_pixels"]
        self.second_order_max_iter = self.config["Training"]["RGN"]["second_order"]["max_iter"]
        self.second_order_num_backward_gaussians = self.config["Training"]["RGN"]["second_order"]["num_backward_gaussians"]
        self.pnorm = self.config["Training"]["RGN"]["pnorm"]
        self.repeat_dim = self.config["Training"]["RGN"]["second_order"]["repeat_dim"]
        self.stack_dim = self.config["Training"]["RGN"]["second_order"]["stack_dim"]
        self.sketch_dim = self.config["Training"]["RGN"]["second_order"]["sketch_dim"]
        self.initial_lambda = self.config["Training"]["RGN"]["second_order"]["initial_lambda"]
        self.max_lambda = self.config["Training"]["RGN"]["second_order"]["max_lambda"]
        self.min_lambda = self.config["Training"]["RGN"]["second_order"]["min_lambda"]
        self.increase_factor = self.config["Training"]["RGN"]["second_order"]["increase_factor"]
        self.decrease_factor = self.config["Training"]["RGN"]["second_order"]["decrease_factor"]
        self.trust_region_cutoff = self.config["Training"]["RGN"]["second_order"]["trust_region_cutoff"]
        self.second_order_converged_threshold = self.config["Training"]["RGN"]["second_order"]["converged_threshold"]
        self.use_nonmonotonic_step = self.config["Training"]["RGN"]["second_order"]["use_nonmonotonic_step"]
        self.use_first_order_best = self.config["Training"]["RGN"]["second_order"]["use_first_order_best"]
        self.use_best_loss = self.config["Training"]["RGN"]["use_best_loss"]
        self.override_mode = self.config["Training"]["RGN"]["override"]["mode"]
        self.override_first_logdir = self.config["Training"]["RGN"]["override"]["first_logdir"]

        self.print_output = self.config["Training"]["RGN"]["print_output"]
        self.log_output = self.config["Training"]["RGN"]["log_output"]
        self.log_basedir = self.config["Training"]["RGN"]["log_basedir"]
        self.log_path = time.strftime("%Y%m%d_%H%M")  # Set logfile base path to be yyyymmdd_HHMM

        self.run_twice = self.config["Training"]["RGN"]["experimental"]["run_twice"]
        self.run_twice_first_only = self.config["Training"]["RGN"]["experimental"]["run_twice_first_only"]
        self.run_twice_override_gt = self.config["Training"]["RGN"]["experimental"]["run_twice_override_gt"]

        self.save_period = self.config["Training"]["RGN"]["save_period"]
        self.experiment_step = self.config["Training"]["experiment_step"]
        
        self.all_profile_data = []
        self.all_profile_data_firstonly = []

        if self.log_output:
            # Check if log_basedir/log_path exists
            self.logdir = os.path.join(self.log_basedir, self.log_path)
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            # Dump the config to the logdir as a json file
            with open(os.path.join(self.logdir, "config.json"), "w") as f:
                json.dump(self.config, f)

        self.tracking_time_sum = 0
        self.first_order_time_sum = 0
        self.first_order_render_sum = 0
        self.first_order_compute_loss_sum = 0
        self.first_order_backward_sum = 0
        self.first_order_count = 0
        self.second_order_time_sum = 0
        self.second_order_forward_sum = 0
        self.second_order_gen_random_sum = 0
        self.second_order_render_sum = 0
        self.second_order_compute_loss_sum = 0
        self.second_order_backward_setup_sum = 0
        self.second_order_backward_sum = 0
        self.second_order_ls_solve_sum = 0
        self.second_order_update_sum = 0
        self.second_order_count = 0

        self.last_kf_idx = 1

        width = self.config["Dataset"]["Calibration"]["width"]
        height = self.config["Dataset"]["Calibration"]["height"]
        tau_len = 6
        exposure_len = 2
        n = tau_len + exposure_len
        m = height * width
        d = self.stack_dim * self.sketch_dim
        chunk_size = m // d

        # These are static tensors so can initialize them here
        self.sketch_index_offsets = (torch.arange(self.stack_dim, dtype=torch.int32, device=self.device) * (chunk_size*d)).repeat_interleave(chunk_size*d // self.stack_dim)
        self.sketch_index_vals = torch.arange(self.sketch_dim, dtype=torch.int32, device=self.device).repeat_interleave(chunk_size).repeat(self.stack_dim)
        self.sketch_index_indices = torch.arange(chunk_size*d, dtype=torch.int32, device=self.device)


    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        
        self.use_gui = self.config["Results"]["use_gui"]
        self.constant_velocity_warmup = 200 # TODO: fix hardcoding

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.to(self.device)
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.T = viewpoint.T_gt

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def gen_default_forward_sketch_args(self):
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

    def gen_forward_sketch_args(self, height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim):

        n = tau_len + exposure_len
        m = height * width
        d = stack_dim * sketch_dim
        chunk_size = m // d

        # # DEBUG
        # self.sketch_index_offsets = (torch.arange(stack_dim, dtype=torch.int32, device=self.device) * (m)).repeat_interleave(chunk_size*d // stack_dim)
        # self.sketch_index_vals = torch.arange(sketch_dim, dtype=torch.int32, device=self.device).repeat_interleave(chunk_size).repeat(stack_dim)
        # self.sketch_index_indices = torch.arange(chunk_size*d, dtype=torch.int32, device=self.device)
        # # DEBUG END

        """
        chunk_size = m // d
        rand_flat_indices = torch.empty((repeat_dim, chunk_size*d), dtype=torch.int32, device=self.device)

        for i in range(repeat_dim):
            rand_flat_indices[i] = torch.randperm(m, dtype=torch.int32, device=self.device)[:(chunk_size*d)]
         
        rand_flat_indices = rand_flat_indices.reshape((repeat_dim, stack_dim, sketch_dim, -1))

        rand_indices_row = rand_flat_indices // width
        rand_indices_col = rand_flat_indices % width
        rand_indices = (rand_indices_row, rand_indices_col)


        sketch_indices = torch.ones((repeat_dim, stack_dim, height, width), dtype=torch.int32, device=self.device) * (-1)

        i_values = torch.arange(sketch_dim, dtype=torch.int32, device=self.device).view(1, 1, -1, 1)

        sketch_indices[torch.arange(repeat_dim).view(-1, 1, 1, 1), 
                       torch.arange(stack_dim).view(1, -1, 1, 1),
                       rand_indices_row, rand_indices_col] = i_values
        """
        chunk_size = m // d
        rand_flat_indices = torch.empty((repeat_dim, chunk_size*d), dtype=torch.int32, device=self.device)
        
        sketch_indices = torch.ones((repeat_dim, stack_dim*height*width), dtype=torch.int32, device=self.device) * (-1)
        rand_indices_row = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=self.device)
        rand_indices_col = torch.empty((repeat_dim, stack_dim, sketch_dim, chunk_size), dtype=torch.int32, device=self.device)
        
        streams = [torch.cuda.Stream(device=self.device) for _ in range(repeat_dim)]
        
        for i in range(repeat_dim):
            with torch.cuda.stream(streams[i]):
                rand_flat_indices[i] = torch.randperm(m, dtype=torch.int32, device=self.device)[:(chunk_size*d)]
                sketch_indices[i, rand_flat_indices[i, self.sketch_index_indices] + self.sketch_index_offsets] = self.sketch_index_vals
                rand_indices_row_i = rand_flat_indices[i] // width
                rand_indices_col_i = rand_flat_indices[i] % width
                
                rand_indices_row[i] = rand_indices_row_i.view(stack_dim, sketch_dim, chunk_size)
                rand_indices_col[i] = rand_indices_col_i.view(stack_dim, sketch_dim, chunk_size)
         
        torch.cuda.synchronize()
        sketch_indices = sketch_indices.reshape((repeat_dim, stack_dim, height, width))
        rand_indices = (rand_indices_row, rand_indices_col)
        

        rand_weights = torch.randint(0, 2, size=(repeat_dim, height, width), dtype=torch.float32, device=self.device) * 2 - 1

        sketch_dtau = torch.empty((stack_dim, sketch_dim, tau_len), dtype=torch.float32, device=self.device, requires_grad=True)
        sketch_dexposure = torch.empty((stack_dim, sketch_dim, exposure_len), dtype=torch.float32, device=self.device, requires_grad=True)

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

    def tracking(self, cur_frame_idx, viewpoint, run_twice_first_rep=True, run_twice_override_gt=True):
        print(f"Frame: {cur_frame_idx}")

        if cur_frame_idx == self.experiment_step:
            return self.tracking_experiment(cur_frame_idx, viewpoint)
            exit()

        # prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        # viewpoint.T = prev.T
        if self.initialized and cur_frame_idx > self.constant_velocity_warmup and self.monocular:
            prev_prev = self.cameras[cur_frame_idx - self.use_every_n_frames -1 ]
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            
            pose_prev_prev = prev_prev.T
            pose_prev = prev.T
            velocity = pose_prev @ torch.linalg.inv(pose_prev_prev)
            pose_new = velocity @ pose_prev
            viewpoint.T = pose_new
            viewpoint.T = pose_prev
        else:
            # if True:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.T = prev.T

        if self.run_twice:
            if run_twice_first_rep:
                orig_viewpoint = TempCamera(viewpoint)
                first_only = not self.run_twice_first_only
                run_twice_override_gt = False
            else:
                first_only = self.run_twice_first_only
        else: 
            first_only = False
            run_twice_override_gt = False

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": self.config["Training"]["lr"]["exposure_a"],
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": self.config["Training"]["lr"]["exposure_b"],
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )
        
        # LM Parameters for convenience

        # if first only, set f100s0
        if first_only:
            first_order_max_iter = 100
            second_order_max_iter = 0
        else:
            first_order_max_iter = self.first_order_max_iter
            second_order_max_iter = self.second_order_max_iter

        first_order_fast_iter = self.first_order_fast_iter
        first_order_num_backward_gaussians = self.first_order_num_backward_gaussians
        first_order_num_pixels = self.first_order_num_pixels
        second_order_num_backward_gaussians = self.second_order_num_backward_gaussians
        repeat_dim = self.repeat_dim
        stack_dim = self.stack_dim
        sketch_dim = self.sketch_dim
        initial_lambda = self.initial_lambda
        max_lambda = self.max_lambda
        min_lambda = self.min_lambda
        increase_factor = self.increase_factor
        decrease_factor = self.decrease_factor
        trust_region_cutoff = self.trust_region_cutoff
        second_order_converged_threshold = self.second_order_converged_threshold
        use_nonmonotonic_step = self.use_nonmonotonic_step
        override_mode = self.override_mode if not run_twice_override_gt else "gt"
        print_output = self.print_output
        log_output = self.log_output
        save_period = self.save_period

        max_iter = first_order_max_iter + second_order_max_iter
        in_second_order = False
        first_order_countdown = 10
        second_order_countup = 0

        best_loss_scalar = float("inf")
        best_trans_error = float("inf")
        best_output = None
        best_viewpoint_params = None
        # best_trans_error = float("inf")
        # best_angle_error = float("inf")
        lambda_ = initial_lambda
        new_viewpoint_params = None
        SJ = None
        Sf = None
        H = None
        g = None
        avg_len = 0
        x_avg = None

        last_kf = self.cameras[self.last_kf_idx]

        # Some constants
        height, width = viewpoint.image_height, viewpoint.image_width
        tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
        exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
        m = height * width
        n = tau_len + exposure_len
        d = stack_dim * sketch_dim

        forward_sketch_args = self.gen_default_forward_sketch_args()

        profile_data = {"timestamps": [], "losses": [], "pose": None, "rasterize_gaussians_backward_time_ms": [], "rasterize_gaussians_C_backward_time_ms": [], "distortions": [], "min_sigmas": [], "residuals": [], "upperbounds": []}

        tracking_start = time.time()
        pose_optimizer = torch.optim.Adam(opt_params)

        tracking_itr = -1
        sketch_solution_ok = True
        # for tracking_itr in range(max_iter):
        while True:
            if not (tracking_itr + 1 < max_iter or not sketch_solution_ok):
                # print(f"tracking_itr = {tracking_itr}, sketch_solution_ok = {sketch_solution_ok}")
                break
            # while tracking_itr + 1 < max_iter or not sketch_solution_ok:
            # DEBUG
            # Some constants
            height, width = viewpoint.image_height, viewpoint.image_width
            tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
            exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
            m = height * width
            n = tau_len + exposure_len
            d = stack_dim * sketch_dim
            # DEBUG END

            tracking_itr += 1
            tracking_iter_start = time.time()

            if log_output:
                profile_data["timestamps"].append(time.time())

            in_second_order = tracking_itr >= first_order_max_iter
            # in_second_order = first_order_countdown <= 0
            # first_order_countdown -= 1

            if tracking_itr == first_order_max_iter:
                if print_output:
                    print("Switching to second order optimization")

                if best_viewpoint_params is not None and self.use_first_order_best:
                    best_viewpoint_params.assign(viewpoint)

            # If in second order and new_viewpoint_params is not None
            # Then cache the old data
            if in_second_order and new_viewpoint_params is not None:
                with torch.no_grad():
                    old_loss_scalar = loss_tracking_scalar
                    old_output = (TempCamera(viewpoint), render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, )
                    new_viewpoint_params.assign(viewpoint)
                    update_pose(viewpoint)

            # Set up sketching in the forward pass
            if in_second_order:
                second_order_random_gen_start = time.time()
                forward_sketch_args = self.gen_forward_sketch_args(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim)
                second_order_random_gen_end = time.time()

            if tracking_itr >= first_order_fast_iter:
                first_order_num_backward_gaussians = -1

            # torch.cuda.synchronize()
            render_start = time.time()

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
            )

            # torch.cuda.synchronize()
            render_end = time.time()

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
            )

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)

            # torch.cuda.synchronize()
            compute_loss_end = time.time()

            if print_output:
                trans_error, angle_error = relative_pose_error(last_kf.T_gt, viewpoint.T_gt, last_kf.T, viewpoint.T)
                print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, trans_error = {trans_error:.4f}, angle_error = {angle_error:.4f}")

            if log_output:
                profile_data["losses"].append(loss_tracking_scalar.item())

            # DEBUG
            if loss_tracking_scalar < best_loss_scalar:
            # if 0.5 * trans_error + angle_error < best_trans_error:
                # best_trans_error = 0.5 * trans_error + angle_error
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)
                best_output = (best_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, )
                # best_trans_error = trans_error
                # best_angle_error = angle_error

            is_new_step = True
            second_to_first = False
            if in_second_order and new_viewpoint_params is not None:
                # If new step is better than old step, then take it
                if loss_tracking_scalar < old_loss_scalar:
                    lambda_ = max(lambda_ / decrease_factor, min_lambda)
                else:
                    lambda_ = min(lambda_ * increase_factor, max_lambda)
                    # If only allowing strictly decreasing steps, then revert to old viewpoint
                    if not use_nonmonotonic_step:
                        is_new_step = False

            # DEBUG
            is_new_step = True

            if not is_new_step:
                old_viewpoint_params, _, _, _, _, _, _, = old_output
                old_viewpoint_params.assign(viewpoint)
                loss_tracking_scalar = old_loss_scalar
                old_SJ = SJ
                old_Sf = Sf

                # We need to run rendering again to align with the new random indices
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                loss_tracking_img = get_loss_tracking_per_pixel(
                    self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                )

            # torch.cuda.synchronize()
            forward_end = time.time()

            if not in_second_order:
                first_order_opt_start = time.time()

                # image._grad_fn.tracking = True

                # Get l1 norm of loss_tracking
                # DEBUG
                num_pixels = first_order_num_pixels
                if num_pixels > 0:
                    # Need to first sum across channels
                    loss_tracking_vec1 = torch.sum(torch.abs(loss_tracking_img), dim=0).flatten() + 1e-8
                    with torch.no_grad():
                        dist = loss_tracking_vec1 / torch.sum(loss_tracking_vec1)
                        selected_pixel_indices = torch.multinomial(dist, num_pixels, replacement=True)
                    loss_tracking_vec = loss_tracking_vec1 / dist
                    loss_tracking = torch.sum(loss_tracking_vec[selected_pixel_indices])
                    loss_tracking = loss_tracking / num_pixels

                    image._grad_fn.select_pixels = True
                    image._grad_fn.selected_pixel_indices = selected_pixel_indices

                    raise ValueError("Need to redo pixel subsampling to account for correct scaling under different norms")

                # DEBUG END

                else:
                    if self.use_huber:
                        loss_tracking_img = HuberLoss.apply(loss_tracking_img, self.huber_delta)
                        loss_tracking = torch.norm(loss_tracking_img.flatten(), p=2)
                    else:
                        loss_tracking = torch.norm(loss_tracking_img.flatten(), p=self.pnorm)

                # subsample_end = time.time()

                with torch.no_grad():
                    first_order_backward_start = time.time()
                    pose_optimizer.zero_grad()
                    loss_tracking.backward()

                    first_order_backward_end = time.time()

                    pose_optimizer.step()

                    first_order_step_end = time.time()

                    first_order_converged = update_pose(viewpoint)
                    first_order_update_pose_end = time.time()

                    # update_pose_end = time.time()
                first_order_opt_end = time.time()

                # print(f"First order render time ms: {(render_end - tracking_iter_start) * 1000}")
                # print(f"First order backward time ms: {(first_order_backward_end - first_order_backward_start) * 1000}")
                # print(f"First order step time ms: {(first_order_step_end - first_order_backward_end) * 1000}")
                # print(f"First order update pose time ms: {(first_order_update_pose_end - first_order_step_end) * 1000}")
                # print(f"First order total time ms: {(first_order_opt_end - tracking_iter_start) * 1000}")

                if first_order_converged:
                    if print_output:
                        print("First order optimization converged")
                    break

            else:

                torch.cuda.synchronize()
                second_order_backward_setup_start = time.time()

                sketch_dim_per_iter = stack_dim * sketch_dim
                if self.use_huber:
                    loss_tracking_img = HuberLoss.apply(loss_tracking_img, self.huber_delta)
                loss_tracking_img = loss_tracking_img.sum(dim=0) / (m / sketch_dim_per_iter)

                rand_weights = forward_sketch_args["rand_weights"]
                rand_indices_row = forward_sketch_args["rand_indices_row"]
                rand_indices_col = forward_sketch_args["rand_indices_col"]

                weighted_loss_tracking_img = loss_tracking_img * rand_weights
                SJ = torch.empty((repeat_dim, stack_dim, sketch_dim, n), device=self.device)
                batch_indices = torch.arange(repeat_dim).view(-1, 1, 1, 1)
                Sf = weighted_loss_tracking_img[batch_indices, rand_indices_row, rand_indices_col].sum(dim=-1)

                torch.cuda.synchronize()
                second_order_backward_setup_end = time.time()

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

                torch.cuda.synchronize()
                second_order_backward_end = time.time()

                with torch.no_grad():
                    SJ = SJ.reshape((-1, n)) / math.sqrt(repeat_dim)
                    Sf = Sf.flatten() / math.sqrt(repeat_dim)

                    damped_SJ = torch.cat((SJ, torch.eye(n, device=self.device) * math.sqrt(lambda_)), dim=0)
                    damped_Sf = torch.cat((Sf, torch.zeros(n, device=self.device)), dim=0)

                    second_order_ls_solve_start = time.time()

                    x = torch.linalg.lstsq(damped_SJ, -damped_Sf).solution

                    # H = SJ.T @ SJ / eta
                    # g = SJ.T @ Sf / eta
                    # H_damp = H + torch.eye(n, device=self.device) * lambda_
                    # x = torch.linalg.solve(H_damp, -g)

                    torch.cuda.synchronize()
                    second_order_ls_solve_end = time.time()

                    # DEBUG
                    distortion = math.sqrt(n / (repeat_dim * d))
                    gamma = (1 + distortion) / (1 - distortion)
                    min_sigma = torch.linalg.svdvals(damped_SJ)[-1]
                    residual = torch.norm(Sf - SJ @ x) + math.sqrt(lambda_) * torch.norm(x)
                    upperbound = residual * gamma * math.sqrt(gamma ** 2 - 1) / min_sigma
                    # # print(f"distortion = {distortion:.4f}, min_sigma = {min_sigma:.4f}, residual = {residual:.4f}, upperbound = {upperbound:.4f}")
                    # target_upperbound = 0.1
                    # if upperbound > target_upperbound:
                    #     sketch_solution_ok = False

                    #     upperbound_sq = upperbound ** 2
                    #     gamma_sq = gamma ** 2
                    #     K_sq = upperbound_sq / (gamma_sq * (gamma_sq - 1))
                    #     # Solve the quadratic equation: upperbound^2 = K^2 * gamma^2 * (gamma^2 - 1)
                    #     target_upperbound_sq = target_upperbound ** 2
                    #     target_gamma_sq = 0.5 * (1 + math.sqrt(1 + 4 * (target_upperbound_sq / K_sq)))
                    #     target_gamma = math.sqrt(target_gamma_sq)

                    #     assert(target_gamma < gamma), f"target_gamma = {target_gamma:.4f}, gamma = {gamma:.4f}"
                    #     assert(target_gamma > 1), f"target_gamma = {target_gamma:.4f}, gamma = {gamma:.4f}"
                    #     target_distortion = (target_gamma - 1) / (target_gamma + 1)
                    #     assert(distortion > target_distortion), f"target_distortion = {target_distortion:.4f}, distortion = {distortion:.4f}"
                    #     distortion_factor = distortion / target_distortion
                    #     # print(f"gamma = {gamma:.4f}, target_gamma = {target_gamma:.4f}, target_distortion = {target_distortion:.4f}, distortion = {distortion:.4f}, distortion_factor = {distortion_factor:.4f}")
                    #     multiplier = 2 ** math.ceil(0.5 * math.log(distortion_factor))
                    #     assert(multiplier >= 1)
                    # else:
                    #     sketch_solution_ok = True
                    #     multiplier = 1
                    # stack_dim *= int(multiplier)

                    # print(f"stack_dim = {stack_dim}, multiplier = {multiplier:.4f}, distortion = {distortion:.4f}, min_sigma = {min_sigma:.4f}, residual = {residual:.4f}, upperbound = {upperbound:.4f}")

                    # if tracking_itr < 30:
                    #     sketch_solution_ok = False
                    #     stack_dim += 4
                    # else:
                    #     sketch_solution_ok = True

                    # DEBUG END

                    if print_output or log_output:

                        if print_output:
                            print(f"distortion = {distortion:.4f}, min_sigma = {min_sigma:.4f}, residual = {residual:.4f}, upperbound = {upperbound:.4f}")

                        if log_output:
                            profile_data["distortions"].append(distortion)
                            profile_data["min_sigmas"].append(min_sigma.item())
                            profile_data["residuals"].append(residual.item())
                            profile_data["upperbounds"].append(upperbound.item())

                    new_viewpoint_params = TempCamera(viewpoint)
                    new_viewpoint_params.step(x)
                    second_order_converged = x.norm() < second_order_converged_threshold

                    torch.cuda.synchronize()
                    second_order_update_end = time.time()

                    if print_output:
                        print(f"step norm = {x.norm()}, lambda = {lambda_:.4f}")
                        # print(f"x = {x}")

                    if second_order_converged:
                        if print_output:
                            print(f"step norm {x.norm():.4f} < {second_order_converged_threshold}. Second order optimization converged")
                        break

                second_order_solve_end = time.time()
                # print(f"Forward time ms: {(forward_end - tracking_iter_start) * 1000}")
                # print(f"Second order random gen1 time ms: {(second_order_random_gen1_end - tracking_iter_start) * 1000}")
                # print(f"Second order random gen time ms: {(second_order_random_gen_end - tracking_iter_start) * 1000}")
                # print(f"Second order backward time ms: {(second_order_backward_end - second_order_setup_end) * 1000}")
                # print(f"Second order ls solve time ms: {(second_order_ls_solve_end - second_order_ls_solve_start) * 1000}")
                # print(f"Second order solve time ms: {(second_order_solve_end - second_order_backward_end) * 1000}")



                
            tracking_itr_end = time.time()
            if in_second_order:
                self.second_order_time_sum += tracking_itr_end - tracking_iter_start
                self.second_order_forward_sum += forward_end - tracking_iter_start
                self.second_order_gen_random_sum += second_order_random_gen_end - second_order_random_gen_start
                self.second_order_render_sum += render_end - render_start
                self.second_order_compute_loss_sum += compute_loss_end - render_end
                self.second_order_backward_setup_sum += second_order_backward_setup_end - second_order_backward_setup_start
                self.second_order_backward_sum += second_order_backward_end - second_order_backward_setup_end
                self.second_order_ls_solve_sum += second_order_ls_solve_end - second_order_ls_solve_start
                self.second_order_update_sum += second_order_update_end - second_order_ls_solve_end
                self.second_order_count += 1
            else:
                self.first_order_time_sum += tracking_itr_end - tracking_iter_start
                self.first_order_render_sum += render_end - render_start
                self.first_order_compute_loss_sum += compute_loss_end - render_end
                self.first_order_backward_sum += first_order_backward_end - first_order_backward_start
                self.first_order_count += 1

            if tracking_itr % 50 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=CameraMsg(viewpoint),
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )

        if print_output:
            print("Tracking converged in {} iterations".format(tracking_itr))

        tracking_end = time.time()
        # print(f"Tracking time ms: {(tracking_end - tracking_start) * 1000}")
        self.tracking_time_sum += tracking_end - tracking_start

        if (cur_frame_idx + 1) % 10 == 0:
            print(f"Average tracking time ms: {(self.tracking_time_sum / 10) * 1000}")
            avg_first_order_time = 0
            avg_second_order_time = 0
            if self.first_order_count > 0:
                avg_first_order_time = self.first_order_time_sum / self.first_order_count
                avg_first_order_render = self.first_order_render_sum / self.first_order_count
                avg_first_order_compute_loss = self.first_order_compute_loss_sum / self.first_order_count
                avg_first_order_backward = self.first_order_backward_sum / self.first_order_count
                print(f"Average first order time ms: {avg_first_order_time * 1000}")
                print(f"Average first order render time ms: {avg_first_order_render * 1000}")
                print(f"Average first order compute loss time ms: {avg_first_order_compute_loss * 1000}")
                print(f"Average first order backward time ms: {avg_first_order_backward * 1000}")
            if self.second_order_count > 0:
                avg_second_order_time = self.second_order_time_sum / self.second_order_count
                avg_second_order_forward = self.second_order_forward_sum / self.second_order_count
                avg_second_order_gen_random = self.second_order_gen_random_sum / self.second_order_count
                avg_second_order_render = self.second_order_render_sum / self.second_order_count
                avg_second_order_compute_loss = self.second_order_compute_loss_sum / self.second_order_count
                avg_second_order_backward_setup = self.second_order_backward_setup_sum / self.second_order_count
                avg_second_order_backward = self.second_order_backward_sum / self.second_order_count
                avg_second_order_ls_solve = self.second_order_ls_solve_sum / self.second_order_count
                avg_second_order_update = self.second_order_update_sum / self.second_order_count
                print(f"Average second order forward time ms: {avg_second_order_forward * 1000}")
                print(f"Average second order gen random time ms: {avg_second_order_gen_random * 1000}")
                print(f"Average second order render time ms: {avg_second_order_render * 1000}")
                print(f"Average second order compute loss time ms: {avg_second_order_compute_loss * 1000}")
                print(f"Average second order backward setup time ms: {avg_second_order_backward_setup * 1000}")
                print(f"Average second order backward time ms: {avg_second_order_backward * 1000}")
                print(f"Average second order ls solve time ms: {avg_second_order_ls_solve * 1000}")
                print(f"Average second order update time ms: {avg_second_order_update * 1000}")
                print(f"Average second order time ms: {avg_second_order_time * 1000}")
            projected_tracking_time = (avg_first_order_time * first_order_max_iter + avg_second_order_time * second_order_max_iter)
            print(f"Projected tracking time ms = {(projected_tracking_time) * 1000}")
            self.tracking_time_sum = 0
            self.first_order_time_sum = 0
            self.first_order_render_sum = 0
            self.first_order_compute_loss_sum = 0
            self.first_order_backward_sum = 0
            self.first_order_count = 0
            self.second_order_forward_sum = 0
            self.second_order_gen_random_sum = 0
            self.second_order_render_sum = 0
            self.second_order_compute_loss_sum = 0
            self.second_order_time_sum = 0
            self.second_order_backward_setup_sum = 0
            self.second_order_backward_sum = 0
            self.second_order_ls_solve_sum = 0
            self.second_order_update_sum = 0
            self.second_order_count = 0

        if log_output:
            profile_data["frame"] = cur_frame_idx
            profile_data["timestamps"].append(time.time())
            profile_data["pose"] = [viewpoint.T]
            profile_data["gt_pose"] = [viewpoint.T_gt]
            profile_data["exposure"] = [viewpoint.exposure_a, viewpoint.exposure_b]
            profile_data["num_iters"] = tracking_itr
            profile_data["loss_tracking_scalar"] = loss_tracking_scalar.item()

            all_profile_data_obj = self.all_profile_data if first_only else self.all_profile_data_firstonly

            all_profile_data_obj.append(profile_data)

            if (cur_frame_idx + 1) % save_period == 0:
                # Save to self.logdir/run-frame{cur_frame_idx}.pt
                fname = f"run-frame{cur_frame_idx:06d}.pt" if not first_only else f"run-frame{cur_frame_idx:06d}_firstonly.pt"
                if print_output:
                    print(f"Saving profile data to {os.path.join(self.logdir, fname)}")
                torch.save(all_profile_data_obj, os.path.join(self.logdir, fname))
                all_profile_data_obj.clear()

        override = False
        if override_mode == "gt":
            if print_output:
                print("Overriding with GT pose")
            # Set to GT
            viewpoint.T = viewpoint.T_gt.clone()
            viewpoint.cam_rot_delta.data.fill_(0)
            viewpoint.cam_trans_delta.data.fill_(0)
            override = True

        elif override_mode == "none":
            override = False

        if override:
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint
            )

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)

            if print_output:
                print(f"override loss = {loss_tracking_scalar.item():.4f}") #, trans_error = {trans_error.item():.4f}, angle_error = {angle_error.item():.4f}, lambda = {lambda_:.4f}")

        elif not self.use_best_loss:
            trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            if print_output:
                print(f"Last loss = {loss_tracking_scalar.item():.4f}, trans error = {trans_error.item():.4f}")
        else:
            best_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, = best_output
            best_viewpoint_params.assign(viewpoint)
            loss_tracking_scalar = best_loss_scalar
            trans_error, angle_error = relative_pose_error(last_kf.T_gt, viewpoint.T_gt, last_kf.T, viewpoint.T)
            if print_output:
                print(f"Best loss = {best_loss_scalar.item():.4f}, Best trans error = {trans_error.item():.4f}, best angle error = {angle_error.item():.4f}")

        self.median_depth = get_median_depth(depth, opacity)

        if self.run_twice:
            if run_twice_first_rep:
                # If first only, reset viewpoint and run again
                orig_viewpoint.assign(viewpoint)
                return self.tracking(cur_frame_idx, viewpoint, run_twice_first_rep=False, run_twice_override_gt=self.run_twice_override_gt)

        return render_pkg

    def tracking_experiment(self, cur_frame_idx, viewpoint):
        print("Running tracking experiment")
        if self.initialized and cur_frame_idx > self.constant_velocity_warmup and self.monocular:
            prev_prev = self.cameras[cur_frame_idx - self.use_every_n_frames -1 ]
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            
            pose_prev_prev = prev_prev.T
            pose_prev = prev.T
            velocity = pose_prev @ torch.linalg.inv(pose_prev_prev)
            pose_new = velocity @ pose_prev
            viewpoint.T = pose_new
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.T = prev.T

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": self.config["Training"]["lr"]["exposure_a"],
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": self.config["Training"]["lr"]["exposure_b"],
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )
        
        # LM Parameters for convenience
        first_order_max_iter = self.first_order_max_iter
        first_order_fast_iter = self.first_order_fast_iter
        first_order_num_backward_gaussians = self.first_order_num_backward_gaussians
        first_order_num_pixels = self.first_order_num_pixels
        second_order_max_iter = self.second_order_max_iter
        second_order_num_backward_gaussians = self.second_order_num_backward_gaussians
        repeat_dim = self.repeat_dim
        stack_dim = self.stack_dim
        sketch_dim = self.sketch_dim
        initial_lambda = self.initial_lambda
        max_lambda = self.max_lambda
        min_lambda = self.min_lambda
        increase_factor = self.increase_factor
        decrease_factor = self.decrease_factor
        trust_region_cutoff = self.trust_region_cutoff
        second_order_converged_threshold = self.second_order_converged_threshold
        use_nonmonotonic_step = self.use_nonmonotonic_step
        override_mode = self.override_mode
        override_first_logdir = self.override_first_logdir
        print_output = self.print_output
        log_output = self.log_output
        save_period = self.save_period
        # lambda_ = initial_lambda
        lambda_ = 1

        best_loss_scalar = float("inf")
        best_viewpoint_params = None

        default_forward_sketch_args = self.gen_default_forward_sketch_args()

        # warmup_iter = first_order_max_iter
        warmup_iter = 10

        # First run some iterations of first order optimization
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(warmup_iter):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=default_forward_sketch_args,
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint, forward_sketch_args=default_forward_sketch_args,
            )
            loss_tracking_img = HuberLoss.apply(loss_tracking_img, 0.1)

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
            l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
            trans_error, angle_error = relative_pose_error(prev.T_gt, viewpoint.T_gt, prev.T, viewpoint.T)
            print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, l2 loss = {l2_loss.item():.4f}, trans error = {trans_error.item()}, angle error = {angle_error.item()}")

            loss_tracking = torch.norm(loss_tracking_img.flatten(), p=self.pnorm)

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)

            pose_optimizer.zero_grad()
            loss_tracking.backward()

            pose_optimizer.step()

            update_pose(viewpoint)

        if self.use_best_loss:
            print("Using best loss")
            best_viewpoint_params.assign(viewpoint)


        viewpoint.cam_trans_delta.data.fill_(0)
        viewpoint.cam_rot_delta.data.fill_(0)

        check_grad = True
        check_sketch = False
        repeat_second_order = False
        exp_first_order = False

        if check_grad:
            height, width = viewpoint.image_height, viewpoint.image_width
            tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
            exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
            m = height * width
            n = tau_len + exposure_len
            sketch_dim_per_iter = stack_dim * sketch_dim
            d = repeat_dim * stack_dim * sketch_dim

            # Do a default forward
            default_render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=default_forward_sketch_args,
            )

            default_image, default_depth, default_opacity = (
                default_render_pkg["render"],
                default_render_pkg["depth"],
                default_render_pkg["opacity"],
            )

            default_loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, default_image, default_depth, default_opacity, viewpoint, forward_sketch_args=default_forward_sketch_args,
            )
            default_loss_tracking_img = default_loss_tracking_img.sum(dim=0) / (m / sketch_dim_per_iter)

            # Do a sketched forward
            forward_sketch_args = self.gen_forward_sketch_args(height, width, tau_len, exposure_len, repeat_dim, stack_dim, sketch_dim)
            chunk_size = forward_sketch_args["chunk_size"]

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
            )

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
            )
            loss_tracking_img = loss_tracking_img.sum(dim=0) / (m / sketch_dim_per_iter)

            rand_weights = forward_sketch_args["rand_weights"]
            rand_indices_row = forward_sketch_args["rand_indices_row"]
            rand_indices_col = forward_sketch_args["rand_indices_col"]

            weighted_loss_tracking_img = loss_tracking_img * rand_weights
            SJ = torch.empty((repeat_dim, stack_dim, sketch_dim, n), device=self.device)
            batch_indices = torch.arange(repeat_dim).view(-1, 1, 1, 1)
            Sf = weighted_loss_tracking_img[batch_indices, rand_indices_row, rand_indices_col].sum(dim=-1)

            for i in range(repeat_dim):
                print(f"repeat_iter = {i}")

                viewpoint.cam_rot_delta.grad = None
                viewpoint.cam_trans_delta.grad = None
                viewpoint.exposure_a.grad = None
                viewpoint.exposure_b.grad = None
                forward_sketch_args["sketch_dtau"].grad = None
                forward_sketch_args["sketch_dexposure"].grad = None

                pose_optimizer.zero_grad()

                weighted_loss_tracking_img[i].backward(gradient=torch.ones_like(weighted_loss_tracking_img[i]), retain_graph=True)

                SJ_i = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=2)
                SJ[i] = SJ_i

            weighted_default_loss_tracking_img = default_loss_tracking_img * rand_weights
            default_Sf = weighted_default_loss_tracking_img[batch_indices, rand_indices_row, rand_indices_col].sum(dim=-1)

            torch.set_printoptions(precision=10)

            tol = 1e-4

            for i in range(repeat_dim):
                for j in range(stack_dim):
                    for k in range(sketch_dim):
                        print(f"repeat_iter = {i}, stack_iter = {j}, sketch_iter = {k}")
                        weighted_default_loss_tracking_img_ijk = default_Sf[i, j, k]

                        viewpoint.cam_rot_delta.grad = None
                        viewpoint.cam_trans_delta.grad = None
                        viewpoint.exposure_a.grad = None
                        viewpoint.exposure_b.grad = None

                        pose_optimizer.zero_grad()

                        weighted_default_loss_tracking_img_ijk.backward(retain_graph=True)

                        correct_SJ_ijk = torch.cat((viewpoint.cam_trans_delta.grad, viewpoint.cam_rot_delta.grad, viewpoint.exposure_a.grad, viewpoint.exposure_b.grad), dim=0)

                        if not torch.allclose(SJ[i, j, k], correct_SJ_ijk, atol=tol):
                            diff = SJ[i, j, k] - correct_SJ_ijk
                            assert torch.allclose(SJ[i, j, k], correct_SJ_ijk, atol=tol), f"i = {i}, j = {j}, k = {k}, SJ[i, j, k] = {SJ[i, j, k]}, correct_SJ_ijk = {correct_SJ_ijk}, diff = {diff}"

            print("Gradient check passed")

            SJ = SJ.reshape((-1, n))
            Sf = Sf.flatten()

        elif check_sketch:
            prev_avg_SJ = None
            height, width = viewpoint.image_height, viewpoint.image_width
            tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
            exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
            m = height * width
            n = tau_len + exposure_len
            d = int(sketch_aspect * n)

            data = [[] for _ in range(stack_dim)]
            means = [[] for _ in range(stack_dim)]
            ci_lower_bounds = [[] for _ in range(stack_dim)]
            ci_upper_bounds = [[] for _ in range(stack_dim)]
            num_trials = 1000
            confidence = 0.95

            orig_viewpoint = TempCamera(viewpoint)

            for i in range(num_trials):

                # Reset viewpoint
                orig_viewpoint.assign(viewpoint)
                new_viewpoint_params = None
                old_SJ = None
                old_Sf = None
                gen_random = True

                for j in range(stack_dim):

                    # If in second order and new_viewpoint_params is not None
                    # Then cache the old data
                    if new_viewpoint_params is not None:
                        old_loss_scalar = loss_tracking_scalar
                        old_output = (TempCamera(viewpoint), render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, )
                        new_viewpoint_params.assign(viewpoint)
                        update_pose(viewpoint)

                    if gen_random:
                        gen_random = True
                        sketch_mode = 1
                        height, width = viewpoint.image_height, viewpoint.image_width
                        tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
                        exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
                        n = tau_len + exposure_len
                        m = height * width
                        d = int(sketch_aspect * n)

                        # First permute the flattened indices and split them into d parts
                        # Each part must be equal in size so we can stack them into a tensor
                        chunk_size = m // d
                        rand_flat_indices = torch.randperm(m, device=self.device, dtype=torch.int32)
                        rand_indices_row = rand_flat_indices // width
                        rand_indices_col = rand_flat_indices % width
                        rand_indices_row = rand_indices_row.reshape((d, -1))
                        rand_indices_col = rand_indices_col.reshape((d, -1))
                        rand_indices = (rand_indices_row, rand_indices_col)
                        rand_weights = torch.randint(0, 2, size=(height, width), device=self.device, dtype=torch.float32) * 2 - 1

                        sketch_indices = torch.ones((height, width), device=self.device, dtype=torch.int32) * (-1)

                        i_values = torch.arange(d, device=self.device, dtype=torch.int32).view(-1, 1).expand(-1, chunk_size)

                        sketch_indices[rand_indices_row, rand_indices_col] = i_values

                        # This is used to pass into forward functions so we can recover the grad
                        sketch_dtau = torch.empty((d, tau_len), device=self.device, dtype=torch.float32, requires_grad=True)
                        sketch_dexposure = torch.empty((d, exposure_len), device=self.device, dtype=torch.float32, requires_grad=True)

                        forward_sketch_args = {"sketch_mode": sketch_mode, "sketch_dim": d, "sketch_indices": sketch_indices, "rand_indices": rand_indices, "sketch_dtau": sketch_dtau, "sketch_dexposure": sketch_dexposure, }


                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
                    )

                    image, depth, opacity = (
                        render_pkg["render"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                    )

                    loss_tracking_img = get_loss_tracking_per_pixel(
                        self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                    )

                    loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)

                    is_new_step = True
                    second_to_first = False
                    if new_viewpoint_params is not None:
                        # DEBUG
                        is_new_step = False
                        # DEBUG END

                    if not is_new_step:
                        # DEBUG
                        old_viewpoint_params, _, _, _, _, _, _, = old_output
                        # DEBUG END
                        # old_viewpoint_params, _, _, _, _, _, , = old_output
                        old_viewpoint_params.assign(viewpoint)
                        loss_tracking_scalar = old_loss_scalar
                        old_SJ = SJ
                        old_Sf = Sf

                        # We need to run rendering again to align with the new random indices
                        render_pkg = render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
                        )
                        image, depth, opacity = (
                            render_pkg["render"],
                            render_pkg["depth"],
                            render_pkg["opacity"],
                        )
                        loss_tracking_img = get_loss_tracking_per_pixel(
                            self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                        )

                    if torch.isnan(loss_tracking_img).any():
                        raise ValueError("Loss tracking image has nan values")

                    # This is a somewhat arbitrary rescaling, but it works well with the lambdas
                    loss_tracking_img = loss_tracking_img / (m / d)

                    # DEBUG
                    # Manually zero out the gradients for now
                    forward_sketch_args["sketch_dtau"].grad = None
                    forward_sketch_args["sketch_dexposure"].grad = None
                    # print(f"SJ = {forward_sketch_args['sketch_dtau'].grad} {forward_sketch_args['sketch_dexposure'].grad}")
                    # DEBUG END

                    pose_optimizer.zero_grad()
                    loss_tracking_img.backward(grad_tensor=torch.ones_like(loss_tracking_img))
                    exit()

                    with torch.no_grad():

                        SJ = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=1)

                        raw_sigma_max = torch.linalg.matrix_norm(SJ, ord=2)

                        if not is_new_step:
                            # eta = 1
                            SJ = torch.cat((SJ, old_SJ), dim=0)
                            Sf = torch.cat((Sf, old_Sf), dim=0)
                            assert(SJ.shape[0] % d == 0)
                            eta = SJ.shape[0] // d
                        else:
                            eta = 1

                        print(f"eta = {eta}")

                        damped_SJ = torch.cat((SJ / math.sqrt(eta), torch.eye(n, device=self.device) * math.sqrt(lambda_)), dim=0)
                        damped_Sf = torch.cat((Sf / math.sqrt(eta), torch.zeros(n, device=self.device)), dim=0)
                        x = torch.linalg.lstsq(damped_SJ, -damped_Sf).solution

                        new_viewpoint_params = TempCamera(viewpoint)
                        new_viewpoint_params.step(x)
                        second_order_converged = x.norm() < second_order_converged_threshold

                        sigma_max = torch.linalg.matrix_norm(damped_SJ, ord=2)

                    data[j].append(sigma_max.item())
                    # data[j].append(x.norm().item())
                    # data[j].append(damped_Sf.norm().item())
                    mean = np.mean(data[j])
                    std_err = stats.sem(data[j])
                    t_value = stats.t.ppf((1 + confidence) / 2, num_trials - 1)
                    margin_of_error = t_value * std_err
                    ci_lower_bound = mean - margin_of_error
                    ci_upper_bound = mean + margin_of_error
                    means[j].append(mean)
                    ci_lower_bounds[j].append(ci_lower_bound)
                    ci_upper_bounds[j].append(ci_upper_bound)

                    print(f"trial = {i}, j = {j}, sigma_max = {sigma_max:.4f}, mean = {mean:.4f}, ci = ({ci_lower_bound:.4f}, {ci_upper_bound:.4f}), damped_SJ.shape = {damped_SJ.shape}, lambda = {lambda_:.4f}, raw_sigma_max = {raw_sigma_max:.4f}")



        elif repeat_second_order:
            cached_forward_sketch_args = None

            image_gt = viewpoint.original_image.to(self.device)
            image_np = image_gt.permute(1, 2, 0)
            image_np = image_np.detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"outputs/images/frame-{cur_frame_idx}/frame-{cur_frame_idx}_gt.png", image_np)

            for tracking_itr in range(10):
                if cached_forward_sketch_args is None:
                    # Set up sketching in the forward pass
                    sketch_mode = 1
                    height, width = viewpoint.image_height, viewpoint.image_width
                    tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
                    exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
                    n = tau_len + exposure_len
                    m = height * width
                    d = int(sketch_aspect * n)

                    # First permute the flattened indices and split them into d parts
                    # Each part must be equal in size so we can stack them into a tensor
                    chunk_size = m // d
                    rand_flat_indices = torch.randperm(m, device=self.device, dtype=torch.int32)
                    rand_indices_row = rand_flat_indices // width
                    rand_indices_col = rand_flat_indices % width
                    rand_indices_row = rand_indices_row.reshape((d, -1))
                    rand_indices_col = rand_indices_col.reshape((d, -1))
                    rand_indices = (rand_indices_row, rand_indices_col)
                    rand_weights = torch.randint(0, 2, size=(height, width), device=self.device, dtype=torch.float32) * 2 - 1

                    sketch_indices = torch.ones((height, width), device=self.device, dtype=torch.int32) * (-1)

                    i_values = torch.arange(d, device=self.device, dtype=torch.int32).view(-1, 1).expand(-1, chunk_size)

                    sketch_indices[rand_indices_row, rand_indices_col] = i_values

                    # for i in range(d):
                    #     sketch_indices[rand_indices_row[i], rand_indices_col[i]] = i

                    # This is used to pass into forward functions so we can recover the grad
                    sketch_dtau = torch.empty((d, tau_len), device=self.device, dtype=torch.float32, requires_grad=True)
                    sketch_dexposure = torch.empty((d, exposure_len), device=self.device, dtype=torch.float32, requires_grad=True)
                    forward_sketch_args = {"sketch_mode": sketch_mode, "sketch_dim": d, "sketch_indices": sketch_indices, "rand_indices": rand_indices, "sketch_dtau": sketch_dtau, "sketch_dexposure": sketch_dexposure, }
                    cached_forward_sketch_args = forward_sketch_args

                # Then run 1 backward
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )

                loss_tracking_img = get_loss_tracking_per_pixel(
                    self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                )
                
                loss_tracking_img = HuberLoss.apply(loss_tracking_img, 0.1)

                loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
                l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
                trans_error, angle_error = pose_diff(viewpoint.T, viewpoint.T_gt)
                exp_a = viewpoint.exposure_a.item()
                exp_b = viewpoint.exposure_b.item()
                print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, l2 loss = {l2_loss.item():.4f}, trans error = {trans_error.item():.4f}, angle error = {angle_error.item():.4f}")

                loss_tracking_img1 = torch.sum(loss_tracking_img * rand_weights, dim=0)

                Sf = torch.sum(loss_tracking_img1[rand_indices_row, rand_indices_col], dim=1) / chunk_size
                l2_loss_1 = torch.norm(Sf.flatten(), p=2)
                Sf1 = Sf

                phi = torch.sum(loss_tracking_img1) / (m / d)
                
                pose_optimizer.zero_grad()

                # Manually zero out the gradients for now
                forward_sketch_args["sketch_dtau"].grad = None
                forward_sketch_args["sketch_dexposure"].grad = None

                phi.backward()

                SJ = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=1)

                eta = 1

                damped_SJ = torch.cat((SJ, torch.eye(n, device=self.device) * math.sqrt(lambda_)), dim=0)
                damped_Sf = torch.cat((Sf, torch.zeros(n, device=self.device)), dim=0)
                x = torch.linalg.lstsq(damped_SJ, -damped_Sf).solution

                # distortion = math.sqrt(n / (d * eta))
                # sigmas = torch.linalg.svdvals(SJ)
                # min_sigma = sigmas[-1] + math.sqrt(lambda_)
                # residual = torch.norm(Sf - SJ @ x) + math.sqrt(lambda_) * torch.norm(x)
                # upperbound = residual * 2 * distortion * (1 + distortion) / (((1 - distortion) ** 2) * min_sigma)
                # print(f"distortion = {distortion:.4f}, min_sigma = {min_sigma:.4f}, residual = {residual:.4f}, upperbound = {upperbound:.4f}")



                print(f"x = {x}")
                print(f"x norm = {x.norm()}")

                old_viewpoint = TempCamera(viewpoint)

                new_viewpoint_params = TempCamera(viewpoint)
                new_viewpoint_params.step(x)
                new_viewpoint_params.assign(viewpoint)
                print(f"right after assign viewpoint.T = {viewpoint.T}")

                update_pose(viewpoint)
                print(f"right after update viewpoint.T = {viewpoint.T}")

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )


                loss_tracking_img = get_loss_tracking_per_pixel(
                    self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                )
                loss_tracking_img = HuberLoss.apply(loss_tracking_img, 0.1)

                loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
                l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
                trans_error, angle_error = pose_diff(viewpoint.T, viewpoint.T_gt)
                print(f"lambda = {lambda_}, loss = {loss_tracking_scalar.item()}, l2_loss = {l2_loss.item()}, trans error = {trans_error.item()}, angle error = {angle_error.item()}")

                loss_tracking_img1 = torch.sum(loss_tracking_img * rand_weights, dim=0)

                Sf = torch.sum(loss_tracking_img1[rand_indices_row, rand_indices_col], dim=1) / chunk_size
                l2_loss_2 = torch.norm(Sf.flatten(), p=2)

                pred_reduction = (SJ @ x + Sf1).norm() - l2_loss_1
                actual_reduction = l2_loss_2 - l2_loss_1
                rho = actual_reduction / pred_reduction

                print(f"l1 = {l2_loss_1}, l2 = {l2_loss_2}, pred_reduction = {pred_reduction}, actual_reduction = {actual_reduction}, rho = {rho}")

                if rho < 0.25:
                    lambda_ = lambda_ * 2
                elif rho > 0.75:
                    lambda_ = lambda_ / 2
                elif rho < 0:
                    lambda_ = lambda_ * 10

                # save image
                image_np = image.permute(1, 2, 0)
                image_np = image_np.detach().cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                cv2.imwrite(f"outputs/images/frame-{cur_frame_idx}/frame-{cur_frame_idx}_iter-{tracking_itr}_loss-{loss_tracking_scalar.item()}_trans-error-{trans_error.item()}_angle-error-{angle_error.item()}.png", image_np)
                # save opacity.
                opacity_np = opacity.detach().cpu().numpy().squeeze(0)
                opacity_np = (opacity_np * 255).astype(np.uint8)
                cv2.imwrite(f"outputs/images/frame-{cur_frame_idx}/frame-{cur_frame_idx}_iter-{tracking_itr}_opacity.png", opacity_np)


                print("\n\n")

                # DEBUG
                # old_viewpoint.assign(viewpoint)
                # DEBUG END

        elif not exp_first_order:
            # Set up sketching in the forward pass
            sketch_mode = 1
            height, width = viewpoint.image_height, viewpoint.image_width
            tau_len = viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0]
            exposure_len = viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0]
            n = tau_len + exposure_len
            m = height * width
            d = int(sketch_aspect * n)

            # First permute the flattened indices and split them into d parts
            # Each part must be equal in size so we can stack them into a tensor
            chunk_size = m // d
            rand_flat_indices = torch.randperm(m, device=self.device, dtype=torch.int32)
            rand_indices_row = rand_flat_indices // width
            rand_indices_col = rand_flat_indices % width
            rand_indices_row = rand_indices_row.reshape((d, -1))
            rand_indices_col = rand_indices_col.reshape((d, -1))
            rand_indices = (rand_indices_row, rand_indices_col)
            rand_weights = torch.randint(0, 2, size=(height, width), device=self.device, dtype=torch.float32) * 2 - 1

            sketch_indices = torch.ones((height, width), device=self.device, dtype=torch.int32) * (-1)

            i_values = torch.arange(d, device=self.device, dtype=torch.int32).view(-1, 1).expand(-1, chunk_size)

            sketch_indices[rand_indices_row, rand_indices_col] = i_values

            # for i in range(d):
            #     sketch_indices[rand_indices_row[i], rand_indices_col[i]] = i

            # This is used to pass into forward functions so we can recover the grad
            sketch_dtau = torch.empty((d, tau_len), device=self.device, dtype=torch.float32, requires_grad=True)
            sketch_dexposure = torch.empty((d, exposure_len), device=self.device, dtype=torch.float32, requires_grad=True)

            forward_sketch_args = {"sketch_mode": sketch_mode, "sketch_dim": d, "sketch_indices": sketch_indices, "rand_indices": rand_indices, "sketch_dtau": sketch_dtau, "sketch_dexposure": sketch_dexposure, }

            # Then run 1 backward
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
            )

            l2_loss_1 = torch.norm(loss_tracking_img.flatten(), p=2)

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
            print(f"iter = {warmup_iter}, loss = {loss_tracking_scalar.item():.4f}, l2_loss = {l2_loss_1.item():.4f}")

            loss_tracking_img1 = torch.sum(loss_tracking_img * rand_weights, dim=0)

            Sf = torch.sum(loss_tracking_img1[rand_indices_row, rand_indices_col], dim=1) / chunk_size
            l2_loss_1 = torch.norm(Sf, p=2)

            phi = torch.sum(loss_tracking_img1) / (m / d)
            print(f"phi = {phi}")
            
            pose_optimizer.zero_grad()
            phi.backward(retain_graph=True)


            SJ = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=1)

            eta = 1

            old_viewpoint = TempCamera(viewpoint)

            for e in range(10, -5, -1):
                lambda_ = 10**(e)

                damped_SJ = torch.cat((SJ, torch.eye(n, device=self.device) * math.sqrt(lambda_)), dim=0)
                damped_Sf = torch.cat((Sf, torch.zeros(n, device=self.device)), dim=0)
                x = torch.linalg.lstsq(damped_SJ, -damped_Sf).solution

                # distortion = math.sqrt(n / (d * eta))
                # sigmas = torch.linalg.svdvals(SJ)
                # min_sigma = sigmas[-1] + math.sqrt(lambda_)
                # residual = torch.norm(Sf - SJ @ x) + math.sqrt(lambda_) * torch.norm(x)
                # upperbound = residual * 2 * distortion * (1 + distortion) / (((1 - distortion) ** 2) * min_sigma)
                # print(f"distortion = {distortion:.4f}, min_sigma = {min_sigma:.4f}, residual = {residual:.4f}, upperbound = {upperbound:.4f}")

                print(f"x = {x}")

                viewpoint.cam_trans_delta.data += x[:3]
                viewpoint.cam_rot_delta.data += x[3:6]
                viewpoint.exposure_a.data += x[6:7]
                viewpoint.exposure_b.data += x[7:8]

                update_pose(viewpoint)

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )

                loss_tracking_img = get_loss_tracking_per_pixel(
                    self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                )


                loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)

                loss_tracking_img1 = torch.sum(loss_tracking_img * rand_weights, dim=0)
                Sf = torch.sum(loss_tracking_img1[rand_indices_row, rand_indices_col], dim=1) / chunk_size
                l2_loss_2 = torch.norm(Sf, p=2)

                pred_reduction = -(SJ @ x).norm()
                actual_reduction = l2_loss_2 - l2_loss_1

                print(f"lambda = {lambda_}, loss = {loss_tracking_scalar.item()}, l2_loss = {l2_loss_2.item()}")
                print(f"pred_reduction = {pred_reduction}, actual_reduction = {actual_reduction}")
                print("\n\n")

                Sf = torch.sum(loss_tracking_img1[rand_indices_row, rand_indices_col], dim=1) / chunk_size
                phi = torch.sum(loss_tracking_img1) / (m / d)

                old_viewpoint.assign(viewpoint)

        else:
            # Then run 1 backward
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
            )

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
            print(f"iter = {warmup_iter}, loss = {loss_tracking_scalar.item():.4f}")

            loss_tracking = torch.norm(loss_tracking_img.flatten(), p=self.pnorm)

            pose_optimizer.zero_grad()
            loss_tracking.backward()

            x = torch.cat([viewpoint.cam_trans_delta.grad, viewpoint.cam_rot_delta.grad, viewpoint.exposure_a.grad, viewpoint.exposure_b.grad], axis=0)
            print(f"Raw grad: {x}")

            print(f"old viewpoint: {viewpoint.T}\n\n")

            old_viewpoint = TempCamera(viewpoint)

            for e in range(10, 1, -1):
                alpha = -(10**(-e))
                alpha_x = alpha * x
                viewpoint.cam_trans_delta.data += alpha_x[:3]
                viewpoint.cam_rot_delta.data += alpha_x[3:6]
                viewpoint.exposure_a.data += alpha_x[6:7]
                viewpoint.exposure_b.data += alpha_x[7:8]

                tau = torch.cat([viewpoint.cam_trans_delta,
                                 viewpoint.cam_rot_delta], axis=0)
                print(f"tau = {tau}")
                print((tau**2).sum().sqrt().item())

                update_pose(viewpoint)

                print(alpha_x)
                print(viewpoint.T)

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, forward_sketch_args=forward_sketch_args,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )

                loss_tracking_img = get_loss_tracking_per_pixel(
                    self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                )

                loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
                print(f"alpha = {alpha}, loss = {loss_tracking_scalar.item()}\n\n")

                old_viewpoint.assign(viewpoint)

        viewpoint.T.detach_()

        # render_pkg = render(
        #     viewpoint, self.gaussians, self.pipeline_params, self.background
        # )
        # image, depth, opacity = (
        #     render_pkg["render"],
        #     render_pkg["depth"],
        #     render_pkg["opacity"],
        # )

        self.median_depth = get_median_depth(depth, opacity)
        exit()
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = curr_frame.T
        last_kf_CW = last_kf.T
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])

        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(curr_frame.T)

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = kf_i.T
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(kf_j.T)

                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())

                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_T in keyframes:
            self.cameras[kf_id].T = kf_T

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.001)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                
                if cur_frame_idx % 5 == 0:
                    keyframes = [CameraMsg(self.cameras[kf_idx])
                                for kf_idx in self.current_window]
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )
                else:
                    keyframes = [CameraMsg(self.cameras[kf_idx])
                                for kf_idx in self.current_window]
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )                    

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf

                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)

                # DEBUG
                if create_kf:
                    self.last_kf_idx = cur_frame_idx
                # DEBUG END

                cur_frame_idx += 1

                # DEBUG
                save_trj_kf_intv = self.save_trj_kf_intv
                # save_trj_kf_intv = 1
                # print(f"len self.kf_indices = {len(self.kf_indices)}")

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    print("self.cameras ref = ", hex(id(self.cameras)))
                    print("frame 40 ref = ", hex(id(self.cameras[40])))
                    print(f"frame 40 T = {self.cameras[40].T}")
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )

            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
