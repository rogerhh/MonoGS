import time

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from copy import deepcopy, copy
import math
import os
import json

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gui import gui_utils
from utils.camera_utils import Camera, CameraMsg
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose, angle_diff
from utils.slam_utils import get_loss_tracking, get_loss_tracking_per_pixel, get_median_depth
from processing.utils import load_data

class TempCamera:
    def __init__(self, viewpoint):
        # Copy the viewpoint's relevant data
        self.T = viewpoint.T.detach().clone()
        self.cam_rot_delta = viewpoint.cam_rot_delta.detach().clone()
        self.cam_trans_delta = viewpoint.cam_trans_delta.detach().clone()
        self.exposure_a = viewpoint.exposure_a.detach().clone()
        self.exposure_b = viewpoint.exposure_b.detach().clone()

    def assign(self, viewpoint):
        # viewpoint.T.data.copy_(self.T)
        # viewpoint.cam_rot_delta.data.copy_(self.cam_rot_delta)
        # viewpoint.cam_trans_delta.data.copy_(self.cam_trans_delta)
        # viewpoint.exposure_a.data.copy_(self.exposure_a)
        # viewpoint.exposure_b.data.copy_(self.exposure_b)
        viewpoint.T = self.T
        viewpoint.cam_rot_delta = nn.Parameter(self.cam_rot_delta)
        viewpoint.cam_trans_delta = nn.Parameter(self.cam_trans_delta)
        viewpoint.exposure_a = nn.Parameter(self.exposure_a)
        viewpoint.exposure_b = nn.Parameter(self.exposure_b)

    def step(self, x):
        self.cam_trans_delta.data += x[:3]
        self.cam_rot_delta.data += x[3:6]
        self.exposure_a.data += x[6]
        self.exposure_b.data += x[7]

class SubsamplePixels(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss_tracking_img, num_pixels):
        # Compute loss_tracking_vec1 and dist
        loss_tracking_vec1 = torch.sum(torch.abs(loss_tracking_img), dim=0).flatten() + 1e-8
        with torch.no_grad():
            dist = loss_tracking_vec1 / torch.sum(loss_tracking_vec1)
            selected_indices = torch.multinomial(dist, num_pixels, replacement=True)
        
        # Compute loss_tracking_vec and loss_tracking
        loss_tracking_vec = loss_tracking_vec1 / dist
        loss_tracking = torch.sum(loss_tracking_vec[selected_indices])
        loss_tracking = loss_tracking / num_pixels

        # Save tensors needed for backward
        ctx.save_for_backward(loss_tracking_vec, dist, selected_indices)
        ctx.num_pixels = num_pixels
        ctx.img_shape = loss_tracking_img.shape

        return loss_tracking, selected_indices

    @staticmethod
    def backward(ctx, grad_output, grad_output_indices):
        subsample_start = time.time()
        
        # Retrieve saved tensors and context
        loss_tracking_vec, dist, selected_indices = ctx.saved_tensors
        num_pixels = ctx.num_pixels

        # Compute the gradient for loss_tracking_img
        grad_loss_tracking_vec = torch.zeros_like(loss_tracking_vec)
        grad_loss_tracking_vec[selected_indices] = grad_output / num_pixels

        grad_loss_tracking_vec1 = grad_loss_tracking_vec / dist
        grad_loss_tracking_img = grad_loss_tracking_vec1.view(ctx.img_shape[1:])
        grad_loss_tracking_img = grad_loss_tracking_img.repeat(ctx.img_shape[0], 1, 1)
        print(f"grad_loss_tracking_img.shape = {grad_loss_tracking_img.shape}")

        subsample_end = time.time()
        print(f"Subsample time ms: {(subsample_end - subsample_start) * 1000}")

        # grad for num_pixels is None since it's not differentiable
        return grad_loss_tracking_img, None



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
        self.device = "cuda:0"
        self.pause = False

        # LM Parameters
        self.first_order_max_iter = self.config["Training"]["RGN"]["first_order"]["max_iter"]
        self.first_order_fast_iter = self.config["Training"]["RGN"]["first_order"]["fast_iter"]
        self.first_order_num_backward_gaussians = self.config["Training"]["RGN"]["first_order"]["num_backward_gaussians"]
        self.first_order_num_pixels = self.config["Training"]["RGN"]["first_order"]["num_pixels"]
        self.second_order_max_iter = self.config["Training"]["RGN"]["second_order"]["max_iter"]
        self.second_order_num_backward_gaussians = self.config["Training"]["RGN"]["second_order"]["num_backward_gaussians"]
        self.pnorm = self.config["Training"]["RGN"]["pnorm"]
        self.sketch_aspect = self.config["Training"]["RGN"]["second_order"]["sketch_aspect"]
        self.initial_lambda = self.config["Training"]["RGN"]["second_order"]["initial_lambda"]
        self.max_lambda = self.config["Training"]["RGN"]["second_order"]["max_lambda"]
        self.min_lambda = self.config["Training"]["RGN"]["second_order"]["min_lambda"]
        self.increase_factor = self.config["Training"]["RGN"]["second_order"]["increase_factor"]
        self.decrease_factor = self.config["Training"]["RGN"]["second_order"]["decrease_factor"]
        self.trust_region_cutoff = self.config["Training"]["RGN"]["second_order"]["trust_region_cutoff"]
        self.second_order_converged_threshold = self.config["Training"]["RGN"]["second_order"]["converged_threshold"]
        self.use_nonmonotonic_step = self.config["Training"]["RGN"]["second_order"]["use_nonmonotonic_step"]
        self.use_best_loss = self.config["Training"]["RGN"]["use_best_loss"]
        self.override_mode = self.config["Training"]["RGN"]["override"]["mode"]
        self.override_first_logdir = self.config["Training"]["RGN"]["override"]["first_logdir"]

        if self.override_mode == "first" or self.override_mode == "best":
            self.first_override_data = load_data(self.override_first_logdir)

        self.print_output = self.config["Training"]["RGN"]["print_output"]
        self.log_output = self.config["Training"]["RGN"]["log_output"]
        self.log_basedir = self.config["Training"]["RGN"]["log_basedir"]
        self.log_path = time.strftime("%Y%m%d_%H%M")  # Set logfile base path to be yyyymmdd_HHMM
        self.save_period = self.config["Training"]["RGN"]["save_period"]
        self.experiment_step = self.config["Training"]["experiment_step"]
        
        self.all_profile_data = []

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
        self.first_order_backward_sum = 0
        self.first_order_count = 0
        self.second_order_time_sum = 0
        self.second_order_forward_sum = 0
        self.second_order_gen_random_sum = 0
        self.second_order_setup_sum = 0
        self.second_order_backward_sum = 0
        self.second_order_ls_solve_sum = 0
        self.second_order_update_sum = 0
        self.second_order_count = 0


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
        gt_img = viewpoint.original_image.cuda()
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

    def tracking(self, cur_frame_idx, viewpoint):
        print(f"Frame: {cur_frame_idx}")

        if cur_frame_idx == self.experiment_step:
            return self.tracking_experiment(cur_frame_idx, viewpoint)
            exit()


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
        sketch_aspect = self.sketch_aspect
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

        max_iter = first_order_max_iter + second_order_max_iter
        in_second_order = False
        first_order_countdown = 10
        second_order_countup = 0

        best_loss_scalar = float("inf")
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

        forward_sketch_args = {"sketch_mode": 0, "sketch_dim": 0, "sketch_indices": None, "rand_indices": None, "sketch_dtau": None, "sketch_dexposure": None, }

        profile_data = {"timestamps": [], "losses": [], "pose": None, "rasterize_gaussians_backward_time_ms": [], "rasterize_gaussians_C_backward_time_ms": []}

        tracking_start = time.time()
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(max_iter):
            tracking_iter_start = time.time()

            if log_output:
                profile_data["timestamps"].append(time.time())

            in_second_order = tracking_itr >= first_order_max_iter
            # in_second_order = first_order_countdown <= 0
            # first_order_countdown -= 1

            if tracking_itr == first_order_max_iter:
                if print_output:
                    print("Switching to second order optimization")

                # DEBUG
                # Use current best
                if best_viewpoint_params is not None:
                    best_viewpoint_params.assign(viewpoint)
                # DEBUG END

            # If in second order and new_viewpoint_params is not None
            # Then cache the old data
            if in_second_order and new_viewpoint_params is not None:
                old_loss_scalar = loss_tracking_scalar
                old_output = (TempCamera(viewpoint), render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, )
                new_viewpoint_params.assign(viewpoint)
                update_pose(viewpoint)

                print(f"after swap viewpoint.T = {viewpoint.T}")

            # Set up sketching in the forward pass
            if in_second_order:
                second_order_random_gen_start = time.time()
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

                second_order_random_gen1_end = time.time()
                i_values = torch.arange(d, device=self.device, dtype=torch.int32).view(-1, 1).expand(-1, chunk_size)

                sketch_indices[rand_indices_row, rand_indices_col] = i_values

                # for i in range(d):
                #     sketch_indices[rand_indices_row[i], rand_indices_col[i]] = i

                # This is used to pass into forward functions so we can recover the grad
                sketch_dtau = torch.empty((d, tau_len), device=self.device, dtype=torch.float32, requires_grad=True)
                sketch_dexposure = torch.empty((d, exposure_len), device=self.device, dtype=torch.float32, requires_grad=True)

                forward_sketch_args = {"sketch_mode": sketch_mode, "sketch_dim": d, "sketch_indices": sketch_indices, "rand_indices": rand_indices, "sketch_dtau": sketch_dtau, "sketch_dexposure": sketch_dexposure, }
                second_order_random_gen_end = time.time()


            if tracking_itr >= first_order_fast_iter:
                first_order_num_backward_gaussians = -1

            num_backward_gaussians = first_order_num_backward_gaussians if not in_second_order else second_order_num_backward_gaussians
            num_backward_gaussians = -1 if num_backward_gaussians <= 0 else num_backward_gaussians

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=num_backward_gaussians, forward_sketch_args=forward_sketch_args,
            )

            render_end = time.time()
            # print(f"Render time ms: {(render_end - tracking_iter_start) * 1000}")

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
            )

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=1)
            # trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            # angle_error = angle_diff(viewpoint.R, viewpoint.R_gt)

            if print_output:
                print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}") #, trans_error = {trans_error.item():.4f}, angle_error = {angle_error.item():.4f}, lambda = {lambda_:.4f}")

            if log_output:
                profile_data["losses"].append(loss_tracking_scalar.item())

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)
                best_output = (best_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, )
                # best_trans_error = trans_error
                # best_angle_error = angle_error

            is_new_step = True
            second_to_first = False
            if in_second_order and new_viewpoint_params is not None:
                # if lambda_ <= 1 and loss_tracking_scalar < old_loss_scalar:
                #     second_order_countup += 1

                #     if second_order_countup >= second_order_max_iter:
                #         if print_output:
                #             print("Second order optimization converged")
                #         break

                # # If labmda is a high value but cost is not reduced
                # # revert to using first order method
                # if lambda_ >= 10 and loss_tracking_scalar > old_loss_scalar:
                #     lambda_ = initial_lambda
                #     is_new_step = True
                #     in_second_order = False
                #     first_order_countdown = 3
                #     second_order_countup = 0
                #     second_to_first = True
                #     new_viewpoint_params = None

                # If new step is better than old step, then take it
                if loss_tracking_scalar < old_loss_scalar:
                    lambda_ = max(lambda_ / decrease_factor, min_lambda)
                else:
                    lambda_ = min(lambda_ * increase_factor, max_lambda)
                    # If only allowing strictly decreasing steps, then revert to old viewpoint
                    if not use_nonmonotonic_step:
                        is_new_step = False

                # if lambda_ >= max_lambda:
                #     print(f"Trust region cutoff reached: {lambda_} >= {max_lambda}\nSecond order optimization converged")
                #     break

            if not is_new_step:
                old_viewpoint_params, _, _, _, _, _, _, = old_output
                old_viewpoint_params.assign(viewpoint)
                loss_tracking_scalar = old_loss_scalar
                old_SJ = SJ
                old_Sf = Sf

                # We need to run rendering again to align with the new random indices
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=num_backward_gaussians, forward_sketch_args=forward_sketch_args,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                loss_tracking_img = get_loss_tracking_per_pixel(
                    self.config, image, depth, opacity, viewpoint, forward_sketch_args=forward_sketch_args,
                )

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
                    # loss_tracking, selected_pixel_indices = SubsamplePixels.apply(loss_tracking_img, num_pixels)

                    image._grad_fn.select_pixels = True
                    image._grad_fn.selected_pixel_indices = selected_pixel_indices

                    raise ValueError("Need to redo pixel subsampling to account for correct scaling under different norms")

                # DEBUG END

                else:
                    loss_tracking = torch.norm(loss_tracking_img.flatten(), p=self.pnorm)

                # subsample_end = time.time()

                with torch.no_grad():
                    first_order_backward_start = time.time()
                    pose_optimizer.zero_grad()
                    loss_tracking.backward()

                    first_order_backward_end = time.time()
                    # profile_data["rasterize_gaussians_backward_time_ms"].append(first_order_backward_stats["rasterize_gaussians_backward_time_ms"])
                    # profile_data["rasterize_gaussians_C_backward_time_ms"].append(first_order_backward_stats["rasterize_gaussians_C_backward_time_ms"])

                    pose_optimizer.step()

                    # if second_to_first:
                    #     tau = torch.cat([viewpoint.cam_trans_delta,
                    #                      viewpoint.cam_rot_delta], axis=0)
                    #     print(f"step norm = {(tau**2).sum().sqrt().item():.4f}")
                    #     import code; code.interact(local=locals())

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

                # TEST
                # mplier = 1.1
                # first_order_num_backward_gaussians = int(first_order_num_backward_gaussians * mplier)
                # if first_order_num_backward_gaussians > 10000 or first_order_num_backward_gaussians < 0:
                #     first_order_num_backward_gaussians = -1
                # TEST END

            else:

                if torch.isnan(loss_tracking_img).any():
                    raise ValueError("Loss tracking image has nan values")

                second_order_setup_start = time.time()

                # Sum across channels. Now the dimensions are (h, w)
                # We need to do this because it is more efficient to parallelize across pixels than across channels
                # Absolute values mess up the gradient at 0
                # so to include information from the channels, we weight each channel loss by random +-1
                loss_tracking_img1 = torch.sum(loss_tracking_img * rand_weights, dim=0)

                if self.pnorm == 2:
                    pass
                elif self.pnorm == 1:
                    # Convert to L1 norm
                    loss_tracking_img1 = loss_tracking_img1 / (torch.sqrt(torch.abs(loss_tracking_img1)) + 1e-8)
                    print(loss_tracking_img1)
                    print(torch.isnan(loss_tracking_img1).any())

                Sf = torch.empty(d, device=loss_tracking_img1.device, requires_grad=False)
                Sf = torch.sum(loss_tracking_img1[rand_indices_row, rand_indices_col], dim=1) / chunk_size

                phi = torch.sum(loss_tracking_img1) / (m / d)

                # torch.cuda.synchronize()
                second_order_setup_end = time.time()
                
                pose_optimizer.zero_grad()

                # DEBUG
                # Manually zero out the gradients for now
                forward_sketch_args["sketch_dtau"].grad = None
                forward_sketch_args["sketch_dexposure"].grad = None
                # print(f"SJ = {forward_sketch_args['sketch_dtau'].grad} {forward_sketch_args['sketch_dexposure'].grad}")
                # DEBUG END

                phi.backward(retain_graph=False)

                # torch.cuda.synchronize()
                second_order_backward_end = time.time()

                with torch.no_grad():
                    # torch.cuda.synchronize()
                    second_order_ls_solve_start = time.time()

                    SJ = torch.cat((forward_sketch_args["sketch_dtau"].grad, forward_sketch_args["sketch_dexposure"].grad), dim=1)

                    if not is_new_step:
                        SJ = torch.cat((SJ, old_SJ), dim=0)
                        Sf = torch.cat((Sf, old_Sf), dim=0)
                        assert(SJ.shape[0] % d == 0)
                        eta = SJ.shape[0] // d
                    else:
                        eta = 1


                    damped_SJ = torch.cat((SJ / math.sqrt(eta), torch.eye(n, device=self.device) * math.sqrt(lambda_)), dim=0)
                    damped_Sf = torch.cat((Sf / math.sqrt(eta), torch.zeros(n, device=self.device)), dim=0)
                    x = torch.linalg.lstsq(damped_SJ, -damped_Sf).solution

                    # H = SJ.T @ SJ / eta
                    # g = SJ.T @ Sf / eta
                    # H_damp = H + torch.eye(n, device=self.device) * lambda_
                    # x = torch.linalg.solve(H_damp, -g)

                    second_order_ls_solve_end = time.time()

                    # distortion = math.sqrt(n / (d * eta))
                    # sigmas = torch.linalg.svdvals(SJ)
                    # min_sigma = sigmas[-1] + math.sqrt(lambda_)
                    # residual = torch.norm(Sf - SJ @ x) + math.sqrt(lambda_) * torch.norm(x)
                    # upperbound = residual * 2 * distortion * (1 + distortion) / (((1 - distortion) ** 2) * min_sigma)
                    # print(f"distortion = {distortion:.4f}, min_sigma = {min_sigma:.4f}, residual = {residual:.4f}, upperbound = {upperbound:.4f}")

                    new_viewpoint_params = TempCamera(viewpoint)
                    new_viewpoint_params.step(x)
                    second_order_converged = x.norm() < second_order_converged_threshold

                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
                    print(f"Right after update lambda = {lambda_}, loss = {loss_tracking_scalar.item()}\n\n")
                    print(f"viewpoint.T = {viewpoint.T}")

                    second_order_update_end = time.time()

                    if print_output:
                        print(f"step norm = {x.norm()}, lambda = {lambda_:.4f}")

                    if second_order_converged:
                        if print_output:
                            print(f"step norm {x.norm():.4f} < {second_order_converged_threshold}. Second order optimization converged")
                        break

                second_order_solve_end = time.time()
                # print(f"Forward time ms: {(forward_end - tracking_iter_start) * 1000}")
                # print(f"Second order random gen1 time ms: {(second_order_random_gen1_end - tracking_iter_start) * 1000}")
                # print(f"Second order random gen time ms: {(second_order_random_gen_end - tracking_iter_start) * 1000}")
                # print(f"Second order setup time ms: {(second_order_setup_end - second_order_setup_start) * 1000}")
                # print(f"Second order backward time ms: {(second_order_backward_end - second_order_setup_end) * 1000}")
                # print(f"Second order ls solve time ms: {(second_order_ls_solve_end - second_order_ls_solve_start) * 1000}")
                # print(f"Second order solve time ms: {(second_order_solve_end - second_order_backward_end) * 1000}")



                
            tracking_itr_end = time.time()
            if in_second_order:
                self.second_order_time_sum += tracking_itr_end - tracking_iter_start
                self.second_order_forward_sum += forward_end - tracking_iter_start
                self.second_order_gen_random_sum += second_order_random_gen_end - second_order_random_gen_start
                self.second_order_setup_sum += second_order_setup_end - second_order_setup_start
                self.second_order_backward_sum += second_order_backward_end - second_order_setup_end
                self.second_order_ls_solve_sum += second_order_ls_solve_end - second_order_ls_solve_start
                self.second_order_update_sum += second_order_update_end - second_order_ls_solve_end
                self.second_order_count += 1
            else:
                self.first_order_time_sum += tracking_itr_end - tracking_iter_start
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

            # DEBUG
            # tracking_end = time.time()

            # render_time_ms = (render_end - tracking_iter_start) * 1000
            # compute_loss_time_ms = (compute_loss_end - render_end) * 1000
            # subsample_time_ms = (subsample_end - compute_loss_end) * 1000
            # first_order_backward_time_ms = (first_order_backward_end - subsample_end) * 1000
            # optimizer_step_time_ms = (optimizer_step_end - first_order_backward_end) * 1000
            # update_pose_time_ms = (update_pose_end - optimizer_step_end) * 1000
            # tracking_time_ms = (tracking_end - tracking_iter_start) * 1000

            # print(f"render time ms: {render_time_ms:.4f}, compute loss time ms: {compute_loss_time_ms:.4f}, subsample time ms: {subsample_time_ms:.4f}, first order backward time ms: {first_order_backward_time_ms:.4f}, optimizer step time ms: {optimizer_step_time_ms:.4f}, update pose time ms: {update_pose_time_ms:.4f}, tracking time ms: {tracking_time_ms:.4f}")

            # DEBUG END

        if print_output:
            print("Tracking converged in {} iterations".format(tracking_itr))

        override = False
        if override_mode == "first":
            if print_output:
                print("Overriding with first-order pose with frame {}".format(self.first_override_data[cur_frame_idx]["frame"]))
            viewpoint.R = self.first_override_data[cur_frame_idx]["pose"][0]
            viewpoint.T = self.first_override_data[cur_frame_idx]["pose"][1]
            viewpoint.cam_rot_delta.data.fill_(0)
            viewpoint.cam_trans_delta.data.fill_(0)
            viewpoint.expousre_a = self.first_override_data[cur_frame_idx]["exposure"][0]
            viewpoint.exposure_b = self.first_override_data[cur_frame_idx]["exposure"][1]
            override = True

        elif override_mode == "gt":
            if print_output:
                print("Overriding with GT pose")
            # Set to GT
            viewpoint.T = viewpoint.T_gt.clone()
            viewpoint.cam_rot_delta.data.fill_(0)
            viewpoint.cam_trans_delta.data.fill_(0)
            override = True

        elif override_mode == "best":
            if self.first_override_data[cur_frame_idx]["loss_tracking_scalar"] < best_loss_scalar:
                if print_output:
                    print(f"best loss = {best_loss_scalar:.4f}, override loss = {self.first_override_data[cur_frame_idx]['loss_tracking_scalar']:.4f}")
                    print("Overriding with better first-order pose with frame {}".format(self.first_override_data[cur_frame_idx]["frame"]))
                viewpoint.R = self.first_override_data[cur_frame_idx]["pose"][0]
                viewpoint.T = self.first_override_data[cur_frame_idx]["pose"][1]
                viewpoint.cam_rot_delta.data.fill_(0)
                viewpoint.cam_trans_delta.data.fill_(0)
                viewpoint.expousre_a = self.first_override_data[cur_frame_idx]["exposure"][0]
                viewpoint.exposure_b = self.first_override_data[cur_frame_idx]["exposure"][1]
                override = True
            else:
                override = False
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

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=2)
            # trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            # angle_error = angle_diff(viewpoint.R, viewpoint.R_gt)

            if print_output:
                print(f"override loss = {loss_tracking_scalar.item():.4f}") #, trans_error = {trans_error.item():.4f}, angle_error = {angle_error.item():.4f}, lambda = {lambda_:.4f}")

        elif not self.use_best_loss:
            old_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, = old_output
            old_viewpoint_params.assign(viewpoint)
            loss_tracking_scalar = old_loss_scalar
            trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            if print_output:
                print(f"Best loss = {old_loss_scalar.item():.4f}, trans error = {trans_error.item():.4f}")
        else:
            best_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img, forward_sketch_args, = best_output
            best_viewpoint_params.assign(viewpoint)
            loss_tracking_scalar = best_loss_scalar
            trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            if print_output:
                print(f"Best loss = {best_loss_scalar.item():.4f}, Best trans error = {trans_error.item():.4f}") #, best angle error = {best_angle_error.item():.4f}")

        tracking_end = time.time()
        # print(f"Tracking time ms: {(tracking_end - tracking_start) * 1000}")
        self.tracking_time_sum += tracking_end - tracking_start

        if (cur_frame_idx + 1) % 10 == 0:
            print(f"Average tracking time ms: {(self.tracking_time_sum / 10) * 1000}")
            avg_first_order_time = 0
            avg_second_order_time = 0
            if self.first_order_count > 0:
                avg_first_order_time = self.first_order_time_sum / self.first_order_count
                avg_first_order_backward = self.first_order_backward_sum / self.first_order_count
                print(f"Average first order time ms: {avg_first_order_time * 1000}")
                print(f"Average first order backward time ms: {avg_first_order_backward * 1000}")
            if self.second_order_count > 0:
                avg_second_order_time = self.second_order_time_sum / self.second_order_count
                avg_second_order_forward = self.second_order_forward_sum / self.second_order_count
                avg_second_order_gen_random = self.second_order_gen_random_sum / self.second_order_count
                avg_second_order_setup = self.second_order_setup_sum / self.second_order_count
                avg_second_order_backward = self.second_order_backward_sum / self.second_order_count
                avg_second_order_ls_solve = self.second_order_ls_solve_sum / self.second_order_count
                avg_second_order_update = self.second_order_update_sum / self.second_order_count
                print(f"Average second order forward time ms: {avg_second_order_forward * 1000}")
                print(f"Average second order gen random time ms: {avg_second_order_gen_random * 1000}")
                print(f"Average second order setup time ms: {avg_second_order_setup * 1000}")
                print(f"Average second order backward time ms: {avg_second_order_backward * 1000}")
                print(f"Average second order ls solve time ms: {avg_second_order_ls_solve * 1000}")
                print(f"Average second order update time ms: {avg_second_order_update * 1000}")
                print(f"Average second order time ms: {avg_second_order_time * 1000}")
            projected_tracking_time = (avg_first_order_time * first_order_max_iter + avg_second_order_time * second_order_max_iter)
            print(f"Projected tracking time ms = {(projected_tracking_time) * 1000}")
            self.tracking_time_sum = 0
            self.first_order_time_sum = 0
            self.first_order_backward_sum = 0
            self.first_order_count = 0
            self.second_order_forward_sum = 0
            self.second_order_gen_random_sum = 0
            self.second_order_setup_sum = 0
            self.second_order_time_sum = 0
            self.second_order_backward_sum = 0
            self.second_order_ls_solve_sum = 0
            self.second_order_update_sum = 0
            self.second_order_count = 0

        if log_output:
            profile_data["frame"] = cur_frame_idx
            profile_data["timestamps"].append(time.time())
            profile_data["pose"] = [viewpoint.R, viewpoint.T]
            profile_data["exposure"] = [viewpoint.exposure_a, viewpoint.exposure_b]
            profile_data["num_iters"] = tracking_itr
            profile_data["loss_tracking_scalar"] = loss_tracking_scalar.item()

            self.all_profile_data.append(profile_data)

            if (cur_frame_idx + 1) % save_period == 0:
                # Save to self.logdir/run-frame{cur_frame_idx}.pt
                fname = f"run-frame{cur_frame_idx:06d}.pt"
                if print_output:
                    print(f"Saving profile data to {os.path.join(self.logdir, fname)}")
                torch.save(self.all_profile_data, os.path.join(self.logdir, fname))
                self.all_profile_data = []

        self.median_depth = get_median_depth(depth, opacity)

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
        # sketch_aspect = self.sketch_aspect
        sketch_aspect = 4
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
        lambda_ = initial_lambda

        best_loss_scalar = float("inf")
        best_viewpoint_params = None

        forward_sketch_args = {"sketch_mode": 0, "sketch_dim": 0, "sketch_indices": None, "rand_indices": None, "sketch_dtau": None, "sketch_dexposure": None, }

        warmup_iter = first_order_max_iter

        # First run some iterations of first order optimization
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(warmup_iter):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
            l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
            trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, l2 loss = {l2_loss.item():.4f}, trans error = {trans_error.item():.4f}")

            loss_tracking = torch.norm(loss_tracking_img.flatten(), p=self.pnorm)

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)

            pose_optimizer.zero_grad()
            loss_tracking.backward()

            pose_optimizer.step()

            # update_pose(viewpoint)

        if self.use_best_loss:
            print("Using best loss")
            best_viewpoint_params.assign(viewpoint)


        viewpoint.cam_trans_delta.data.fill_(0)
        viewpoint.cam_rot_delta.data.fill_(0)

        repeat_second_order = True
        exp_first_order = False


        if repeat_second_order:
            cached_forward_sketch_args = None

            for tracking_itr in range(5):
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
                    viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
                l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
                trans_error = (viewpoint.T - viewpoint.T_gt).norm()
                print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, l2 loss = {l2_loss.item():.4f} trans error = {trans_error.item():.4f}")

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
                    viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
                l2_loss = torch.norm(loss_tracking_img.flatten(), p=2)
                trans_error = (viewpoint.T - viewpoint.T_gt).norm()
                print(f"lambda = {lambda_}, loss = {loss_tracking_scalar.item()}, l2_loss = {l2_loss.item()}, trans error = {trans_error.item()}")

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
                viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
                    viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
                viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
                    viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=-1, forward_sketch_args=forward_sketch_args,
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
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
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
