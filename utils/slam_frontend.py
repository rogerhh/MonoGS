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
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose, angle_diff
from utils.slam_utils import get_loss_tracking, get_loss_tracking_per_pixel, get_median_depth
from processing.utils import load_data

class TempCamera:
    def __init__(self, viewpoint):
        # Copy the viewpoint's relevant data
        self.R = viewpoint.R.detach().clone()
        self.T = viewpoint.T.detach().clone()
        self.cam_rot_delta = viewpoint.cam_rot_delta.detach().clone()
        self.cam_trans_delta = viewpoint.cam_trans_delta.detach().clone()
        self.exposure_a = viewpoint.exposure_a.detach().clone()
        self.exposure_b = viewpoint.exposure_b.detach().clone()

    def assign(self, viewpoint):
        viewpoint.R = self.R
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
        self.sketch_aspect = self.config["Training"]["RGN"]["second_order"]["sketch_aspect"]
        self.initial_lambda = self.config["Training"]["RGN"]["second_order"]["initial_lambda"]
        self.max_lambda = self.config["Training"]["RGN"]["second_order"]["max_lambda"]
        self.min_lambda = self.config["Training"]["RGN"]["second_order"]["min_lambda"]
        self.increase_factor = self.config["Training"]["RGN"]["second_order"]["increase_factor"]
        self.decrease_factor = self.config["Training"]["RGN"]["second_order"]["decrease_factor"]
        self.trust_region_cutoff = self.config["Training"]["RGN"]["second_order"]["trust_region_cutoff"]
        self.second_order_converged_threshold = self.config["Training"]["RGN"]["second_order"]["converged_threshold"]
        self.use_nonmonotonic_step = self.config["Training"]["RGN"]["second_order"]["use_nonmonotonic_step"]
        self.override_mode = self.config["Training"]["RGN"]["override"]["mode"]
        self.override_first_logdir = self.config["Training"]["RGN"]["override"]["first_logdir"]
        self.override = (self.override_mode != "none")

        if self.override_mode == "first":
            self.first_override_data = load_data(self.override_first_logdir)

        self.print_output = self.config["Training"]["RGN"]["print_output"]
        self.log_output = self.config["Training"]["RGN"]["log_output"]
        self.log_basedir = self.config["Training"]["RGN"]["log_basedir"]
        self.log_path = time.strftime("%Y%m%d_%H%M")  # Set logfile base path to be yyyymmdd_HHMM
        self.save_period = self.config["Training"]["RGN"]["save_period"]
        self.all_profile_data = []

        if self.log_output:
            # Check if log_basedir/log_path exists
            self.logdir = os.path.join(self.log_basedir, self.log_path)
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            # Dump the config to the logdir as a json file
            with open(os.path.join(self.logdir, "config.json"), "w") as f:
                json.dump(self.config, f)


    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

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
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        print(f"Frame: {cur_frame_idx}")

        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

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
        override = self.override
        override_mode = self.override_mode
        override_first_logdir = self.override_first_logdir
        print_output = self.print_output
        log_output = self.log_output
        save_period = self.save_period

        max_iter = first_order_max_iter + second_order_max_iter
        in_second_order = False

        best_loss_scalar = float("inf")
        best_output = None
        best_viewpoint_params = None
        best_trans_error = float("inf")
        best_angle_error = float("inf")
        lambda_ = initial_lambda
        new_viewpoint_params = None
        compute_new_hessian = True
        H = None
        g = None

        profile_data = {"timestamps": [], "losses": [], "pose": None}

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(max_iter):
            if log_output:
                profile_data["timestamps"].append(time.time())

            in_second_order = tracking_itr >= first_order_max_iter

            if tracking_itr == first_order_max_iter:
                print("Switching to second order optimization")

            # If in second order and new_viewpoint_params is not None
            # Then cache the old data
            if in_second_order and new_viewpoint_params is not None:
                old_loss_scalar = loss_tracking_scalar
                old_output = (TempCamera(viewpoint), render_pkg, image, depth, opacity, loss_tracking_img,)
                new_viewpoint_params.assign(viewpoint)
                update_pose(viewpoint)

            if tracking_itr >= first_order_fast_iter:
                first_order_num_backward_gaussians = -1

            num_backward_gaussians = first_order_num_backward_gaussians if not in_second_order else second_order_num_backward_gaussians
            num_backward_gaussians = -1 if num_backward_gaussians <= 0 else num_backward_gaussians

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, num_backward_gaussians=num_backward_gaussians
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            # print(f"image grad_fn = {image._grad_fn}")
            # image._grad_fn.sampled_pixel_indices = torch.tensor([0, 1, 2, 3, 5], device=image.device)
            loss_tracking_img = get_loss_tracking_per_pixel(
                self.config, image, depth, opacity, viewpoint
            )

            loss_tracking_scalar = torch.norm(loss_tracking_img.flatten(), p=2)
            trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            angle_error = angle_diff(viewpoint.R, viewpoint.R_gt)

            if print_output:
                print(f"iter = {tracking_itr}, loss = {loss_tracking_scalar.item():.4f}, trans_error = {trans_error.item():.4f}, angle_error = {angle_error.item():.4f}, lambda = {lambda_:.4f}")

            if log_output:
                profile_data["losses"].append(loss_tracking_scalar.item())

            if loss_tracking_scalar < best_loss_scalar:
                best_loss_scalar = loss_tracking_scalar
                best_viewpoint_params = TempCamera(viewpoint)
                best_output = (best_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img,)
                best_trans_error = trans_error
                best_angle_error = angle_error

            compute_new_hessian = True
            if in_second_order and new_viewpoint_params is not None:

                # If new step is better than old step, then take it
                if loss_tracking_scalar < old_loss_scalar:
                    compute_new_hessian = True
                    lambda_ = max(lambda_ / decrease_factor, min_lambda)
                else:
                    lambda_ = min(lambda_ * increase_factor, max_lambda)
                    # If only allowing strictly decreasing steps, then revert to old viewpoint
                    if not use_nonmonotonic_step:
                        old_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img = old_output
                        old_viewpoint_params.assign(viewpoint)
                        loss_tracking_scalar = old_loss_scalar
                        # Having new random mixture is very important!!
                        compute_new_hessian = False
                        # DEBUG
                        compute_new_hessian = True
                        # DEBUG END
                if lambda_ >= max_lambda:
                    print(f"Trust region cutoff reached: {lambda_} >= {max_lambda}\nSecond order optimization converged")
                    break

            if not in_second_order:
                # Get l1 norm of loss_tracking
                # DEBUG
                num_pixels = first_order_num_pixels
                if num_pixels > 0:
                    # total_num_tiles = (loss_tracking_img.shape[1] // 16) * (loss_tracking_img.shape[2 // 16])
                    # loss_tracking_vec = torch.abs(loss_tracking_img.flatten())
                    # dist = loss_tracking_vec / torch.sum(loss_tracking_vec) + 1e-8
                    # selected_indices = torch.multinomial(dist, num_pixels, replacement=False)
                    # loss_tracking_vec = loss_tracking_vec / dist
                    # loss_tracking = torch.sum(loss_tracking_vec[selected_indices])
                    # loss_tracking = loss_tracking / num_pixels
                    # Need to first sum across channels
                    loss_tracking_vec1 = torch.sum(torch.abs(loss_tracking_img), dim=0).flatten() + 1e-8
                    with torch.no_grad():
                        dist = loss_tracking_vec1 / torch.sum(loss_tracking_vec1)
                        selected_indices = torch.multinomial(dist, num_pixels, replacement=False)
                    loss_tracking_vec = loss_tracking_vec1 / dist
                    loss_tracking = torch.sum(loss_tracking_vec[selected_indices])
                    loss_tracking = loss_tracking / num_pixels

                    image._grad_fn.select_pixels = False
                    image._grad_fn.selected_pixel_indices = selected_indices
                    print("we should not be here either")

                # DEBUG END

                else:
                    loss_tracking = torch.norm(loss_tracking_img.flatten(), p=1)

                with torch.no_grad():
                    pose_optimizer.zero_grad()
                    loss_tracking.backward()
                    # import code; code.interact(local=locals())
                    # print("loss_tracking_vec1.grad = ", loss_tracking_vec1.grad)
                    # print("exiting")
                    # exit()

                    # DEBUG
                    grad = torch.cat([viewpoint.cam_trans_delta.grad, viewpoint.cam_rot_delta.grad, viewpoint.exposure_a.grad, viewpoint.exposure_b.grad])
                    # DEBUG END

                    pose_optimizer.step()
                    first_order_converged = update_pose(viewpoint)

                if first_order_converged:
                    print("First order optimization converged")
                    break

                # TEST
                # mplier = 1.1
                # first_order_num_backward_gaussians = int(first_order_num_backward_gaussians * mplier)
                # if first_order_num_backward_gaussians > 10000 or first_order_num_backward_gaussians < 0:
                #     first_order_num_backward_gaussians = -1
                # TEST END

            else:
                loss_tracking_vec = loss_tracking_img.flatten()

                # check if loss_tracking_vec has nan values
                if torch.isnan(loss_tracking_vec).any():
                    raise ValueError("Loss tracking vector has nan values")

                n = (viewpoint.cam_trans_delta.shape[0] + viewpoint.cam_rot_delta.shape[0] + viewpoint.exposure_a.shape[0] + viewpoint.exposure_b.shape[0])
                m = loss_tracking_vec.shape[0]
                d = sketch_aspect * n

                # Only do backward passes if computing new hessian
                if compute_new_hessian:
                    # Instead of sampling indices, shuffle the indices then split them into d parts
                    sketch_indices = torch.randperm(m)
                    sketch_indices = torch.split(sketch_indices, m // d)

                    J = torch.empty((d, n), device=loss_tracking_vec.device, requires_grad=False)
                    f = torch.empty(d, device=loss_tracking_vec.device, requires_grad=False)

                    # Generate a vector (m, ) of either 1 or -1
                    rand_weights = torch.randint(0, 2, (m,), device=loss_tracking_vec.device) * 2 - 1
                    loss_tracking_vec = loss_tracking_vec * rand_weights

                    for i in range(d):
                        indices = sketch_indices[i]
                        sketch_len = len(indices)
                        # This (m / d) is some weird normalization factor. Need this so that lambda is some sane relative value
                        loss_i = torch.sum(loss_tracking_vec[indices]) / (m / d)

                        with torch.no_grad():
                            # Reset grad first
                            pose_optimizer.zero_grad()
                            loss_i.backward(retain_graph=True)

                            J[i, :] = torch.cat([viewpoint.cam_trans_delta.grad, viewpoint.cam_rot_delta.grad, viewpoint.exposure_a.grad, viewpoint.exposure_b.grad])
                            f[i] = loss_i


                    with torch.no_grad():
                        H = J.T @ J
                        g = J.T @ f

                with torch.no_grad():

                    H_damp = H + torch.eye(n, device=H.device) * lambda_
                    x = torch.linalg.solve(H_damp, -g)

                    new_viewpoint_params = TempCamera(viewpoint)
                    new_viewpoint_params.step(x)
                    second_order_converged = x.norm() < second_order_converged_threshold

                    # # DEBUG
                    # # Compute the minimum singular value of J
                    # SJ = J * (m / d)
                    # Sf = f * (m / d)
                    # _, s, _ = torch.svd(SJ)
                    # min_s = torch.min(s)
                    # # The residual is SJx - Sf
                    # Sr = SJ @ x - Sf

                    # print("min_s = ", min_s)
                    # print("Sr.norm() = ", Sr.norm())
                    # # DEBUG END

                    if second_order_converged:
                        print("Second order optimization converged")
                        break



            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )

        print("Tracking converged in {} iterations".format(tracking_itr))

        if override:
            if override_mode == "first":
                print("Overriding with first-order pose with frame {}".format(self.first_override_data[cur_frame_idx]["frame"]))
                viewpoint.R = self.first_override_data[cur_frame_idx]["pose"][0]
                viewpoint.T = self.first_override_data[cur_frame_idx]["pose"][1]
                viewpoint.cam_rot_delta.data.fill_(0)
                viewpoint.cam_trans_delta.data.fill_(0)
                viewpoint.expousre_a = self.first_override_data[cur_frame_idx]["exposure"][0]
                viewpoint.exposure_b = self.first_override_data[cur_frame_idx]["exposure"][1]

            elif override_mode == "gt":
                print("Overriding with GT pose")
                # Set to GT
                viewpoint.R = viewpoint.R_gt.clone()
                viewpoint.T = viewpoint.T_gt.clone()
                viewpoint.cam_rot_delta.data.fill_(0)
                viewpoint.cam_trans_delta.data.fill_(0)

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
            trans_error = (viewpoint.T - viewpoint.T_gt).norm()
            angle_error = angle_diff(viewpoint.R, viewpoint.R_gt)

            print(f"override loss = {loss_tracking_scalar.item():.4f}, trans_error = {trans_error.item():.4f}, angle_error = {angle_error.item():.4f}, lambda = {lambda_:.4f}")

        else:
            best_viewpoint_params, render_pkg, image, depth, opacity, loss_tracking_img = best_output
            best_viewpoint_params.assign(viewpoint)
            print(f"Best loss = {best_loss_scalar.item():.4f}, Best trans error = {best_trans_error.item():.4f}, best angle error = {best_angle_error.item():.4f}")

        if log_output:
            profile_data["frame"] = cur_frame_idx
            profile_data["timestamps"].append(time.time())
            profile_data["pose"] = [viewpoint.R, viewpoint.T]
            profile_data["exposure"] = [viewpoint.exposure_a, viewpoint.exposure_b]
            profile_data["num_iters"] = tracking_itr

            self.all_profile_data.append(profile_data)

            if (cur_frame_idx + 1) % save_period == 0:
                # Save to self.logdir/run-frame{cur_frame_idx}.pt
                fname = f"run-frame{cur_frame_idx:06d}.pt"
                print(f"Saving profile data to {os.path.join(self.logdir, fname)}")
                torch.save(self.all_profile_data, os.path.join(self.logdir, fname))
                self.all_profile_data = []

        self.median_depth = get_median_depth(depth, opacity)
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
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
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
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
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

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
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

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

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
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

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
                tic.record()
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
                    time.sleep(0.01)
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
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
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
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
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
