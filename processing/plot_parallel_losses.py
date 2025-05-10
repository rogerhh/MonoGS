import matplotlib
matplotlib.use("TkAgg")  # Use a non-interactive backend

import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import re
from processing.utils import load_data, load_data_firstonly, pose_difference
from processing.consistent_color_order import get_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch_data", action="store_true")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--saveto", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    logdir = args.logdir

    # Use regex to list files of the form run-frame*.pt
    files = os.listdir(logdir)
    print(files)
    files = [f for f in files if re.match(r"run-frame\d+.pt", f)]

    print(f"Found {len(files)} files")

    all_profile_data = load_data(logdir)
    all_profile_data_firstonly = load_data_firstonly(logdir)

    max_frame = len(all_profile_data)

    all_losses = []
    all_losses_firstonly = []

    all_distortions = []
    all_min_sigmas = []
    all_residuals = []
    all_upperbounds = []

    min_losses = []
    angle_diffs = []
    transl_diffs = []
    exposure_a_diffs = []
    exposure_b_diffs = []

    for frame in range(1, max_frame + 1):
        print(f"Frame {frame}")
        losses = np.array(all_profile_data[frame]["losses"])
        losses_firstonly = np.array(all_profile_data_firstonly[frame]["losses"])

        # Normalize losses to firstonly data
        losses = (losses - np.min(losses_firstonly)) / (losses_firstonly[0] - np.min(losses_firstonly))

        losses_firstonly = (losses_firstonly - np.min(losses_firstonly)) / (losses_firstonly[0] - np.min(losses_firstonly))

        all_losses.append(losses)
        all_losses_firstonly.append(losses_firstonly)

        min_losses.append(np.min(losses))
        angle_diff, transl_diff = pose_difference(
            all_profile_data[frame]["pose"], all_profile_data_firstonly[frame]["pose"]
        )
        angle_diffs.append(angle_diff)
        transl_diffs.append(transl_diff)

        exposures = all_profile_data[frame]["exposure"]
        exposures_firstonly = all_profile_data_firstonly[frame]["exposure"]

        exposure_a_diff = (exposures[0] - exposures_firstonly[0]).cpu().detach().numpy()
        exposure_b_diff = (exposures[1] - exposures_firstonly[1]).cpu().detach().numpy()

        exposure_a_diffs.append(exposure_a_diff)
        exposure_b_diffs.append(exposure_b_diff)

        max_iter = len(losses)
        second_order_iter = len(all_profile_data[frame]["upperbounds"])

        import code; code.interact(local=locals())

        distortions = np.zeros(max_iter)
        distortions[-second_order_iter:] = all_profile_data[frame]["distortions"]
        all_distortions.append(distortions)

        min_sigmas = np.zeros(max_iter)
        min_sigmas[-second_order_iter:] = all_profile_data[frame]["min_sigmas"]
        all_min_sigmas.append(min_sigmas)

        residuals = np.zeros(max_iter)
        residuals[-second_order_iter:] = all_profile_data[frame]["residuals"]
        all_residuals.append(residuals)

        upperbounds = np.zeros(max_iter)
        upperbounds[-second_order_iter:] = all_profile_data[frame]["upperbounds"]
        all_upperbounds.append(upperbounds)

        print(len(losses))

    # First convert lists to 2d arrays
    all_losses = np.array(all_losses)
    all_distortions = np.array(all_distortions)
    all_gammas = (1 + all_distortions) / (1 - all_distortions)
    all_min_sigmas = np.array(all_min_sigmas)
    all_residuals = np.array(all_residuals)
    all_upperbounds = np.array(all_upperbounds)

    percentile_losses = np.percentile(all_losses, 90, axis=0)
    percentile_distortions = np.percentile(all_distortions, 90, axis=0)
    percentile_min_sigmas = np.percentile(all_min_sigmas, 10, axis=0)
    percentile_residuals = np.percentile(all_residuals, 90, axis=0)

    # percentile_gamma = (1 + percentile_distortions) / (1 - percentile_distortions)
    # percentile_upperbounds = percentile_residuals * percentile_gamma * np.sqrt(percentile_gamma ** 2 - 1) / percentile_min_sigmas
    # import code; code.interact(local=locals())

    percentile_upperbounds = np.percentile(all_upperbounds, 90, axis=0)
    percentile_upperbounds_low = np.percentile(all_upperbounds, 10, axis=0)

    avg_upperbounds = np.mean(all_upperbounds[:,-2:], axis=1)

    angle_diffs = np.array(angle_diffs)
    transl_diffs = np.array(transl_diffs)

    sorted_indices = np.argsort(transl_diffs)[::-1]

    print(f"Sorted indices: {sorted_indices + 1}")

    n = 20
    sorted_indices = sorted_indices[:n]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, dpi=200, figsize=(12, 8))
    ax1.plot(range(1, len(angle_diffs) + 1), angle_diffs, "-")
    ax1.set_title("Angle Difference")
    ax2.plot(range(1, len(transl_diffs) + 1), transl_diffs, "-")
    ax2.set_title("Translation Difference")
    ax2.plot(sorted_indices + 1, transl_diffs[sorted_indices], "ro")
    ax3.plot(range(1, len(exposure_a_diffs) + 1), exposure_a_diffs, "-")
    ax3.set_title("Exposure A Difference")
    ax4.plot(range(1, len(exposure_b_diffs) + 1), exposure_b_diffs, "-")
    ax4.set_title("Exposure B Difference")
    ax1.set_ylim(0, 4)
    ax2.set_ylim(0, 0.08)

    # Plot the losses on two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=200, figsize=(12, 8)) 
    color_idx = 0
    for frame in range(1, max_frame + 1):
        losses = all_losses[frame - 1]
        losses_firstonly = all_losses_firstonly[frame - 1]

        if frame in sorted_indices[:n] + 1:
            ax1.plot(range(1, len(losses) + 1), losses, '-', label=f"Frame {frame}", color=get_color(color_idx))
            ax2.plot(range(1, len(losses_firstonly) + 1), losses_firstonly, '-', label=f"Frame {frame}", color=get_color(color_idx))
            color_idx += 1
        else:
            pass
            # ax1.plot(range(1, len(losses) + 1), losses, '-')
            # ax2.plot(range(1, len(losses_firstonly) + 1), losses_firstonly, '-')

    ax1.plot(range(1, len(percentile_losses) + 1), percentile_losses, "k--", label="90th Percentile")
    ax1.legend()

    # Plot distortions, min sigmas, residuals, and upperbounds
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, dpi=200, figsize=(12, 8))
    color_idx = 0
    for frame in range(1, max_frame + 1):
        distortions = all_distortions[frame - 1]
        min_sigmas = all_min_sigmas[frame - 1]
        residuals = all_residuals[frame - 1]
        upperbounds = all_upperbounds[frame - 1]

        if frame in sorted_indices[:n] + 1:
            ax1.plot(range(1, len(distortions) + 1), distortions, '-', label=f"Frame {frame}", color=get_color(color_idx))
            ax2.plot(range(1, len(min_sigmas) + 1), min_sigmas, '-', label=f"Frame {frame}", color=get_color(color_idx))
            ax3.plot(range(1, len(residuals) + 1), residuals, '-', label=f"Frame {frame}", color=get_color(color_idx))
            ax4.plot(range(1, len(upperbounds) + 1), upperbounds, '-', label=f"Frame {frame}", color=get_color(color_idx))
            color_idx += 1
        else:
            pass
            # ax.plot(range(1, len(upperbounds) + 1), upperbounds, '-')
            # ax2.plot(range(1, len(min_sigmas) + 1), min_sigmas, '-', label=f"Frame {frame}")
            # ax3.plot(range(1, len(residuals) + 1), residuals, '-', label=f"Frame {frame}")
            # ax4.plot(range(1, len(upperbounds) + 1), upperbounds, '-', label=f"Frame {frame}")
    ax1.plot(range(1, len(percentile_distortions) + 1), percentile_distortions, "k--", label="90th Percentile")
    ax1.set_ylim(0, 0.4)
    ax2.plot(range(1, len(percentile_min_sigmas) + 1), percentile_min_sigmas, "k--", label="90th Percentile")
    ax2.set_ylim(0, 0.15)
    ax3.plot(range(1, len(percentile_residuals) + 1), percentile_residuals, "k--", label="90th Percentile")
    ax3.set_ylim(0, 0.08)
    ax4.plot(range(1, len(percentile_upperbounds) + 1), percentile_upperbounds, "k--", label="90th Percentile")
    ax4.plot(range(1, len(percentile_upperbounds_low) + 1), percentile_upperbounds_low, "k--", label="10th Percentile")
    ax4.set_ylim(0, 1)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    # Plot correlation of avg upperbound and trans and angle diff in scatter plot
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=200, figsize=(12, 8))
    ax1.scatter(avg_upperbounds, transl_diffs, label="Translation Difference")
    ax1.set_title("Avg Upperbound vs Translation Difference")
    ax2.scatter(avg_upperbounds, angle_diffs, label="Angle Difference")
    ax2.set_title("Avg Upperbound vs Angle Difference")
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("Avg Upperbound")
    ax2.set_xlabel("Avg Upperbound")
    ax1.set_ylabel("Translation Difference")
    ax2.set_ylabel("Angle Difference")


    if args.saveto is not None:
        print(f"Saving plot to {args.saveto}")
        plt.savefig(args.saveto, dpi=200)

    if args.plot:
        plt.show()

