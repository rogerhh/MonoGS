import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import re
from processing.utils import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--logdir_ref", type=str, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    logdir = args.logdir

    # Use regex to list files of the form run-frame*.pt
    files = os.listdir(logdir)
    print(files)
    files = [f for f in files if re.match(r"run-frame\d+.pt", f)]

    print(f"Found {len(files)} files")

    all_profile_data = load_data(logdir)

    all_ref_data = load_data(args.logdir_ref)

    # Randomly pick a frame to plot
    if args.frame == -1:
        frame = np.random.choice(list(all_profile_data.keys()))
    else:
        frame = args.frame
    data = all_profile_data[frame]
    timestamps = data["timestamps"]
    timestamps = [t - timestamps[0] for t in timestamps]
    losses = data["losses"]
    print(f"Frame {frame}")

    ref_data = all_ref_data[frame]
    ref_timestamps = ref_data["timestamps"]
    ref_timestamps = [t - ref_timestamps[0] for t in ref_timestamps]
    ref_losses = ref_data["losses"]
    print(f"Frame {frame}")

    plt.figure(dpi=200, figsize=(12, 8))
    
    # Split figure into 2 subplots
    ax1 = plt.subplot(211)
    ax1.plot(range(len(losses)), losses, 'o-')
    ax1.plot(range(len(ref_losses)), ref_losses, 'o-')

    ax1.set_ylabel("Loss", fontsize=20)
    ax1.set_xlabel("Iteration", fontsize=20)
    ax1.legend(["Adam", "2nd-order Method"], fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2 = plt.subplot(212)

    ax2.plot(timestamps[:-1], losses, 'o-')
    ax2.plot(ref_timestamps[:-1], ref_losses, 'o-')
    ax2.set_ylabel("Loss", fontsize=20)
    ax2.set_xlabel("Time (s)", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    plt.subplots_adjust(hspace=0.30)


    if args.output is not None:
        plt.savefig(args.output, bbox_inches="tight")

    if args.plot:
        plt.show()





