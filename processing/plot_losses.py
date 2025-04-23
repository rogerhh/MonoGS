import matplotlib
matplotlib.use("TkAgg")  # Use a non-interactive backend

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

    max_frame = len(all_profile_data)

    fig, ax = plt.subplots(dpi=200, figsize=(12, 8))

    all_losses = []
    step20_normalized_losses = []

    for frame in range(1, max_frame + 1):
        losses = np.array(all_profile_data[frame]["losses"])
        
        # Normalize losses
        losses = losses - np.min(losses)
        losses = losses / losses[0]

        all_losses.append(losses)
        step20_normalized_losses.append(losses[20])

    # argsort step20 normalized losses from high to low
    sorted_indices = np.argsort(step20_normalized_losses)[::-1] + 1
    
    # Set all indices that is less than 40 to -1
    sorted_indices[sorted_indices < 40] = -1

    sorted_indices = list(sorted_indices)
    # Remove -1 from sorted_indices
    sorted_indices = [i for i in sorted_indices if i != -1]

    print(f"Sorted indices: {sorted_indices}")

    for frame in range(1, max_frame + 1):
        losses = all_losses[frame - 1]

        if frame < 100:
            continue

        if frame in sorted_indices[:10]:
            ax.plot(range(len(losses)), losses, '-', label=f"Frame {frame}")
        else:
            ax.plot(range(len(losses)), losses, '-')

    ax.legend(fontsize=20)

    if args.saveto is not None:
        print(f"Saving plot to {args.saveto}")
        plt.savefig(args.saveto, dpi=200)

    if args.plot:
        plt.show()

