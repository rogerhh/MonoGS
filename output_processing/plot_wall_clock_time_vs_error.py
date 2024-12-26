import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    args = parser.parse_args()

    logdir = args.logdir

    # Use regex to list files of the form run-frame*.pt
    files = os.listdir(logdir)
    print(files)
    files = [f for f in files if re.match(r"run-frame\d+.pt", f)]

    print(f"Found {len(files)} files")

    all_profile_data = {}

    for fname in files:
        # extract last_frame number
        last_frame = int(re.search(r"frame(\d+).pt", fname).group(1))

        data = torch.load(os.path.join(logdir, fname))

        num_frames = len(data)

        print(f"Loaded {num_frames} frames from {fname}")

        for i in range(num_frames):
            frame = last_frame - num_frames + 1 + i
            assert frame not in all_profile_data
            all_profile_data[frame] = data[i]

    print(all_profile_data.keys())

    # Randomly pick a frame to plot
    frame = np.random.choice(list(all_profile_data.keys()))
    data = all_profile_data[frame]
    timestamps = data["timestamps"]
    timestamps = [t - timestamps[0] for t in timestamps]
    losses = data["losses"]
    print(f"Frame {frame}")
    plt.plot(timestamps[:-1], losses, 'o-')
    plt.show()





