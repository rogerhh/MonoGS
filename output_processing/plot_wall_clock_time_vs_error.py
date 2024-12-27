import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import re
from output_processing.utils import load_data

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

    all_profile_data = load_data(logdir)

    # Randomly pick a frame to plot
    frame = np.random.choice(list(all_profile_data.keys()))
    data = all_profile_data[frame]
    timestamps = data["timestamps"]
    timestamps = [t - timestamps[0] for t in timestamps]
    losses = data["losses"]
    print(f"Frame {frame}")
    plt.plot(timestamps[:-1], losses, 'o-')
    plt.show()





