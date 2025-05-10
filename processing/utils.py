import os
import re
import torch
import numpy as np

def load_data(logdir):
    print(f"Loading data from {logdir}")

    # Use regex to list files of the form run-frame*.pt
    files = os.listdir(logdir)
    # print(files)
    files = [f for f in files if re.match(r"run-frame\d+.pt", f)]

    # print(f"Found {len(files)} files")

    all_profile_data = {}

    for fname in files:
        # extract last_frame number
        last_frame = int(re.search(r"frame(\d+).pt", fname).group(1))

        data = torch.load(os.path.join(logdir, fname))

        num_frames = len(data)

        print(f"Loaded {num_frames} frames from {fname}")

        for i in range(num_frames):
            frame = last_frame - num_frames + 1 + i
            assert (frame not in all_profile_data), f"Frame {frame} already exists in all_profile_data"
            all_profile_data[frame] = data[i]

    # print(all_profile_data.keys())

    return all_profile_data

def load_data_firstonly(logdir):
    print(f"Loading firstonly data from {logdir}")

    # Use regex to list files of the form run-frame*.pt
    files = os.listdir(logdir)
    # print(files)
    files = [f for f in files if re.match(r"run-frame\d+_firstonly.pt", f)]

    # print(f"Found {len(files)} files")

    all_profile_data = {}

    for fname in files:
        # extract last_frame number
        last_frame = int(re.search(r"frame(\d+)_firstonly.pt", fname).group(1))

        data = torch.load(os.path.join(logdir, fname))

        num_frames = len(data)

        print(f"Loaded {num_frames} frames from {fname}")

        for i in range(num_frames):
            frame = last_frame - num_frames + 1 + i
            assert frame not in all_profile_data
            all_profile_data[frame] = data[i]

    # print(all_profile_data.keys())

    return all_profile_data


def pose_difference(T1, T2):
    # Returns the angular difference and translation difference between two SE3 transforms
    # T1 and T2 are 4x4 numpy arrays
    T1 = T1[0].cpu().numpy()
    T2 = T2[0].cpu().numpy()
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]

    tr = (np.trace(R1.T @ R2) - 1) / 2

    print(f"T1: {T1}")
    print(f"T2: {T2}")
    print(f"{np.trace(R1.T @ R2)}, tr: {tr}")

    angle_difference = np.arccos(tr) * 180 / np.pi
    translation_difference = np.linalg.norm(t1 - t2)

    return angle_difference, translation_difference
