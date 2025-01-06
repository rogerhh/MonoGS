import os
import re
import torch

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
            assert frame not in all_profile_data
            all_profile_data[frame] = data[i]

    # print(all_profile_data.keys())

    return all_profile_data

