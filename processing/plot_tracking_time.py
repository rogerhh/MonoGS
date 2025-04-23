import os
import sys
import subprocess
import matplotlib.pyplot as plt
import math
import numpy as np

def grep_data(saved_run_path, exp_pattern, data_pattern):
    # Run grep "ate 0" saved_run_path/exp_pattern1 and get output
    result = subprocess.run(f"grep '{data_pattern}' {saved_run_path}/{exp_pattern}/run.log", capture_output=True, shell=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    lines = result.stdout.splitlines()
    data = []

    for line in lines:
        d = float(line.split()[-1])
        data.append(d)

    return data

if __name__ == "__main__":

    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    saved_run_path = os.path.join(project_path, 'saved_runs')

    print(f"Project path: {project_path}")
    print(f"Saved run path: {saved_run_path}")

    exps = [
        ["20250409_1[34]*", "Gradient Descent"],
        ["*f20s5d32*", "2nd Order, d=32"],
        ["*f20s5d64*", "2nd Order, d=64"],
        ["*f20s5d128*", "2nd Order, d=128"]
    ]

    avg_tracking_times = []

    for exp in exps:
        exp_pattern = exp[0]
        exp_name = exp[1]

        avg_tracking_time = grep_data(saved_run_path, exp_pattern, "Average tracking time ms:")

        avg_tracking_times.append(avg_tracking_time)

        print("Average tracking time for", exp_name, ":", np.mean(avg_tracking_time), "ms")






