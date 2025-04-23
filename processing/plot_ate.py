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

    print(f"Output: {result.stdout}")

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

    ates = []
    fpss = []

    for exp in exps:
        exp_pattern = exp[0]
        exp_name = exp[1]

        # Get ATE and FPS
        ate = grep_data(saved_run_path, exp_pattern, "ate 0")
        fps = grep_data(saved_run_path, exp_pattern, "FPS")

        ates.append(ate)
        fpss.append(fps)

    plt.rcParams.update({'font.size': 20})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 12), sharex=True)

    for i, exp in enumerate(exps):
        print("Plotting", i, np.mean(ates[i]), np.std(ates[i]), np.mean(fpss[i]), np.std(fpss[i]))
        ax1.scatter([i] * len(ates[i]), ates[i], label=exp[1])
        ax2.scatter([i] * len(fpss[i]), fpss[i], label=exp[1])


    ax1.set_ylabel("ATE (m)")
    ax2.set_ylabel("FPS")

    ax2.tick_params(axis="x", rotation=45)
    ax2.set_xticks(range(len(exps)), [exp[1] for exp in exps])
    ax2.set_xticklabels([exp[1] for exp in exps])
    plt.subplots_adjust(bottom=0.4)


    # plt.show()



