from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 20})

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to plot")

    frame = parser.parse_args().frame

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7), dpi=700)

    gn_data = []
    sgd_data = []
    adam_data = []

    fnames = ["GN", "Adam", "SGD", "rk_block-size-2", "rek_block-size-2"]
    names = ["Gauss-Newton", "Adam", "SGD", "RK", "REK"]
    for name, fname, in zip(names, fnames):
        filename = f"experiment/data/frame{frame:06d}_outer_losses_{fname}.npy"
        data = np.load(filename, allow_pickle=True)

        outer_losses = data

        # Plot inner losses and errors
        ax1.plot(outer_losses, label=name)

        if fname == "GN":
            gn_data = data
        elif fname == "SGD":
            sgd_data = data
        elif fname == "Adam":
            adam_data = data

    ax1.set_xlim([0, 100])

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend()
    fig.tight_layout()
    plt.savefig(f"experiment/data/frame{frame:06d}_outer_losses1.png")
    plt.savefig(f"experiment/data/frame{frame:06d}_outer_losses1.pdf")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7), dpi=700)

    ax1.plot(gn_data, label="Gauss-Newton")

    project_modes = ["gaussian", "sparse-count-sketch", "very-sparse-count-sketch"]
    project_mode_names = ["Gaussian Projection", "Count Sketch (c=8)", "Count Sketch (c=1)"]

    for project_mode, project_mode_name in zip(project_modes, project_mode_names):
        for d in [64, 128]:
            filename = f"experiment/data/frame{frame:06d}_outer_losses_project_{project_mode}_d-{d}.npy"
            data = np.load(filename, allow_pickle=True)

            outer_losses = data

            name = f"{project_mode_name} (d={d})"

            # Plot inner losses and errors
            ax1.plot(outer_losses, label=name)

    sample_modes = ["uniform", "row-norm-squared", "leverage-score"]
    sample_mode_names = ["Uniform Sampling", "Row Norm Squared Sampling", "Leverage Score Sampling"]

    ax1.set_xlim([0, 20])

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend()
    fig.tight_layout()
    plt.savefig(f"experiment/data/frame{frame:06d}_outer_losses2.png")
    plt.savefig(f"experiment/data/frame{frame:06d}_outer_losses2.pdf")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7), dpi=700)

    ax1.plot(gn_data, label="Gauss-Newton")

    for sample_mode, sample_mode_name in zip(sample_modes, sample_mode_names):
        for d in [64, 128]:
            filename = f"experiment/data/frame{frame:06d}_outer_losses_sample_{sample_mode}_d-{d}.npy"
            data = np.load(filename, allow_pickle=True)

            outer_losses = data

            name = f"{sample_mode_name} (d={d})"

            # Plot inner losses and errors
            ax1.plot(outer_losses, label=name)

    ax1.set_xlim([0, 20])

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend()
    fig.tight_layout()
    plt.savefig(f"experiment/data/frame{frame:06d}_outer_losses3.png")
    plt.savefig(f"experiment/data/frame{frame:06d}_outer_losses3.pdf")

