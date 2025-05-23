from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 16})

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to plot")

    frame = parser.parse_args().frame

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=700)

    # Load npy data
    block_sizes = [1, 2, 5]
    data = {}
    for block_size in block_sizes:
        filename = f"experiment/data/frame{frame:06d}_rek_inner_losses_block-size-{block_size}.npy"
        data[block_size] = np.load(filename, allow_pickle=True).item()

        min_inner_loss = data[block_size]["min_inner_loss"]
        inner_losses = data[block_size]["inner_losses"]
        inner_errors = data[block_size]["inner_errors"]

        # Plot inner losses and errors
        ax1.plot(inner_losses, label=f"block size = {block_size}")
        ax2.plot(inner_errors, label=f"block size = {block_size}")

    ax1.plot(range(len(inner_losses)), [min_inner_loss] * len(inner_losses), linestyle="--", color="black")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual Norm")
    ax1.legend()

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error Norm")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"experiment/data/frame{frame:06d}_rek_inner_losses.png")
    plt.savefig(f"experiment/data/frame{frame:06d}_rek_inner_losses.pdf")
    print(f"Saved inner losses and errors plot to experiment/data/frame{frame:06d}_rek_inner_losses_block-size-{block_size}.pdf")

