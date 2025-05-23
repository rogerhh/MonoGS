import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

def generate_inconsistent_system(m, n, x_norm=1.0, r_norm=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(m, n)
    x = np.random.randn(n)
    x = x / np.linalg.norm(x) * x_norm
    b = A @ x
    r = np.random.randn(m)
    r = r / np.linalg.norm(r) * r_norm
    b_noisy = b + r

    return A, x, b_noisy, b, r

def blocked_randomized_kaczmarz(A, b, block_size=10, max_iters=200):
    m, n = A.shape
    x = np.random.randn(n) * 1
    losses = []

    for _ in range(max_iters):
        # Record the residual norm
        loss = np.linalg.norm(A @ x - b) ** 2
        losses.append(loss)

        print(f"Iteration {_ + 1}/{max_iters}, Loss: {loss:.6f}")

        # Select a random block of rows
        block = np.random.choice(m, size=block_size, replace=False)
        A_block = A[block]
        b_block = b[block]

        # Compute pseudo-inverse update step
        residual = b_block - A_block @ x
        update = np.linalg.pinv(A_block) @ residual
        x += update

        # Record the residual norm
        loss = np.linalg.norm(A @ x - b) ** 2
        losses.append(loss)

        print(f"Iteration {_ + 1}/{max_iters}, Loss: {loss:.6f}")

    return x, losses

# Example usage
if __name__ == "__main__":
    m, n = 100000, 8
    x_norm = 1.0
    r_norm = 0.005
    block_size = 5
    max_iters = 100

    A, x_true, b_noisy, _, _ = generate_inconsistent_system(m, n, x_norm, r_norm, seed=42)
    x_est, losses = blocked_randomized_kaczmarz(A, b_noisy, block_size, max_iters)

    # Plot the loss curve
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss: ||Ax - b||^2")
    plt.title("Blocked Randomized Kaczmarz")
    plt.grid(True)
    plt.savefig("blocked_randomized_kaczmarz_loss.png", dpi=300)
    # plt.show()

