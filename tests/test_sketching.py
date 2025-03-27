import numpy as np
from scipy.linalg import lstsq
import math
from sketch_utils import run_test

if __name__ == "__main__":
    # Generate a well-conditioned least squares problem
    m, n = 300000, 8  # m >> n for overdetermined system
    noise = 1e-5
    lambda_ = 0.01
    x_norm = 0.015
    gen_max_sigma = 5
    gen_min_sigma = 1e-2
    repeat_dim = 2
    stack_dim = 1
    sketch_dim = 64
    solve_mode = "append_damp" # "sketch_damp" or "append_damp"
    sketch_mode = "count" # "count" or "gaussian"

    run_test(m, n, noise, x_norm, lambda_, gen_max_sigma, gen_min_sigma, repeat_dim, stack_dim, sketch_dim, solve_mode, sketch_mode)
