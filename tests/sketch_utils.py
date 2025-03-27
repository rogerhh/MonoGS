import numpy as np
from scipy.linalg import lstsq
import math

def gen_A(m, n, max_sigma=1, min_sigma=1e-2, lambda_=0.01):
    A = np.random.randn(m, n)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.random.uniform(min_sigma, max_sigma / 1.5, n)
    S[0] = max_sigma
    S[-1] = min_sigma
    A = U @ np.diag(S) @ Vt
    A_damp = np.vstack([A, np.sqrt(lambda_) * np.eye(n)])
    return A, A_damp

def gen_problem(m, n, max_sigma=1, min_sigma=1e-2, lambda_=0.01, noise=0.00001, x_norm=0.015):
    A, A_damp = gen_A(m, n, max_sigma, min_sigma, lambda_)
    x = np.random.randn(n)
    x = x_norm * x / np.linalg.norm(x)
    b = A @ x + noise * np.random.randn(m)
    b_damp = np.concatenate([b, np.zeros(n)])
    return A, A_damp, b, b_damp, x

def get_sketching_matrix(m, n, repeat_dim, stack_dim, sketch_dim, mode="count"):
    if mode == "count":
        d = repeat_dim * stack_dim * sketch_dim
        sketch_dim_per_iter = stack_dim * sketch_dim
        S = np.zeros((repeat_dim, sketch_dim_per_iter, m))
        indices = np.random.choice(sketch_dim_per_iter, size=(repeat_dim, m), replace=True)
        S[np.arange(repeat_dim)[:, None], indices, np.arange(m)] = 1
        S = S.reshape(d, m) / np.sqrt(repeat_dim)
    elif mode == "gaussian":
        d = repeat_dim * stack_dim * sketch_dim
        S = np.random.randn(d, m) / np.sqrt(d)
    return S

def get_distortion(A, A_tilde):
    sigmas = np.linalg.svd(A, compute_uv=False)
    sigmas_tilde = np.linalg.svd(A_tilde, compute_uv=False)
    sigma_max, sigma_min = sigmas[0], sigmas[-1]
    sigma_max_tilde, sigma_min_tilde = sigmas_tilde[0], sigmas_tilde[-1]
    d1 = math.fabs(sigma_max - sigma_max_tilde) / sigma_max
    d2 = math.fabs(sigma_min - sigma_min_tilde) / sigma_min
    distortion = max(d1, d2)
    return distortion

    # n = A.shape[1]
    # num_samples = 1000
    # max_distortion = 0
    # for _ in range(num_samples):
    #     x = np.random.randn(n)
    #     Ax_norm = np.linalg.norm(A @ x)
    #     A_tilde_x_norm = np.linalg.norm(A_tilde @ x)
    #     distortion = math.fabs(Ax_norm - A_tilde_x_norm) / Ax_norm
    #     if distortion > max_distortion:
    #         max_distortion = distortion
    # return max_distortion
        
def run_test(m, n, linear_noise, x_norm, lambda_, max_sval, min_sval, repeat_dim, stack_dim, sketch_dim, solve_mode, sketch_mode):

    A, A_damp, b, b_damp, x = gen_problem(m, n, lambda_=lambda_, noise=linear_noise, x_norm=x_norm, max_sigma=max_sval, min_sigma=min_sval)

    # Solve the full least squares problem
    x_opt, _, _, _ = lstsq(A_damp, b_damp)
    res = np.linalg.norm(A_damp @ x_opt - b_damp, 2)

    # Apply a sketch
    d = repeat_dim * stack_dim * sketch_dim

    if solve_mode == "append_damp":
        S = get_sketching_matrix(m, n, repeat_dim, stack_dim, sketch_dim, mode="count")
        # Compute the sketched system
        A_tilde = S @ A
        b_tilde = S @ b
        A_tilde = np.vstack([A_tilde, np.sqrt(lambda_) * np.eye(n)])
        b_tilde = np.concatenate([b_tilde, np.zeros(n)])
    elif solve_mode == "sketch_damp":
        S = get_sketching_matrix(m+n, n, repeat_dim, stack_dim, sketch_dim, mode="count")
        A_tilde = S @ A_damp
        b_tilde = S @ b_damp

    # Solve the sketched least squares problem
    x_sketch, _, _, _ = lstsq(A_tilde, b_tilde)
    res_sketch = np.linalg.norm(A_damp @ x_sketch - b_damp, 2)
    res_hat = np.linalg.norm(A_tilde @ x_sketch - b_tilde, 2)
    print(f"res_hat: {res_hat:.4f}")
    distortion = get_distortion(A_damp, A_tilde)
    distortion_hat = math.sqrt(n / d)
    x_diff = x_opt - x_sketch

    sigma_min = np.linalg.svd(A_damp, compute_uv=False)[-1]
    sigma_min_hat = np.linalg.svd(A_tilde, compute_uv=False)[-1]

    gamma = (1 + distortion) / (1 - distortion)
    gamma_hat = (1 + distortion_hat) / (1 - distortion_hat)

    upperbound = res * math.sqrt(gamma ** 2 - 1) / sigma_min

    upperbound_hat = res_sketch * gamma * math.sqrt(gamma_hat ** 2 - 1) / sigma_min_hat
    # upperbound_hat = res_sketch * 2 * distortion_hat * (1 + distortion_hat) / (((1 - distortion_hat) ** 2) * sigma_min_hat)

    # Check the angle between the optimal and sketched solutions
    cos_angle = np.dot(x_opt, x_sketch) / (np.linalg.norm(x_opt) * np.linalg.norm(x_sketch))
    angle = np.arccos(cos_angle)
    angle_deg = np.degrees(angle)

    stats = {"sigma_min": sigma_min, "sigma_min_hat": sigma_min_hat, "x_opt_norm": np.linalg.norm(x_opt), "x_sketch_norm": np.linalg.norm(x_sketch), "x_diff_norm": np.linalg.norm(x_diff), "upperbound": upperbound, "upperbound_hat": upperbound_hat, "distortion": distortion, "distortion_hat": distortion_hat, "residual": res, "sketched_residual": res_sketch, "angle_deg": angle_deg}

    # Compare norms
    print(f"sigma_min: {sigma_min}")
    print(f"sigma_min_hat: {sigma_min_hat}")
    print("||x_opt||_2:", np.linalg.norm(x_opt))
    print("||x_sketch||_2:", np.linalg.norm(x_sketch))
    print(f"x_diff norm: {np.linalg.norm(x_diff):.4f}")
    print(f"upperbound: {upperbound:.4f}")
    print(f"upperbound_hat: {upperbound_hat:.4f}")
    print(f"distortion: {distortion:.4f}")
    print(f"distortion_hat: {distortion_hat:.4f}")
    print(f"residual: {res:.4f}, sketched residual: {res_sketch:.4f}")

    print("Angle between x_opt and x_sketch:", angle_deg)


    assert(np.linalg.norm(x_diff) < upperbound), f"||x_diff||_2: {np.linalg.norm(x_diff)} > {upperbound}"
    assert(np.linalg.norm(x_diff) < upperbound_hat), f"||x_diff||_2: {np.linalg.norm(x_diff)} > {upperbound_hat}"
