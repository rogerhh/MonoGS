import torch
import time
import math


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err

class HuberLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta=0.1):
        ctx.delta = delta
        signs = torch.sign(x)
        ctx.save_for_backward(x)
        delta_sq = delta ** 2
        loss = torch.where(x.abs() < delta, x, torch.sqrt(2 * delta * x.abs() - delta_sq) * signs)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        delta = ctx.delta
        x, = ctx.saved_tensors
        signs = torch.sign(x)
        delta_sq = delta ** 2
        grad_input = torch.where(x.abs() < delta, grad_output, grad_output * delta / torch.sqrt(2 * delta * x.abs() - delta_sq))
        return grad_input, None

def huber_loss(x, delta=0.1):
    delta_sq = delta ** 2
    import code; code.interact(local=locals())
    loss = torch.where(x.abs() < delta, x.abs(), torch.sqrt(2 * delta * x.abs() - delta_sq))
    # loss = torch.abs(x)
    return loss

def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    # image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    image_ab = (torch.abs(viewpoint.exposure_a) + viewpoint.exposure_eps) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * viewpoint.rgb_pixel_mask - 
                             gt_image * viewpoint.rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - viewpoint.gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

class ApplyExposure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, exposure_a, exposure_b, exposure_eps, sketch_mode=0, sketch_dim=0, stack_dim=0, rand_indices=None, sketch_dexposure=None):
        ctx.sketch_mode = sketch_mode
        ctx.sketch_dim = sketch_dim
        ctx.stack_dim = stack_dim
        rand_indices_row = None
        rand_indices_col = None
        if sketch_mode != 0:
            rand_indices_row = rand_indices[0]
            rand_indices_col = rand_indices[1]
            ctx.repeat_iter = 0

        ctx.save_for_backward(image, exposure_a, exposure_b, sketch_dexposure, rand_indices_row, rand_indices_col, )

        return (torch.abs(exposure_a) + exposure_eps) * image + exposure_b

    @staticmethod
    def backward(ctx, grad_output):
        apply_exposure_start = time.time()
        saved_tensors = ctx.saved_tensors
        image = saved_tensors[0]
        exposure_a = saved_tensors[1]
        exposure_b = saved_tensors[2]
        sketch_dexposure = saved_tensors[3]
        rand_indices_row = saved_tensors[4]
        rand_indices_col = saved_tensors[5]

        grad_output_image = grad_output * image

        # grad_image = torch.sign(exposure_a) * exposure_a * grad_output
        grad_image = torch.abs(exposure_a) * grad_output
        grad_exposure_a = torch.sum(grad_output_image).reshape(exposure_a.shape)
        grad_exposure_b = torch.sum(grad_output).reshape(exposure_b.shape)
        grad_exposure_eps = None
        grad_sketch_mode = None
        grad_sketch_dim = None
        grad_stack_dim = None
        grad_rand_indices = None
        grad_sketch_dexposure = None

        if ctx.sketch_mode != 0:
            sum_start = time.time()
            d = ctx.sketch_dim
            stack_dim = ctx.stack_dim
            assert(sketch_dexposure is not None)
            grad_sketch_dexposure = torch.empty(sketch_dexposure.shape, device=grad_output.device)

            grad_output_image_gray = grad_output_image.sum(dim=0)
            grad_output_gray = grad_output.sum(dim=0)

            temp_indices = torch.arange(3).view(-1, 1, 1, 1)

            repeat_iter = ctx.repeat_iter
            rand_indices_row_i = rand_indices_row[repeat_iter]
            rand_indices_col_i = rand_indices_col[repeat_iter]

            grad_sketch_dexposure[:, :, 0] = grad_output_image[temp_indices, rand_indices_row_i, rand_indices_col_i].sum(dim=(0, -1))
            grad_sketch_dexposure[:, :, 1] = grad_output[temp_indices, rand_indices_row_i, rand_indices_col_i].sum(dim=(0, -1))

            sum_end = time.time()

            ctx.repeat_iter += 1

        apply_exposure_end = time.time()
        # print(f"ApplyExposure backward time ms: {(apply_exposure_end - apply_exposure_start) * 1000}")

        return grad_image, grad_exposure_a, grad_exposure_b, grad_exposure_eps, grad_sketch_mode, grad_sketch_dim, grad_stack_dim, grad_rand_indices, grad_sketch_dexposure


def get_loss_tracking_per_pixel(config, image, depth, opacity, viewpoint, initialization=False, forward_sketch_args=None):
    # We need this but it seems to affect runtime quite a bit
    # image_ab, fn_tracker = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b, None
    if forward_sketch_args is None:
        image_ab = (torch.abs(viewpoint.exposure_a) + viewpoint.exposure_eps) * image + viewpoint.exposure_b
    else:
        image_ab = ApplyExposure.apply(image, viewpoint.exposure_a, viewpoint.exposure_b, viewpoint.exposure_eps, forward_sketch_args["sketch_mode"], forward_sketch_args["sketch_dim"], forward_sketch_args["stack_dim"], forward_sketch_args["rand_indices"], forward_sketch_args["sketch_dexposure"])
    # image_ab = (torch.abs(viewpoint.exposure_a) + viewpoint.exposure_eps) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb_per_pixel(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd_per_pixel(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb_per_pixel(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1 = opacity * (image * viewpoint.rgb_pixel_mask_mapping - 
                    gt_image * viewpoint.rgb_pixel_mask_mapping)
    return l1


def get_loss_tracking_rgbd_per_pixel(config, image, depth, opacity, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb_per_pixel(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = depth * depth_mask - gt_depth * depth_mask
    raise NotImplementedError("This is currently incorrect because depth loss needs to be stacked on top of rgb loss")
    return alpha * l1_rgb + (1 - alpha) * l1_depth


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        # image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        image_ab = (torch.abs(viewpoint.exposure_a) + viewpoint.exposure_eps) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1_rgb = torch.abs(image * viewpoint.rgb_pixel_mask_mapping - 
                       gt_image * viewpoint.rgb_pixel_mask_mapping)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    gt_image = viewpoint.original_image.cuda()

    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - viewpoint.gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_loss_mapping_per_pixel(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        # image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        image_ab = (torch.abs(viewpoint.exposure_a) + viewpoint.exposure_eps) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb_per_pixel(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd_per_pixel(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb_per_pixel(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1_rgb = image * viewpoint.rgb_pixel_mask_mapping - gt_image * viewpoint.rgb_pixel_mask_mapping

    return l1_rgb

def get_loss_mapping_rgbd_per_pixel(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    gt_image = viewpoint.original_image.cuda()

    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - viewpoint.gt_depth * depth_pixel_mask)

    raise NotImplementedError("This is currently incorrect because depth loss needs to be stacked on top of rgb loss")


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
