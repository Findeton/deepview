import sys

import numpy as np
import torch
import torch.nn.functional

from utils_model import meshgrid_abs_torch

########################################################################################################################
def divide_safe_torch(num, den, name=None):
    eps = 1e-8
    den = den.to(torch.float32)
    den += eps * den.eq(torch.tensor(0, device=den.device, dtype=torch.float32))
    return torch.div(num.to(torch.float32), den)


def transpose_torch(rot):
    return torch.transpose(rot, len(rot.shape) - 2, len(rot.shape) - 1)


########################################################################################################################
def inv_homography_torch(k_s, k_t, rot, t, n_hat, a):
    """Computes inverse homography matrix between two cameras via a plane.

    Args:
        k_s: intrinsics for source cameras, [..., 3, 3] matrices
        k_t: intrinsics for target cameras, [..., 3, 3] matrices
        rot: relative rotations between source and target, [..., 3, 3] matrices
        t: [..., 3, 1], translations from source to target camera. Mapping a 3D
          point p from source to target is accomplished via rot * p + t.
        n_hat: [..., 1, 3], plane normal w.r.t source camera frame
        a: [..., 1, 1], plane equation displacement
    Returns:
        homography: [..., 3, 3] inverse homography matrices (homographies mapping
          pixel coordinates from target to source).
    """
    rot_t = transpose_torch(rot)
    k_t_inv = torch.inverse(k_t)

    denom = a - torch.matmul(torch.matmul(n_hat, rot_t), t)
    numerator = torch.matmul(torch.matmul(torch.matmul(rot_t, t), n_hat), rot_t)
    inv_hom = torch.matmul(
        torch.matmul(k_s, rot_t + divide_safe_torch(numerator, denom)),
        k_t_inv)
    return inv_hom


########################################################################################################################
def transform_points_torch(points, homography):
    """Transforms input points according to homography.

    Args:
        points: [..., H, W, 3]; pixel (u,v,1) coordinates.
        homography: [..., 3, 3]; desired matrix transformation
    Returns:
        output_points: [..., H, W, 3]; transformed (u,v,w) coordinates.
    """
    # Because the points have two additional dimensions as they vary across the
    # width and height of an image, we need to reshape to multiply by the
    # per-image homographies.
    points_orig_shape = points.shape
    points_reshaped_shape = list(homography.shape)
    points_reshaped_shape[-2] = -1

    points_reshaped = torch.reshape(points, points_reshaped_shape)
    transformed_points = torch.matmul(points_reshaped, transpose_torch(homography))
    transformed_points = torch.reshape(transformed_points, points_orig_shape)
    return transformed_points


########################################################################################################################
def normalize_homogeneous_torch(points):
    """Converts homogeneous coordinates to regular coordinates.

    Args:
        points: [..., n_dims_coords+1]; points in homogeneous coordinates.
    Returns:
        points_uv_norm: [..., n_dims_coords];
            points in standard coordinates after dividing by the last entry.
    """
    uv = points[..., :-1]
    w = torch.unsqueeze(points[..., -1], -1)
    return divide_safe_torch(uv, w)


########################################################################################################################
def bilinear_wrapper_torch(imgs, coords):
    """Wrapper around bilinear sampling function, handles arbitrary input sizes.

    Args:
      imgs: [..., H_s, W_s, C] images to resample
      coords: [..., H_t, W_t, 2], source pixel locations from which to copy
    Returns:
      [..., H_t, W_t, C] images after bilinear sampling from input.
    """
    # The bilinear sampling code only handles 4D input, so we'll need to reshape.
    init_dims = list(imgs.shape[:-3:])
    end_dims_img = list(imgs.shape[-3::])
    end_dims_coords = list(coords.shape[-3::])
    prod_init_dims = init_dims[0]
    for ix in range(1, len(init_dims)):
        prod_init_dims *= init_dims[ix]

    imgs = torch.reshape(imgs, [prod_init_dims] + end_dims_img)
    coords = torch.reshape(
        coords, [prod_init_dims] + end_dims_coords)
    # change image from (N, H, W, C) to (N, C, H, W)
    imgs = imgs.permute([0, 3, 1, 2])
    # TODO: resize coords from (0,1) to (-1, 1)
    coords2 = torch.Tensor([-1, -1]).to(imgs.device) + 2.0 * coords
    imgs_sampled = torch.nn.functional.grid_sample(imgs, coords2, align_corners=True)
    # imgs_sampled = torch.div(2.0* (imgs_sampled0 + torch.Tensor([1.0, 1.0])).to(device), torch.Tensor([(x_max - x_min), (y_max - y_min)])).to(device)
    # permute back to (N, H, W, C)
    imgs = imgs.permute([0, 2, 3, 1])
    imgs_sampled = torch.reshape(
        imgs_sampled, init_dims + list(imgs_sampled.shape)[-3::])
    return imgs_sampled


########################################################################################################################
def transform_plane_imgs_torch(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
    """Transforms input imgs via homographies for corresponding planes.

    Args:
      imgs: are [..., H_s, W_s, C]
      pixel_coords_trg: [..., H_t, W_t, 3]; pixel (u,v,1) coordinates.
      k_s: intrinsics for source cameras, [..., 3, 3] matrices
      k_t: intrinsics for target cameras, [..., 3, 3] matrices
      rot: relative rotation, [..., 3, 3] matrices
      t: [..., 3, 1], translations from source to target camera
      n_hat: [..., 1, 3], plane normal w.r.t source camera frame
      a: [..., 1, 1], plane equation displacement
    Returns:
      [..., H_t, W_t, C] images after bilinear sampling from input.
        Coordinates outside the image are sampled as 0.
    """
    hom_t2s_planes = inv_homography_torch(k_s, k_t, rot, t, n_hat, a)
    # print("hom_t2s_planes ", L(hom_t2s_planes))
    pixel_coords_t2s = transform_points_torch(pixel_coords_trg, hom_t2s_planes)
    # print("pixel_coords_t2s ", L(pixel_coords_t2s))
    pixel_coords_t2s = normalize_homogeneous_torch(pixel_coords_t2s)
    # print("imgs shape", imgs.shape)
    # print("pixel_coords_trg shape", pixel_coords_trg.shape)
    # print("pixel_coords_t2s shape", pixel_coords_t2s.shape)

    # convert from [0-height-1, width -1] to [0-1, 0-1]
    height_t = pixel_coords_trg.shape[-3]
    width_t = pixel_coords_trg.shape[-2]
    pixel_coords_t2s = pixel_coords_t2s / torch.tensor([width_t-1, height_t-1], device=imgs.device)

    # print("pixel_coords_t2s ", L(pixel_coords_t2s))

    imgs_s2t = bilinear_wrapper_torch(imgs, pixel_coords_t2s)
    # print("imgs_s2t ", L(imgs_s2t))

    return imgs_s2t


########################################################################################################################
def planar_transform_torch(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
    """Transforms imgs, masks and computes dmaps according to planar transform.

    Args:
      imgs: are [L, B, H, W, C], typically RGB images per layer
      pixel_coords_trg: tensors with shape [B, H_t, W_t, 3];
          pixel (u,v,1) coordinates of target image pixels. (typically meshgrid)
      k_s: intrinsics for source cameras, [B, 3, 3] matrices
      k_t: intrinsics for target cameras, [B, 3, 3] matrices
      rot: relative rotation, [B, 3, 3] matrices
      t: [B, 3, 1] matrices, translations from source to target camera
         (R*p_src + t = p_tgt)
      n_hat: [L, B, 1, 3] matrices, plane normal w.r.t source camera frame
        (typically [0 0 1])
      a: [L, B, 1, 1] matrices, plane equation displacement
        (n_hat * p_src + a = 0)
    Returns:
      imgs_transformed: [L, ..., C] images in trg frame
    Assumes the first dimension corresponds to layers.
    """
    n_layers = list(imgs.shape)[0]
    rot_rep_dims = [n_layers]
    rot_rep_dims += [1 for _ in range(len(list(k_s.shape)))]

    cds_rep_dims = [n_layers]
    cds_rep_dims += [1 for _ in range(len(list(pixel_coords_trg.shape)))]

    k_s = torch.unsqueeze(k_s, 0).repeat(rot_rep_dims)
    k_t = torch.unsqueeze(k_t, 0).repeat(rot_rep_dims)
    t = torch.unsqueeze(t, 0).repeat(rot_rep_dims)
    rot = torch.unsqueeze(rot, 0).repeat(rot_rep_dims)
    pixel_coords_trg = torch.unsqueeze(pixel_coords_trg, 0).repeat(cds_rep_dims)

    imgs_trg = transform_plane_imgs_torch(
        imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return imgs_trg


########################################################################################################################
def projective_forward_homography_torch(src_images, intrin_tgt, intrin_ref, pose, depths):
    """Use homography for forward warping.

    Args:
      src_images: [layers, batch, height, width, channels]
      intrin_tgt: target camera intrinsics [batch, 3, 3]
      intrin_ref: source camera intrinsics [batch, 3, 3]
      pose: [batch, 4, 4]
      depths: [layers, batch]
    Returns:
      proj_src_images: [layers, batch, channels, height, width]
    """
    n_layers, n_batch, height, width, _ = src_images.shape
    # Format for planar_transform code:
    # rot: relativplane_sweep_torch_onee rotation, [..., 3, 3] matrices
    # t: [B, 3, 1], translations from source to target camera (R*p_s + t = p_t)
    # n_hat: [L, B, 1, 3], plane normal w.r.t source camera frame [0,0,1]
    #        in our case
    # a: [L, B, 1, 1], plane equation displacement (n_hat * p_src + a = 0)
    rot = pose[:, :3, :3]
    t = pose[:, :3, 3:]
    n_hat = torch.tensor([0., 0., 1.], device=src_images.device).reshape(
        [1, 1, 1, 3])  # tf.constant([0., 0., 1.], shape=[1, 1, 1, 3])
    n_hat = n_hat.repeat([n_layers, n_batch, 1, 1])
    a = -torch.reshape(depths, [n_layers, n_batch, 1, 1])
    k_s = intrin_ref
    k_t = intrin_tgt
    pixel_coords_trg = meshgrid_abs_torch(n_batch, height, width, src_images.device, True)
    proj_src_images = planar_transform_torch(
        src_images, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return proj_src_images


########################################################################################################################
def over_composite(rgbas):
    """Combines a list of RGBA images using the over operation.

    Combines RGBA images from back to front with the over operation.
    The alpha image of the first image is ignored and assumed to be 1.0.

    Args:
      rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
    Returns:
      Composited RGB image.
    """
    for i in range(len(rgbas)):
        rgb = rgbas[i][:, :, :, 0:3]
        alpha = rgbas[i][:, :, :, 3:]
        # print('rgb.shape', rgb.shape)
        # print('alpha.shape', alpha.shape)
        if i == 0:
            output = rgb
        else:
            rgb_by_alpha = rgb * alpha
            output = rgb_by_alpha + output * (1.0 - alpha)
    return output


########################################################################################################################
def mpi_render_view_torch(rgba_layers, tgt_pose, planes, intrin_tgt, intrin_ref):
    """Render a target view from an MPI representation.

    Args:
      rgba_layers: input MPI [batch, height, width, #planes, 4]
      tgt_pose: target pose to render from [batch, 4, 4]
      planes: list of depth for each plane
      intrin_tgt: target camera intrinsics [batch, 3, 3]
      intrin_ref: source camera intrinsics [batch, 3, 3]
    Returns:
      rendered view [batch, height, width, 3]
    """
    batch_size, _, _ = list(tgt_pose.shape)
    depths = planes.reshape([len(planes), 1])
    depths = depths.repeat(1, batch_size)
    # print(rgba_layers.cpu().shape)
    # to [#planes, batch, height, width, 4]
    rgba_layers = rgba_layers.permute([3, 0, 1, 2, 4])
    proj_images = projective_forward_homography_torch(rgba_layers, intrin_tgt, intrin_ref,
                                                      tgt_pose, depths)
    # proj_images is [#planes, batch, 4, height, width]
    # change to [#planes, batch, H, W, 4]
    proj_images = proj_images.permute([0, 1, 3, 4, 2])
    proj_images_list = []
    # print("proj_images.shape", proj_images.shape)
    for i in range(len(planes)):
        proj_images_list.append(proj_images[i])
    output_image = over_composite(proj_images_list)  # same as tensorflow's version!
    return output_image


########################################################################################################################

