# Utils used by the model

import sys

import numpy as np
import torch
import torch.nn.functional


########################################################################################################################
# space_to_depth
# From: https://stackoverflow.com/questions/58857720/is-there-an-equivalent-pytorch-function-for-tf-nn-space-to-depth
class SpaceToDepth(torch.nn.Module):
    __constants__ = ['downscale_factor']
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super(SpaceToDepth, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.downscale_factor, stride=self.downscale_factor)
        return unfolded_x.view(n, c * self.downscale_factor * self.downscale_factor, h // self.downscale_factor,
                               w // self.downscale_factor)

    def extra_repr(self) -> str:
        return 'downscale_factor={}'.format(self.downscale_factor)


########################################################################################################################
# Yes, globals are evil, but barely acceptable for caching
# and python does not need mutexes, only 1 thread per process !
meshgrid_cache = {}


def meshgrid_abs_torch(batch, height, width, device, permute):
    """
    Construct a 2D meshgrid in the absolute (homogeneous) coordinates.
    for each pixel (x, y) gives a 3-vector (x, y, 1), where x, y are in PIXELS


    Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    Returns:
    x,y grid coordinates [batch, 3, height, width]
    """
    global meshgrid_cache
    # Cache size Precaution, but the present code creates only 2 entries !
    if len(meshgrid_cache) > 20:
        meshgrid_cache = {}

    key = (batch, height, width, device, permute)
    try:
        res = meshgrid_cache[key]
    except KeyError:
        xs = torch.linspace(0.0, width - 1, width)
        ys = torch.linspace(0.0, height - 1, height)
        ys, xs = torch.meshgrid(ys, xs)
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], axis=0)

        res = torch.unsqueeze(coords, 0).repeat(batch, 1, 1, 1).to(device=device)
        if permute:
            res = res.permute([0, 2, 3, 1])
        meshgrid_cache[key] = res
        # print('CREATING NEW MESHGRID, KEY=', key)
    return res

########################################################################################################################
def pixel2cam_torch(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
    Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.shape
    depth = torch.reshape(depth, [batch, 1, -1])
    pixel_coords = torch.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth

    if is_homogeneous:
        ones = torch.ones([batch, 1, height * width], device=pixel_coords.device)
    cam_coords = torch.cat([cam_coords, ones], axis=1)
    cam_coords = torch.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def cam2pixel_torch(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
    Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.shape
    cam_coords = torch.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords)
    xy_u = unnormalized_pixel_coords[:, 0:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]

    pixel_coords = xy_u / (z_u + 1e-10)
    pixel_coords = torch.reshape(pixel_coords, [batch, 2, height, width])
    return pixel_coords.permute([0, 2, 3, 1])


########################################################################################################################
def resampler_wrapper_torch(imgs, coords):
    """
    equivalent to tfa.image.resampler
    Args:
    imgs: [N, H, W, C] images to resample
    coords: [N, H, W, 2], source pixel locations from which to copy
    Returns:
    [N, H, W, C] sampled pixels
    """
    return torch.nn.functional.grid_sample(
        imgs.permute([0, 3, 1, 2]),  # change images from (N, H, W, C) to (N, C, H, W)
        torch.tensor([-1, -1], device=imgs.device) + 2.0 * coords,  # resize coords from (0,1) to (-1, 1)
        align_corners=True
    ).permute([0, 2, 3, 1])  # change result from (N, C, H, W) to (N, H, W, C)


########################################################################################################################
def projective_inverse_warp_torch2(
        img, depth, pose, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width, ret_flows=False):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 4, 4]
      src_intrinsics: source camera intrinsics [batch, 3, 3]
      tgt_intrinsics: target camera intrinsics [batch, 3, 3]
      tgt_height: pixel height for the target image
      tgt_width: pixel width for the target image
      ret_flows: whether to return the displacements/flows as well
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    batch, height, width, channels = img.shape
    # Construct pixel grid coordinates (x, y, 1) for each pixel.
    # Duplicated for N (e.g. 4) of INPUT images (batch)
    pixel_coords = meshgrid_abs_torch(batch, tgt_height, tgt_width, img.device, False)

    # Note: "target" here means actually "ref image", forget about the ground truth targets!
    # You project pixels from "target" to the multiple inputs, not the other way round
    # Convert pixel coordinates to the target camera frame, 3D camera coords (X, Y, Z), seems OK so far...
    # Note: these are points in 3D camera coords (C) of the target camera, not world coords (W) !!!
    cam_coords = pixel2cam_torch(depth, pixel_coords, tgt_intrinsics)

    # Construct a 4x4 intrinsic matrix, why? wouldn't 3x4 suffice?
    filler = torch.tensor([[[0., 0., 0., 1.]]], device=img.device)
    filler = filler.repeat(batch, 1, 1)
    src_intrinsics4 = torch.cat([src_intrinsics, torch.zeros([batch, 3, 1], device=img.device)], axis=2)
    src_intrinsics4 = torch.cat([src_intrinsics4, filler], axis=1)

    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame, looks OK
    proj_tgt_cam_to_src_pixel = torch.matmul(src_intrinsics4, pose)
    src_pixel_coords = cam2pixel_torch(cam_coords, proj_tgt_cam_to_src_pixel)

    # print(f'src_pixel_coords shape {src_pixel_coords.shape}')
    # print(f'src_pixel_coords {L(src_pixel_coords[:, :, :3,:])}')

    # Now we get trouble !
    if False:
        print(('src_pixel_coords', src_pixel_coords.shape, src_pixel_coords.dtype))
        for i in range(2):
            t = src_pixel_coords[0, :, :, i]
            print((i, t.min().item(), t.max().item()))
        sys.exit(0)

    # src_pixel_coords = (src_pixel_coords + torch.tensor([0.5, 0.5], device=img.device)) / torch.tensor([width, height],
    #                                                                                                    device=img.device)

    src_pixel_coords = src_pixel_coords / torch.tensor([width-1, height-1], device=img.device)

    output_img = resampler_wrapper_torch(img, src_pixel_coords)
    if ret_flows:
        return output_img, src_pixel_coords - cam_coords
    else:
        return output_img


########################################################################################################################
def plane_sweep_torch2(img, depth_planes, pose, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width):
    """Construct a plane sweep volume.

    Args:
      img: source image [batch, height, width, #channels]
      depth_planes: a list of depth values for each plane
      pose: target to source camera transformation [batch, 4, 4]
      src_intrinsics: source camera intrinsics [batch, 3, 3]
      tgt_intrinsics: target camera intrinsics [batch, 3, 3]
      tgt_height: pixel height for the target image
      tgt_width: pixel width for the target image
    Returns:
      A plane sweep volume [#planes, batch, height, width, #channels]
    """
    batch, height, width, _ = img.shape
    plane_sweep_volume = []

    for depth in depth_planes:
        curr_depth = torch.zeros([batch, height, width], dtype=torch.float32, device=img.device) + depth
        warped_img = projective_inverse_warp_torch2(img, curr_depth, pose,
                                                    src_intrinsics, tgt_intrinsics, tgt_height, tgt_width)
        plane_sweep_volume.append(warped_img)
    plane_sweep_volume = torch.stack(plane_sweep_volume, dim=0)
    return plane_sweep_volume

########################################################################################################################
def projective_inverse_warp_torch3(
        img, depth, pose, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width, ret_flows=False):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 4, 4]
      src_intrinsics: source camera intrinsics [batch, 3, 3]
      tgt_intrinsics: target camera intrinsics [batch, 3, 3]
      tgt_height: pixel height for the target image
      tgt_width: pixel width for the target image
      ret_flows: whether to return the displacements/flows as well
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    batch, height, width, channels = img.shape
    # Construct pixel grid coordinates (x, y, 1) for each pixel.
    # Duplicated for N (e.g. 4) of INPUT images (batch)
    #delta_xy = src_center_xy - torch.tensor([float(tgt_width - 1) / 2, float(tgt_height - 1) / 2], device=src_center_xy.device)
    #delta_xyz = torch.cat([delta_xy, torch.zeros([batch, 1], device=delta_xy.device)], dim=1).unsqueeze(-1).unsqueeze(-1)
    # delta xyz [batch, 3, 1, 1]
    pixel_coords = meshgrid_abs_torch(batch, tgt_height, tgt_width, img.device, False)
    #pixel_coords = pixel_coords + delta_xyz

    # Note: "target" here means actually "ref image", forget about the ground truth targets!
    # You project pixels from "target" to the multiple inputs, not the other way round
    # Convert pixel coordinates to the target camera frame, 3D camera coords (X, Y, Z), seems OK so far...
    # Note: these are points in 3D camera coords (C) of the target camera, not world coords (W) !!!
    cam_coords = pixel2cam_torch(depth, pixel_coords, tgt_intrinsics)

    # Construct a 4x4 intrinsic matrix, why? wouldn't 3x4 suffice?
    filler = torch.tensor([[[0., 0., 0., 1.]]], device=img.device)
    filler = filler.repeat(batch, 1, 1)
    src_intrinsics4 = torch.cat([src_intrinsics, torch.zeros([batch, 3, 1], device=img.device)], axis=2)
    src_intrinsics4 = torch.cat([src_intrinsics4, filler], axis=1)

    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame, looks OK
    proj_tgt_cam_to_src_pixel = torch.matmul(src_intrinsics4, pose)
    src_pixel_coords = cam2pixel_torch(cam_coords, proj_tgt_cam_to_src_pixel)

    # print(f'src_pixel_coords shape {src_pixel_coords.shape}')
    # print(f'src_pixel_coords {L(src_pixel_coords[:, :, :3,:])}')

    # Now we get trouble !
    if False:
        print(('src_pixel_coords', src_pixel_coords.shape, src_pixel_coords.dtype))
        for i in range(2):
            t = src_pixel_coords[0, :, :, i]
            print((i, t.min().item(), t.max().item()))
        sys.exit(0)

    # src_pixel_coords = (src_pixel_coords + torch.tensor([0.5, 0.5], device=img.device)) / torch.tensor([width, height],
    #                                                                                                    device=img.device)

    src_pixel_coords = src_pixel_coords / torch.tensor([width-1, height-1], device=img.device)

    output_img = resampler_wrapper_torch(img, src_pixel_coords)
    if ret_flows:
        return output_img, src_pixel_coords - cam_coords
    else:
        return output_img
  
########################################################################################################################
def plane_sweep_torch3(img, depth_planes, pose, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width):
    """Construct a plane sweep volume.

    Args:
      img: source image [batch, height, width, #channels]
      depth_planes: a list of depth values for each plane
      pose: target to source camera transformation [batch, 4, 4]
      src_intrinsics: source camera intrinsics [batch, 3, 3]
      tgt_intrinsics: target camera intrinsics [batch, 3, 3]
      tgt_height: pixel height for the target image
      tgt_width: pixel width for the target image
    Returns:
      A plane sweep volume [#planes, batch, height, width, #channels]
    """
    batch = img.shape[0] 
    plane_sweep_volume = []

    for depth in depth_planes:
        curr_depth = torch.zeros([batch, tgt_height, tgt_width], dtype=torch.float32, device=img.device) + depth
        warped_img = projective_inverse_warp_torch3(img, curr_depth, pose,
                                                    src_intrinsics, tgt_intrinsics, tgt_height, tgt_width)
        plane_sweep_volume.append(warped_img)
    plane_sweep_volume = torch.stack(plane_sweep_volume, dim=0)
    return plane_sweep_volume


########################################################################################################################

def rgba_premultiply(rgba):
  """ Premultiplies the RGB channels with the Alpha channel
    Args:
        rgba: [batch, channels=4, height, width] range [0, 1]
    Returns:
        returns [batch,channels=4, height, width] range [0, 1]
  """
  premultiplied_rgb = rgba[:,:3,:,:] * rgba[:,3,:,:].unsqueeze(1).expand(-1, 3, -1, -1)
  return torch.cat([premultiplied_rgb, rgba[:,3,:,:].unsqueeze(1)], 1)


########################################################################################################################

def repeated_over(colours, alpha, depth):
  """
    Args:
    colours: [layers, views, height, width, 3] range [0, 1]
    alpha: [layers, views, height, width] it's alpha, range [0, 1]
    depth: integer
  """
  if depth < 1:
    return colours[0] * alpha[0].unsqueeze(-1) 
  complement_alpha = 1. - alpha
  premultiplied_colours = colours * alpha.unsqueeze(-1)
  return torch.sum(torch.stack([
        premultiplied_colours[i] * torch.prod(complement_alpha[i+1:depth], 0).unsqueeze(-1).expand(-1, -1, -1, 3)
        for i in range(depth)
    ]), 0)


########################################################################################################################

def calculate_mpi_gradient(raw_mpi, mpi_wfc, mpi_planes, views_cfw, views_intrinsics, camera_images, mpi_intrinsics):
  """
    Args:
    raw_mpi: input MPI [depths, channels=8, height, width]
    mpi_wfc: [4, 4]
    mpi_planes: a list of depth values for each plane [depths]
    views_cfw: [views, 4, 4]
    views_intrinsics: camera intrinsics [views, 3, 3]
    camera_images: original camera images [views, height, width, 3]
    mpi_intrinsics: intrinsics of the camera view used by the MPI [3, 3]]
    Returns:
        [depths, views, channels=10, height, width]
  """
  from utils_render import projective_forward_homography_torch
  depths, _, height, width = raw_mpi.shape
  views = views_cfw.shape[0]
  tgt_pose = torch.matmul(views_cfw, torch.unsqueeze(mpi_wfc,0)) # [views, 4, 4]
  rgba_layers = raw_mpi[:,:4].unsqueeze(0).expand(views, -1, -1, -1, -1) # [views, depths, channels, height, width]

  proj_images = projective_forward_homography_torch(
      rgba_layers.permute(1, 0, 3, 4, 2), 
      mpi_intrinsics.unsqueeze(0).expand(views, -1, -1),
      views_intrinsics,
      tgt_pose, 
      mpi_planes.unsqueeze(1).expand(-1, views)
  ).permute(0,1,3,4,2) # [layers, views, height, width, channels=4]

  colours = proj_images[:, :, :, :, :3] # [layers, views, height, width, 3]
  normalized_alphas = proj_images[:,:,:,:,3] # [layers, views, height, width]
  complement_alpha = 1. - normalized_alphas

  net_transmittance = torch.stack([
    torch.prod(complement_alpha[d+1:], 0) for d in range(depths)
  ]) # [layers, views, height, width]

  accumulated_over = torch.stack([
    repeated_over(colours, normalized_alphas, d-1)
    for d in range(depths)
  ]) # [layers, views, height, width, 3]

  broadcasted_over = repeated_over(colours, normalized_alphas, depths).unsqueeze(0).expand(depths, -1, -1, -1, -1) # [layers, views, height, width, 3]

  broadcasted_camera_imgs = camera_images.unsqueeze(0).expand(depths, -1, -1, -1, -1) # [layers, views, height, width, 3]

  # print("one_to_two_normalize(net_transmittance).unsqueeze(-1).shape {}".format(one_to_two_normalize(net_transmittance).unsqueeze(-1).shape))
  # print("one_to_two_normalize(accumulated_over).shape {}".format(one_to_two_normalize(net_transmittance).shape))
  # print("one_to_two_normalize(broadcasted_over).shape {}".format(one_to_two_normalize(broadcasted_over).shape))
  # print("broadcasted_camera_imgs.shape {}".format(broadcasted_camera_imgs.shape))
  # print("camera_images.shape {}".format(camera_images.shape))
  # print("proj_images.shape {}".format(proj_images.shape))
  # print("net_transmittance.shape {}".format(net_transmittance.shape))
  # print("rgba_layers.shape {}".format(rgba_layers.shape))
  stacked_input = torch.cat([
    net_transmittance.unsqueeze(-1), # 1
    accumulated_over, # 3
    broadcasted_over, # 3
    broadcasted_camera_imgs # 3
  ], dim=-1) # [depths, views, height, width, channels=10]
  #print("stacked_input.shape {}".format(stacked_input.shape))

  calculated_depths = torch.zeros([depths, height, width], dtype=torch.float32, device=mpi_planes.device) + mpi_planes.contiguous().view(depths,1,1) # [depths, height, width]
  
  calculated_depths = calculated_depths.unsqueeze(1).expand(-1, views, -1, -1).contiguous().view(depths * views, height, width) # [depths*views, height, width]
  calculated_pose = tgt_pose.unsqueeze(0).expand(depths, -1, -1, -1).contiguous().view(depths * views, 4, 4) # [depths*views, 4, 4]
  calculated_tgt_intrinsics = mpi_intrinsics.unsqueeze(0).expand(depths * views, -1, -1)
  calculated_src_intrinsics = views_intrinsics.unsqueeze(0).expand(depths, -1, -1, -1).contiguous().view(depths * views, 3, 3)

  return projective_inverse_warp_torch2(
      stacked_input.contiguous().view(depths * views, height, width, 10), # [depths * views, height, width, channels=10]
      calculated_depths,
      calculated_pose,
      calculated_src_intrinsics,
      calculated_tgt_intrinsics,
      height, width
  ).contiguous().view(depths, views, height, width, 10).permute(0,1,4,2,3)
