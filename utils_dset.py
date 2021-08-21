"""Utilities for dataset classes"""

import numpy as np
import torch

########################################################################################################################
def make_intrinsics_matrix(fx, fy, cx, cy):
    return torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

########################################################################################################################
def extract_intrinsics_matrix(intrinsics):
    base = intrinsics.cpu().numpy()
    fx = base[0,0]
    fy = base[1,1]
    cx = base[0,2]
    cy = base[1,2]

    return [fx, fy, cx, cy]


########################################################################################################################
def scale_intrinsics(intrinsics, sy, sx):
    """
    scale intrinsics
    wrong for non-diagonal intrinsics ?

    """
    return intrinsics * torch.tensor([
        [sx, 1.0, sx],
        [0.0, sy, sy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)



########################################################################################################################
def crop_intrinsics(intrinsics, height, width, crop_cx, crop_cy):
    """Convert to new camera intrinsics for crop of image from original camera.
    From: https://github.com/BerkeleyAutomation/autolab_core/blob/master/autolab_core/camera_intrinsics.py#L222

    Args:
      intrinsics: [fx, fy, cx, cy]
      height: of crop window
      width: width of crop window
      crop_cx: col of crop window center
      crop_cy: row of crop window center
    Returns:
      3x3 intrinsics matrix
    """
    fx, fy, cx, cy = intrinsics
    cx2 = cx + float(width - 1) / 2 - crop_cx
    cy2 = cy + float(height - 1) / 2 - crop_cy
    return make_intrinsics_matrix(fx, fy, cx2, cy2)


########################################################################################################################
def resize_totensor_intrinsics(img, intrin, im_w, im_h):
    """Resize PIL image with intrinsics and convert to tensor"""
    s_intrin = scale_intrinsics(intrin, im_h / img.height, im_w / img.width)
    s_img = img.resize((im_w, im_h))
    t = torch.tensor(np.array(s_img) / 255, dtype=torch.float32)
    return t, s_intrin


########################################################################################################################
def inv_depths(start_depth, end_depth, num_depths):
    """Sample reversed, sorted inverse depths between a near and far plane.

    Args:
      start_depth: The first depth (i.e. near plane distance).
      end_depth: The last depth (i.e. far plane distance).
      num_depths: The total number of depths to create. start_depth and
          end_depth are always included and other depths are sampled
          between them uniformly according to inverse depth.
    Returns:
      The depths sorted in descending order (so furthest first). This order is
      useful for back to front compositing.
    """
    # NOte: original file had ugly code, I do it in one line !
    return 1.0 / np.linspace(1.0 / start_depth, 1.0 / end_depth, num_depths)[::-1]

########################################################################################################################

def pixel_pos_to_world_pos(pixel_x, pixel_y, z_dist, intrin, wfc):
    """Get the world position from the pixel position on camera and the z/distance.

    Args:
        pixel_x, pixel_y, z_dist: scalar
        intrin: [3,3]
        wfc: [4,4]
    Returns:
        [4] (torch)
    """
    fx, fy, cx, cy = extract_intrinsics_matrix( intrin )
    ref_cam_x =    z_dist * (pixel_x - cx) / fx
    ref_cam_y =    z_dist * (pixel_y - cy) / fy
    ref_cam_center = torch.tensor([ref_cam_x, ref_cam_y, z_dist, 1.0], device=wfc.device)

    return torch.matmul(wfc, ref_cam_center)

########################################################################################################################

def clamped_world_to_pixel(pos_world, cfw, intrin, crop_x, crop_y, img_w, img_h):
    """Convert world position to camera pixel position, clamped thinking on cropping.

    Args:
        pos_world: [4] (last dim value is 1.0)
        cfw: [#cam, 4, 4]
        intrin: [#cam, 3, 3]
        crop_x, crop_y, img_w, img_h: scalars
    Returns:
        [#cam, 2] (numpy) order is [#cam][x, y]
    """
    in_cam_center = torch.matmul(cfw, pos_world)
    in_pixel_center = torch.matmul(intrin, in_cam_center[:,:3].unsqueeze(-1)).squeeze(-1) # [cams, 3]
    in_pixel_xy = torch.div(in_pixel_center[:, :2], in_pixel_center[:, 2].unsqueeze(-1)) # [cams, 2] the 2 are (x, y)
    in_pixel_x = torch.clamp(in_pixel_xy[:,0], min = crop_x / 2, max = img_w - crop_x / 2)
    in_pixel_y = torch.clamp(in_pixel_xy[:,1], min = crop_y / 2, max = img_h - img_h / 2)
    in_pixel_xy_np = torch.floor(torch.cat([in_pixel_x.unsqueeze(-1), in_pixel_y.unsqueeze(-1)], dim = 1)).cpu().numpy() # [cams, 2]
    return in_pixel_xy_np

########################################################################################################################

def crop_image(img, center_x, center_y, crop_w, crop_h):
    h_min = int(center_y - crop_h // 2)
    h_max = int(center_y + crop_h // 2)
    w_min = int(center_x - crop_w // 2)
    w_max = int(center_x + crop_w // 2)
    return img[h_min:h_max, w_min:w_max]

########################################################################################################################
    
def crop_scale_things(res, ref_pos, im_w_max, im_h_max, image_base_width, image_base_height):
    ref_pixel_x, ref_pixel_y, ref_cam_z = ref_pos

    ref_cam_center_world = pixel_pos_to_world_pos(
        ref_pixel_x, ref_pixel_y, ref_cam_z, res['ref_intrin'], res['ref_wfc']
    )

    device = res['ref_intrin'].device
    res['base_pixel_xy'] = torch.tensor([ ref_pixel_x, ref_pixel_y], device=device)

    # calculate pixel position in other cameras
    in_pixel_xy_np = clamped_world_to_pixel(
        ref_cam_center_world, res['in_cfw'], res['in_intrin'], im_w_max, im_h_max, image_base_width, image_base_height)

    res['in_pixel_xy'] = torch.from_numpy(in_pixel_xy_np).to(device)

    # adjust intrinsics
    new_in_intrin = []
    for idx, intrin in enumerate(res['in_intrin']):
        base_intrin = extract_intrinsics_matrix(intrin)
        cropped_intrin = crop_intrinsics(base_intrin, im_h_max, im_w_max, in_pixel_xy_np[idx, 0], in_pixel_xy_np[idx, 1])
        new_in_intrin.append(cropped_intrin.unsqueeze(0))
    res['in_intrin'] = torch.cat(new_in_intrin, dim=0)

    # adjust ref intrinsics
    ref_intrin_ext = extract_intrinsics_matrix(res['ref_intrin'])
    res['ref_intrin'] =  crop_intrinsics(
        ref_intrin_ext, im_h_max, im_w_max, ref_pixel_x,  ref_pixel_y)

    # crop image
    new_in_img = []
    for idx, img in enumerate(res['in_img']):
        cropped_img = crop_image(
            img, in_pixel_xy_np[idx, 0], in_pixel_xy_np[idx, 1], im_w_max, im_h_max
        )
        new_in_img.append(cropped_img.unsqueeze(0))
    res['in_img'] = torch.cat(new_in_img, dim=0)

    # adjust tgt intrinsics
    tgt_cam_xy = clamped_world_to_pixel(
        ref_cam_center_world, res['tgt_cfw'], res['tgt_intrin'],
        im_w_max, im_h_max, image_base_width, image_base_height)[0]
    base_tgt_intrin = extract_intrinsics_matrix( res['tgt_intrin'][0])
    res['tgt_intrin'] = crop_intrinsics(
        base_tgt_intrin, im_h_max, im_w_max, tgt_cam_xy[0], tgt_cam_xy[1]).unsqueeze(0)
    
    # adjust tgt image
    res['tgt_img'] = crop_image(
            res['tgt_img'][0], tgt_cam_xy[0], tgt_cam_xy[1], im_w_max, im_h_max
        ).unsqueeze(0)
    
    return res

def wrap_input(inp):
    out = {}
    for k in inp.keys():
        out[k] = inp[k].unsqueeze(0)
    return out

def unwrap_input(inp):
    out = {}
    for k in inp.keys():
        out[k] = inp[k][0]
    return out
