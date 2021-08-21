import pathlib

import torch
import torch.utils.data
import torch.nn.functional

import utils_dset

import cv2 as cv
import torch
import pytorch3d.transforms
import os
import math
import json
import imageio
import random
import pathlib
import misc

def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)[...,:3]
    else:
        return imageio.imread(f)[...,:3]

# rot [3, 3]
# pos [3]
def create_cfw(rot, pos):
    ret = torch.cat([rot, pos.unsqueeze(-1)], dim=1)
    l = torch.tensor([0,0,0,1.0], device = rot.device)
    return torch.cat([ret, l.unsqueeze(0)], dim=0)

def import_blender_folder(blender_path, device, max_w = 800):
    with open(os.path.join(blender_path, "metadata.json"), 'r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
    
    images = [imread(os.path.join(blender_path, 'images', f)) for f in metadata['images'].keys()]
    num_cams = len(images)

    h0 = images[0].shape[0]
    w0 = images[0].shape[1]
    w = max_w
    h = int(h0*w/w0)

    images = [cv.resize(img, (w, h)) for img in images]

    f = 0.5*h / math.tan(0.5*(math.pi/180)*metadata['vfov'])
    in_intrin = utils_dset.make_intrinsics_matrix(f, f, w*1.0, h*1.0).to(device).unsqueeze(0).expand(num_cams, -1, -1) # [9, 3, 3]

    num_planes = 10
    mpi_planes = torch.tensor(utils_dset.inv_depths(1, 100, num_planes)).float().to(device)

    # now use pytorch3d.transforms.euler_angles_to_matrix
    # https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html

    in_rot = [torch.tensor(image['rotation'], device=device).float()  * math.pi  / 180 for image in metadata['images'].values()]
    in_rot = torch.cat([pytorch3d.transforms.euler_angles_to_matrix(rot, "XYZ").unsqueeze(0) for rot in in_rot], dim=0)

    in_pos = [torch.tensor(image['location'], device=device).float() for image in metadata['images'].values()]

    in_cfw = torch.cat([torch.inverse(create_cfw(in_rot[idx], in_pos[idx])).unsqueeze(0) for idx in range(len(in_rot))], dim=0)

    ref_idx = 4
    ref_idx1 = ref_idx+1

    in_img = torch.cat([torch.from_numpy(img).float().to(device).unsqueeze(0)/255. for img in images], dim=0)
    ref_intrin = in_intrin[ref_idx].unsqueeze(0)
    ref_cfw = in_cfw[ref_idx].unsqueeze(0)
    ref_wfc = torch.inverse(in_cfw[ref_idx]).unsqueeze(0)

    #in_img = torch.cat((in_img[:ref_idx], in_img[ref_idx1:]), axis = 0)
    #in_intrin = torch.cat((in_intrin[:ref_idx], in_intrin[ref_idx1:]), axis = 0)
    #in_cfw = torch.cat((in_cfw[:ref_idx], in_cfw[ref_idx1:]), axis = 0)

    return {
        'in_img': in_img.unsqueeze(0), # [1, 9, 132, 200, 3]
        'in_intrin': in_intrin.unsqueeze(0), # [1, 9, 3, 3]
        'in_cfw': in_cfw.unsqueeze(0), # [1, 9, 4, 4]
        'mpi_planes': mpi_planes.unsqueeze(0), # [1, 10]
        'ref_intrin': ref_intrin, # [1, 9, 3, 3]
        'ref_cfw': ref_cfw, # [1, 4, 4]
        'ref_wfc': ref_wfc, # [1, 3, 3]
    }

def crop_model_input(x, x0, y0, tile_w, tile_h, no_crop = False):
    #x = utils_dset.unwrap_input(inp)
    _, base_h, base_w, _ = x['in_img'].shape

    ref_pixel_x = x0 + tile_w//2
    ref_pixel_y = y0 + tile_h//2

    x['in_intrin_base'] = x['in_intrin']
    x['ref_intrin_base'] = x['ref_intrin']
    x['in_img_base'] = x['in_img']
    ref_cam_z = x['mpi_planes'][x['mpi_planes'].shape[0]//2]
    
    if no_crop:
        return x
    else:
        ref_pos = (ref_pixel_x, ref_pixel_y, ref_cam_z)

        res = utils_dset.crop_scale_things(x, ref_pos, tile_w, tile_h, base_w, base_h)
        return res

########################################################################################################################

def slice_tensor(inp, idx):
    return torch.cat((inp[:idx], inp[(idx+1):]), axis = 0)

def create_input_target(inp, tgt_idx):
    x = {
        'in_img': slice_tensor(inp['in_img'][0], tgt_idx), # [9, 132, 200, 3] we don't need the batch dim here!
        'in_intrin': slice_tensor(inp['in_intrin'][0], tgt_idx),
        'in_cfw': slice_tensor(inp['in_cfw'][0], tgt_idx),
        'tgt_img': inp['in_img'][:, tgt_idx],
        'tgt_intrin': inp['in_intrin'][:, tgt_idx],
        'tgt_cfw': inp['in_cfw'][:, tgt_idx],
        'mpi_planes': inp['mpi_planes'][0],
        'ref_intrin': inp['ref_intrin'][0],
        'ref_cfw': inp['ref_cfw'][0],
        'ref_wfc': inp['ref_wfc'][0],
    }
    return x

########################################################################################################################
class DsetBlender(torch.utils.data.Dataset):
    """
    Blender dataset
    """

    def __init__(self, dataset_path, is_valid=False, im_w=200, im_h=120, num_planes=10, max_w = 800, no_crop = False):
        print(f'DsetBlender: dataset_path={dataset_path}, is_valid={is_valid}')
        self.is_valid = is_valid
        self.dataset_path = pathlib.Path(dataset_path)
        self.im_w = im_w
        self.im_h = im_h
        self.num_planes = num_planes
        self.no_crop = no_crop
        self.max_w = max_w

        device_type = os.environ.get('DEVICE', 'cpu')
        device =  torch.device(device_type)

        self.base_data = import_blender_folder(dataset_path, device, max_w)
        self.scenes = self.create_test_scenes(self.base_data) if self.is_valid else self.create_real_scenes(self.base_data)

    def create_test_scenes(self, base_data):
        tile_h = self.im_h
        tile_w = self.im_w
        self.margin_w = tile_w // 5
        self.margin_h = tile_h // 5

        base_h = base_data['in_img'].shape[2]
        base_w = base_data['in_img'].shape[3]

        iters_w = base_w // tile_w
        iters_h = base_h // tile_h

        possible_tgt_idxs = [0,1,2,3,5,6,7,8]

        scenes = []
        for yi in range(0, iters_h):
            y0 = yi * tile_h
            for xi in range(0, iters_w):
                x0 = xi * tile_w

                x_ij = misc.to_device(base_data, base_data['in_img'].device)  # One batch
                tgt_idx = possible_tgt_idxs[(yi * iters_h + xi) % len(possible_tgt_idxs)]
                x_ij = create_input_target(x_ij, tgt_idx)
                scene = crop_model_input(x_ij, x0, y0, tile_w, tile_h, self.no_crop)

                scenes.append(scene)
        return scenes

    def create_real_scenes(self, base_data):
        tile_h = self.im_h
        tile_w = self.im_w

        base_h = base_data['in_img'].shape[2]
        base_w = base_data['in_img'].shape[3]

        corner_h = int(base_h/10)
        corner_w = int(base_w/10)

        possible_tgt_idxs = [0,1,2,3,5,6,7,8]

        scenes = []
        for i in range(0, 200):
            x0 = random.randrange(corner_w, base_w - tile_w - corner_w)
            y0 = random.randrange(corner_h, base_h - tile_h - corner_h)

            x_ij = misc.to_device(base_data, base_data['in_img'].device)  # One batch
            tgt_idx = possible_tgt_idxs[i % len(possible_tgt_idxs)]
            x_ij = create_input_target(x_ij, tgt_idx)
            scene = crop_model_input(x_ij, x0, y0, tile_w, tile_h, self.no_crop)

            scenes.append(scene)

        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]
