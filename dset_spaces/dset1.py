
import sys

import numpy as np
import torch
import random

from . import base
import utils_dset

# layout format : (idx_in, idx_tgt)
# Note "idx" here means "idx_cam", the camera index 0..15
# Names: name_Nin_Nout
# idx_in = indices of input images
# idx_tgt = indices of output images

STANDARD_LAYOUTS = {
    'large_4_1': ([0, 3, 9, 12], [6]),
    'large_4_9': ([0, 3, 9, 12], [1, 2, 4, 5, 6, 7, 8, 10, 11]),
    'dummy_5_2': ([0, 1, 2, 3, 4], [5, 6]),  # This is just to test size=5 input, don't use this for real
}


#######################################################################################################################
class DsetSpaces1(base.AbstractDsetSpaces):
    def __init__(self, dataset_path, is_val=True, layout='large_4_1', n_planes=10, tiny=False, im_w=200, im_h=120, no_crop=False):
        """
        The "standard" fully deterministic (no randomness) Spaces dataset
        Treat each rig position of each scene as an element, with non-random input and target views
        Different rig positions are not mixed
        For layouts, see Fig.4 of the paper

        :param dataset_path:   Spaces dataset path on disk
        :param is_val:         True=val, False=train
        :param layout:
        """
        super().__init__(dataset_path, is_val, n_planes, tiny, im_w, im_h)
        self.no_crop = no_crop
        if isinstance(layout, str):
            tup = STANDARD_LAYOUTS[layout]
        elif isinstance(layout, tuple):
            tup = layout
        else:
            raise ValueError('DsetSpaces1(): layout must be a string or 2-tuple !')
        self.idxs_in, self.idxs_tgt = tup
        self.n_in, self.n_tgt = len(self.idxs_in), len(self.idxs_tgt)
        assert self.n_in > 0 and self.n_tgt > 0

    def __len__(self):
        return len(self.rig_table)

    def __getitem__(self, item):
        # Add scene and rig to camera indices
        idx_scene, idx_rig = self.rig_table[item]
        idxs_in_full = [(idx_scene, idx_rig, i) for i in self.idxs_in]
        idxs_tgt_full = [(idx_scene, idx_rig, i) for i in self.idxs_tgt]
        res = self._get_elem(idxs_in_full, idxs_tgt_full)
        # Add ref image data
        self._add_ref((idx_scene, idx_rig, 6), res)

        res['in_intrin_base'] = res['in_intrin']
        res['ref_intrin_base'] = res['ref_intrin']
        res['in_img_base'] = res['in_img']

        if not self.no_crop:
            ref_cam_z = res['mpi_planes'][res['mpi_planes'].shape[0]//2]

            if self.is_val:
                ref_pixel_x = (self.image_base_width  / 2)
                ref_pixel_y = (self.image_base_height  / 2)
            else:
                ref_pixel_x = (self.image_base_width  / 2) + random.uniform(-0.25, 0.25) * (self.image_base_width - self.im_w)
                ref_pixel_y = (self.image_base_height  / 2) + random.uniform(-0.25, 0.25) * (self.image_base_height - self.im_h)
            
            ref_pos = (ref_pixel_x, ref_pixel_y, ref_cam_z)
 
            res = utils_dset.crop_scale_things(res, ref_pos, self.im_w, self.im_h, self.image_base_width, self.image_base_height)

        return res

#######################################################################################################################
