# Abstract parent for spaces dataset: loads spaces metainfo, but does nothing else

import pathlib
import os

import PIL
import torch
import torch.utils.data

import utils_dset
import spaces_dataset.code.utils


#######################################################################################################################
class AbstractDsetSpaces(torch.utils.data.Dataset):
    """Abstract Spaces dataset, loads metadata, and contains some useful routines"""

    def __init__(self, dataset_path, is_val=True, n_planes=10, tiny=False, im_w=200, im_h=120):
        self.dataset_path = dataset_path
        self.is_val = is_val
        self.n_planes = n_planes
        self.tiny = tiny

        self.n_in = None
        self.n_tgt = None
        self.im_w = im_w
        self.im_h = im_h
        self.image_base_width = 800
        self.image_base_height = 480

        use_2k = False

        # Load metadata
        metadata_base_path = pathlib.Path(dataset_path) / 'data' / ('2k' if use_2k else '800')
        dirs = os.listdir(metadata_base_path)
        dirs.sort()
        scenes = [spaces_dataset.code.utils.ReadScene(metadata_base_path / p) for p in dirs]
        if not use_2k:
            assert len(scenes) == 100
            if tiny:
                scenes = scenes[:2] if is_val else scenes[10:20]
            else:
                scenes = scenes[:10] if is_val else scenes[10:]

        self.scenes = scenes

        # Create the rig total index table, which might be useful
        # i = scene index, j = rig pos in scene
        self.rig_table = []
        n = len(scenes)
        for i in range(n):
            for j in range(len(scenes[i])):
                self.rig_table.append((i, j))
        # Create PSV planes
        self.psv_planes = torch.tensor(utils_dset.inv_depths(1, 100, self.n_planes))

    def _get_elem(self, idxs_in, idxs_tgt):
        """
        Get dataset element for given input and target indices
        each index is tuple (idx_scene, idx_rig, idx_cam)
        Does not add "ref" data!
        :param idxs_in:        Input indices
        :param idxs_tgt:       Target indices
        :return:
        """
        assert self.n_in == len(idxs_in)
        assert self.n_tgt == len(idxs_tgt)
        # Load all images from 2 lists
        images_in = [self._load_image(i) for i in idxs_in]
        images_tgt = [self._load_image(i) for i in idxs_tgt]
        # Create the result dict
        res = {
            'in_img': torch.stack([m['img'] for m in images_in]).float(),
            'in_intrin': torch.stack([m['intrin'] for m in images_in]).float(),
            'in_cfw': torch.stack([m['cfw'] for m in images_in]).float(),
            'tgt_img': torch.stack([m['img'] for m in images_tgt]).float(),
            'tgt_intrin': torch.stack([m['intrin'] for m in images_tgt]).float(),
            'tgt_cfw': torch.stack([m['cfw'] for m in images_tgt]).float(),
            'mpi_planes': self.psv_planes.float(),
        }
        return res

    def _load_image(self, idx):
        #print(f"fff {str(idx)}")
        data = self.scenes[idx[0]][idx[1]][idx[2]]
        intrin = torch.tensor(data.camera.intrinsics, dtype=torch.float32)
        cfw = torch.tensor(data.camera.c_f_w, dtype=torch.float32)
        image_path = data.image_path
        img = PIL.Image.open(image_path).convert('RGB')
        img, intrin = utils_dset.resize_totensor_intrinsics(img, intrin, self.image_base_width, self.image_base_height)
        return {
            'img': img,
            'intrin': intrin,
            'cfw': cfw,
        }

    def _add_ref(self, idx, res):
        """Add a given position (full index) as ref to dict res"""
        # Currently need a full image load to get ig size, stupid ?
        img = self._load_image(idx)
        res['ref_intrin'] = img['intrin']
        res['ref_cfw'] = img['cfw']
        res['ref_wfc'] = torch.inverse(img['cfw'])

#######################################################################################################################
