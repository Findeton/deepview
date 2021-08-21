import pathlib

import numpy as np
import PIL.Image
import torch
import torch.utils.data
import torch.nn.functional
import random

import utils_dset

########################################################################################################################
def read_file_lines(filename):
    """
    Reads a text file, skips comments, and lines.
    :param filename:
    :return:
    """
    with open(filename) as f:
        lines = [l.replace('\n', '') for l in f if (len(l) > 0 and l[0] != '#')]
    return lines


########################################################################################################################
def parse_camera_lines(lines):
    """
    Parse metadata: youtube URL + cam intrinsics + extrinsics
    :param lines:
    :return:
    """
    # The first line contains the YouTube video URL.
    # Format of each subsequent line: timestamp fx fy px py k1 k2 row0 row1  row2
    # Column number:                  0         1  2  3  4  5  6  7-10 11-14 15-18
    youtube_url = lines[0]
    # record_defaults = ([['']] + [[0.0]] * 18)
    data = [[float(n) if idx > 0 else int(n) for idx, n in enumerate(l.split(' '))] for l in
            lines[1:]]  # simple parse csv by splitting by space

    # We don't accept non-zero k1 and k2.
    assert (0 == len(list([x for x in data if x[5] != 0.0 or x[6] != 0.0])))

    timestamps = [l[0] for l in data]
    intrinsics = [l[1:5] for l in data]  # tf.stack(data[1:5], axis=1)
    poses = [[l[7:11], l[11:15], l[15:19], [0., 0., 0., 1.]] for l in
             data]  # utils.build_matrix([data[7:11], data[11:15], data[15:19]])

    # In camera files, the video id is the last part of the YouTube URL, it comes
    # after the =.
    youtubeIDOffset = youtube_url.find("/watch?v=") + len('/watch?v=')
    youtube_id = youtube_url[youtubeIDOffset:]
    return {
        'youtube_id': youtube_id,
        'timestamps': timestamps,
        'intrinsics': intrinsics,
        'poses': poses,  # poses is world to camera (c_f_w o w_t_c)
    }


########################################################################################################################
class DsetRealEstate1(torch.utils.data.Dataset):
    """
    Real estate dataset
    """

    def __init__(self, dataset_path, is_valid=False, min_dist=200e3, max_dist=1500e3,
                 im_w=200, im_h=200, num_planes=10, num_views=3, max_w=600, no_crop = False):
        print(f'DsetRealEstate: dataset_path={dataset_path}, is_valid={is_valid}')
        self.is_valid = is_valid
        self.dataset_path = pathlib.Path(dataset_path)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.im_w = im_w
        self.im_h = im_h
        self.max_w = max_w
        self.max_h = None
        self.num_planes = num_planes
        self.num_views = num_views
        self.no_crop = no_crop

        metadata_path = self.dataset_path / 'RealEstate10K' / ('test' if is_valid else 'train')
        scenes = []
        for p in metadata_path.iterdir():
            lines = read_file_lines(p)
            scene = parse_camera_lines(lines)
            scenes.append(scene)

        self.scenes = [scene for scene in scenes if self.get_scene_idx(scene)]

    def __len__(self):
        return len(self.scenes)

    def get_scene_idx(self, scene):
        tss = scene['timestamps']
        n_ts = len(tss)
        if n_ts < self.num_views + 1:
            return None
        img_range = list(range(n_ts))
        ref_random_range = np.array(img_range)
        np.random.shuffle(ref_random_range)
        # No need for device here, it's faster on CPU !
        poses = [torch.tensor(p) for p in scene['poses']]

        # Run over random indices, until we find a good one
        for idx in ref_random_range:
            base_ts = tss[idx]
            # print(f'idx={idx}')
            base_pose = poses[idx]

            # enforce time constraints
            lam = lambda i: self.min_dist <= abs(base_ts - tss[i]) <= self.max_dist
            near_range = list(filter(lam, img_range))
            # check that the camera pose is different enough that the photos are different
            # so that the model will be able to infer perspective
            # Note: I'm not sure it's necessary
            near_range = [i for i in near_range if torch.nn.functional.l1_loss(base_pose, poses[i]).item() > 0.018]
            if len(near_range) >= self.num_views:
                # Once we have enough "near" views, return a few randomly chosen ones
                np.random.shuffle(near_range)
                return [int(idx), *near_range[:self.num_views]]

        return None

    def __getitem__(self, i):
        scene = self.scenes[i]

        # the 'indices' variable is an array, where the last element is the target
        # view and the rest are the input views
        # Is that so?
        if self.is_valid:
            # if this is a validation dataset, 'indexes' is just a range [0,1,2..]
            indices = list(range(self.num_views + 1))
        else:
            # select indexes so that the views are taken distanced in time
            indices = self.get_scene_idx(scene)
            assert indices is not None

        selected_metadata = []
        p_images = self.dataset_path / 'transcode' / scene['youtube_id']
        for idx in indices:
            intrinsics0 = scene['intrinsics'][idx]
            p_img = p_images / '{}.jpg'.format(scene['timestamps'][idx])
            img = PIL.Image.open(p_img).convert('RGB')
            # create intrinsic camera matrix
            # [fx fy cx cy] (normalised)
            intrinsics = utils_dset.make_intrinsics_matrix(
                img.width * intrinsics0[0], img.height * intrinsics0[1],
                img.width * intrinsics0[2], img.height * intrinsics0[3]
            )

            if self.max_h is None:
                self.max_h = int(img.height * self.max_w / img.width)
            # resize image, adjusting the intrinsic camera matrix as well
            tensor_image, new_intrinsics = utils_dset.resize_totensor_intrinsics(
                img, intrinsics, self.max_w, self.max_h)
            # Metadata
            selected_metadata.append({
                'timestamp': scene['timestamps'][idx],
                'intrinsics': new_intrinsics,
                'pose': torch.tensor(scene['poses'][idx]),
                'image': tensor_image,
            })

        ref_img = selected_metadata[0]  # this is the camera view pose we'll use to create the MPI
        src_imgs = selected_metadata[:-1]  # all the camera views that are input to the NN
        tgt_img = selected_metadata[-1]  # this is the dependent variable what the output should be

        # the list of plane depths we're going to consider for the MPI
        psv_planes = torch.tensor(utils_dset.inv_depths(1, 100, self.num_planes))
        intrinsics_final = torch.stack([m['intrinsics'] for m in src_imgs])

        res = {
            'in_img': torch.stack([m['image'] for m in src_imgs]).float(),
            'in_intrin': intrinsics_final.float(),
            'in_cfw': torch.stack([m['pose'] for m in src_imgs]).float(),
            'tgt_img': tgt_img['image'].unsqueeze(0).float(),
            'tgt_intrin': tgt_img['intrinsics'].unsqueeze(0).float(),
            'tgt_cfw': tgt_img['pose'].unsqueeze(0).float(),
            'mpi_planes': psv_planes.float(),
            'ref_intrin': ref_img['intrinsics'].float(),
            'ref_cfw': ref_img['pose'].float(),
            'ref_wfc': torch.inverse(ref_img['pose']).float(),
        }

        res['in_intrin_base'] = res['in_intrin']
        res['ref_intrin_base'] = res['ref_intrin']
        res['in_img_base'] = res['in_img']

        if not self.no_crop:
            ref_cam_z = res['mpi_planes'][res['mpi_planes'].shape[0]//2]

            if self.is_valid:
                ref_pixel_x = (self.max_w  / 2)
                ref_pixel_y = (self.max_h  / 2)
            else:
                ref_pixel_x = (self.max_w  / 2) + random.uniform(-0.25, 0.25) * (self.max_w - self.im_w)
                ref_pixel_y = (self.max_h  / 2) + random.uniform(-0.25, 0.25) * (self.max_h - self.im_h)
            
            ref_pos = (ref_pixel_x, ref_pixel_y, ref_cam_z)

            res = utils_dset.crop_scale_things(res, ref_pos, self.im_w, self.im_h, self.max_w, self.max_h)

        return res

########################################################################################################################
