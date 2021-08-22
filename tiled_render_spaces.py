# Tiled render for spaces dataset (try 2k too!)

import numpy as np

import cv2 as cv
import torch
import dset_spaces.dset1
import dset_realestate.dset1
import dset_blender.dset1

import misc
import utils_dset

##### LLFF

import struct
import collections
import os
import imageio
import random
import pathlib
import model_deepview

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images



def load_colmap_data(realdir, max_width=800):    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h0, w0, f0 = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    w = max_width
    ratio = float(w)/float(w0)
    h = int(h0*ratio/2)*2
    f = f0*ratio
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)


    def imread(f):
        if f.endswith('png'):
            img = imageio.imread(f, ignoregamma=True)
        else:
            img = imageio.imread(f)
        return cv.resize(img, (w, h))

    img_paths = [os.path.join(realdir, 'images', f) for f in names]
    imgs = [imread(img_path)[...,:3]/255. for img_path in img_paths]
    imgs = np.stack(imgs, -1) 

    return w2c_mats, imgs, hwf, names


def gen_color():
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

def plot_cameras(in_wfc, names):
    im_h, im_w = 480, 800
    vis = np.zeros((im_h, im_w, 3), dtype=np.uint8)

    cameras_xy = in_wfc[:, :2,3] # [16,2]
    x_min = np.min(cameras_xy[:,0])
    x_max = np.max(cameras_xy[:,0])
    x_delta = x_max - x_min
    y_min = np.min(cameras_xy[:,1])
    y_max = np.max(cameras_xy[:,1])
    y_delta = y_max - y_min

    cameras_pos = (cameras_xy - np.array([x_min, y_min])) * np.array([1./x_delta, 1./y_delta])
    cameras_pos = np.array([0.1, 0.1]) + np.array([0.7, 0.7]) * cameras_pos
    cameras_pos = cameras_pos * np.array([im_h, im_w])
    for i, cam_pos in enumerate(cameras_pos):
        color = gen_color()
        x, y = (int(cam_pos[1]), int(cam_pos[0]))
        cv.circle(vis, (x, y), 5, color, -1)
        # cv.putText(vis, f'{i}', (x + 4, y - 2), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 1)
        cv.putText(vis, f'{names[i]}', (x + 4, y - 2), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 1)
    imageio.imwrite('astronaut.jpg', vis)

def crop_model_input(inp, x0, y0, tile_w, tile_h):
    x = utils_dset.unwrap_input(inp)
    _, base_h, base_w, _ = x['in_img'].shape

    ref_pixel_x = x0 + tile_w//2
    ref_pixel_y = y0 + tile_h//2

    x['in_intrin_base'] = x['in_intrin']
    x['ref_intrin_base'] = x['ref_intrin']
    x['in_img_base'] = x['in_img']
    ref_cam_z = x['mpi_planes'][x['mpi_planes'].shape[0]//2]
    
    ref_pos = (ref_pixel_x, ref_pixel_y, ref_cam_z)

    res = utils_dset.crop_scale_things(x, ref_pos, tile_w, tile_h, base_w, base_h)
    return utils_dset.wrap_input(res)


def create_html_viewer(model, device, x, num_planes, tile_w, tile_h):
    """Infer a batch, and create an HTML viewer from the template"""
    
    # breakpoint()
    _, _, base_h, base_w, _ = x['in_img'].shape
    margin_w = tile_w // 5
    margin_h = tile_h // 5
    subtile_w = tile_w - 2 * margin_w
    subtile_h = tile_h - 2 * margin_h
    iters_w = (base_w - 2 * margin_w) // subtile_w
    iters_h = (base_h - 2 * margin_h) // subtile_h

    yi_rgba_layers = []
    for yi in range(0, iters_h):
        y0 = yi * subtile_h

        min_y = 0 if yi == 0 else margin_h
        max_y = tile_h if yi == (iters_h - 1) else (tile_h - margin_h)
        xi_rgba_layers = []
        for xi in range(0, iters_w):
            x0 = xi * subtile_w
            x_ij = crop_model_input(x, x0, y0, tile_w, tile_h)

            x_ij = misc.to_device(x_ij, device)  # One batch

            with torch.no_grad():
                out = model(x_ij)
            out = torch.sigmoid(out)
            rgba_layers_x0 = out.permute(0, 3, 4, 1, 2) # result: [batch, height, width, layers, colours]

            min_x = 0 if xi == 0 else margin_w
            max_x = tile_w if xi == (iters_w - 1) else (tile_w - margin_w)
            xi_rgba_layers.append(rgba_layers_x0[:,min_y:max_y,min_x:max_x])
        
        rgba_layers_y0 = torch.cat(xi_rgba_layers, dim = 2)
        yi_rgba_layers.append(rgba_layers_y0)

    rgba_layers = torch.cat(yi_rgba_layers, dim = 1)

    # By now we have RGBA MPI in the [0, 1] range
    # Export them to the HTML
    p_viewer = pathlib.Path('generated-html')
    if not p_viewer.exists():
        p_viewer.mkdir()
    # print(rgba_layers.shape, rgba_layers.dtype)
    for i in range(num_planes):
        layer = i
        file_path = 'generated-html/mpi{}.png'.format(("0" + str(layer))[-2:])
        img = rgba_layers[0, :, :, layer, :]
        misc.save_image(img, file_path)

    image_srcs = [misc.get_base64_encoded_image('./generated-html/mpi{}.png'.format(("0" + str(i))[-2:])) for i in
                    range(num_planes)]

    with open("./deepview-mpi-viewer-template.html", "r") as template_file:
        template_str = template_file.read()

    MPI_SOURCES_DATA = ",".join(['\"' + img_src + '\"' for img_src in image_srcs])
    template_str = template_str.replace("const mpiSources = MPI_SOURCES_DATA;",
                                        "const mpiSources = [{}];".format(MPI_SOURCES_DATA))

    with open("./generated-html/deepview-mpi-viewer.html", "w") as output_file:
        output_file.write(template_str)

##### LLFF

########################################################################################################################
def main():
    device_type = os.environ.get('DEVICE', 'cuda')
    device =  torch.device(device_type)
    dset_name = os.environ.get('DSET_NAME', 'spaces:1deterministic') 
    dset_path_spaces = os.environ.get('SPACES_PATH', '/big/workspace/spaces_dataset/')
    dset_path_re = os.environ.get('RE_PATH', '/big/workspace/real-estate-10k-run0')
    dset_path_blender = os.environ.get('BLENDER_PATH', '/big/workspace/negatives-wupi/felix-london-july-74/blender0/')
    tile_w = int(os.environ.get('TILE_W', '200'))
    tile_h = int(os.environ.get('TILE_H', '120'))
    scene_idx = int(os.environ.get('SCENE_INDEX', '1'))

    num_planes = 10
    if dset_name.startswith('spaces'):
        im_w, im_h = 800, 470
        dset = dset_spaces.dset1.DsetSpaces1(dset_path_spaces, False, 'large_4_9', tiny=True, im_w=im_w, im_h=im_h, no_crop=True)
    elif dset_name == 're:1random':
        im_w, im_h = 854, 480
        dset = dset_realestate.dset1.DsetRealEstate1(dset_path_re, False, im_w=im_w, im_h=im_h,
                                                                    num_planes=num_planes, num_views=3, max_w=im_w, no_crop = True)
    elif dset_name =="blender":
        im_w, im_h = tile_w, tile_h # recommendation? use 130, 120
        dset = dset_blender.dset1.DsetBlender(dset_path_blender, False, im_w=im_w, im_h=im_h, max_w = 800, no_crop = True)
    else:
        raise 'Not Implemented'

    dloader = torch.utils.data.DataLoader(dset, batch_size=1)
    iterator = iter(dloader)
    for i in range(scene_idx):
      x = next(iterator)

    # load model
    model = model_deepview.DeepViewLargeModel().to(device=device)
    filepath_s= os.environ.get('MODEL_PATH', './trained-models/dview.pt')
    filepath = pathlib.Path(filepath_s)
    if filepath.exists():
        model.load_state_dict(torch.load(str(filepath)))

    create_html_viewer(model, device, x, num_planes, tile_w, tile_h)


########################################################################################################################
if __name__ == '__main__':
    main()
