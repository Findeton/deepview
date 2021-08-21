import sys
import base64
import numpy as np
import torch
import torchvision


########################################################################################################################
def get_base64_encoded_image(image_path):
    """Utils for the HTML viewer"""
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode('utf-8')


########################################################################################################################
def to_device(x, device):
    """Cast a hierarchical object to pytorch device"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        for k in list(x.keys()):
            x[k] = to_device(x[k], device)
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return type(x)(to_device(t, device) for t in x)
    else:
        raise ValueError('Wrong type !')


########################################################################################################################
def tens2rgb(t):
    """Tensor (HWC) to RGB, [0, 1] -> [0, 255]"""
    t = t.detach().cpu().numpy()
    return (t * 255).astype('uint8')


########################################################################################################################
def save_image(img, path):
    """Utils for the HTML viewer"""
    # print('SAVING ...', img.shape, img.dtype)
    # img = torch.cat([img, img[:, :, 3:]], dim=2).permute(2, 0, 1)
    img = img.permute(2, 0, 1)
    # print('SAVING ...', img.shape, img.dtype)
    pilImg = torchvision.transforms.ToPILImage()(img)
    pilImg.save(path)


########################################################################################################################
def my_crop(t, borders):
    """Crop a tensor for photometric loss"""
    bx, by = borders
    return t[:, by:-by, bx:-bx, :]


########################################################################################################################
