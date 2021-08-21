# DeepView models

import sys
import time

import numpy as np
import torch

import matplotlib.pyplot as plt

import misc
import utils_model
import utils_mem
from torch.utils.checkpoint import checkpoint_sequential


########################################################################################################################
def show_psv(psv):
    print((psv.shape, psv.min().item(), psv.max().item()))
    nl, nim, im_h, im_w, im_c = psv.shape # [10, 4, 120, 200, 3]
    # [depths, cameras, height, width, colours]
    plt.figure(figsize=(19, 6))
    for i_im in range(nim):
        for i_l in range(nl):
            plt.subplot(nim, nl, i_im * nl + i_l + 1)
            t = psv[i_l, i_im]
            plt.imshow(misc.tens2rgb(t))
            plt.axis('off')
    plt.tight_layout()
    plt.show()


########################################################################################################################
def conv_layer(ni, nf, ks, stride, transpose=False, padding=None, dilation=1):
    """Here I try to emulate fast.ai ConvLayer, currently only with ELU, and BATCHNORM !"""
    bias = False
    if padding is None:
        padding = ((ks - 1) // 2 if not transpose else 0)
    if transpose:
        conv = torch.nn.ConvTranspose2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                                        bias=bias)
    else:
        conv = torch.nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, bias=bias)
    res = torch.nn.Sequential(
        conv,
        torch.nn.ELU(),
        torch.nn.BatchNorm2d(nf),
        # torch.nn.InstanceNorm2d(nf),
    )
    return res


########################################################################################################################
class ModelDeepViewBlock(torch.nn.Module):
    """
    Simple DeepView Model (the init block)

    TODO : For a complete deepview, add extra parameters to this class
    (rather than copy-pasting a new nearly-identical class),
    rename it to "ModelDeepViewBlock", and also create a ModelDeepViewFull wrapper class
    """
    def __init__(self, nchan=8, iteration_num = 0, is_final_iteration = False, plot_mem = False):
        """Decrease nchan to speed things up and save GPU RAM, if needed"""
        super().__init__()

        self.mem_log = []
        self.hr = []
        self.plot_mem = utils_mem.plot_mem
        exp = f'exp_{len(self.mem_log)}'

        self.iteration_num = iteration_num

        if self.iteration_num == 1:
            self.pre_cnn = torch.nn.Sequential(
                conv_layer(nchan, 2 * nchan, ks=1, stride=1),
                utils_model.SpaceToDepth(2),
            )
            self.upsampler = torch.nn.Upsample(scale_factor=2,mode='bilinear')
            if self.plot_mem:
                utils_mem._add_hooks_to_sequential_model(self.pre_cnn, self.mem_log, exp, self.hr)
        elif self.iteration_num > 1:
            self.upsampler =  torch.nn.Identity()


        cnn1_in_channels = 14 if self.iteration_num > 0 else 3

        self.cnn1 = torch.nn.Sequential(
            conv_layer(cnn1_in_channels, 2 * nchan, ks=3, stride=1),
            utils_model.SpaceToDepth(2),
            conv_layer(8 * nchan, 4 * nchan, ks=3, stride=1),
            conv_layer(4 * nchan, 8 * nchan, ks=3, stride=1),
            conv_layer(8 * nchan, 8 * nchan, ks=3, stride=1),
            conv_layer(8 * nchan, 3 * nchan, ks=3, stride=1),
            conv_layer(3 * nchan, 12 * nchan, ks=3, stride=1),
        ) if self.iteration_num < 2 else torch.nn.Sequential(
            conv_layer(cnn1_in_channels, 4 * nchan, ks=3, stride=1),
            conv_layer(4 * nchan, 8 * nchan, ks=3, stride=1),
            conv_layer(8 * nchan, 8 * nchan, ks=3, stride=1),
            conv_layer(8 * nchan, 3 * nchan, ks=3, stride=1),
            conv_layer(3 * nchan, 12 * nchan, ks=3, stride=1),
        )
        if self.plot_mem:
            utils_mem._add_hooks_to_sequential_model(self.cnn1, self.mem_log, exp, self.hr)

        self.cnn2 = torch.nn.Sequential(
            conv_layer(24 * nchan, 12 * nchan, ks=1, stride=1),
            conv_layer(12 * nchan, 8 * nchan, ks=1, stride=1),
        )
        if self.plot_mem:
            utils_mem._add_hooks_to_sequential_model(self.cnn2, self.mem_log, exp, self.hr)

        self.cnn3 = torch.nn.Sequential(
            conv_layer(16 * nchan, 16 * nchan, ks=1, stride=1),
            conv_layer(16 * nchan, 12 * nchan, ks=1, stride=1),
        )
        if self.plot_mem:
            utils_mem._add_hooks_to_sequential_model(self.cnn3, self.mem_log, exp, self.hr)

        cnn4_param = 8 if iteration_num < 2 else 2
        cnn4_extra_in = 0 if iteration_num == 0 else (16*4 if iteration_num == 1 else 8)

        cnn4_out_dim = 4 if is_final_iteration else 8
        self.cnn4 = torch.nn.Sequential(
            conv_layer(12 * nchan + cnn4_extra_in, cnn4_param * nchan, ks=3, stride=1),
            conv_layer(cnn4_param * nchan, cnn4_param * nchan, ks=3, stride=1),
            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(cnn4_param * nchan // 4, cnn4_out_dim, kernel_size=3, padding=1, stride=1),
        ) if self.iteration_num < 2 else torch.nn.Sequential(
            conv_layer(12 * nchan + cnn4_extra_in, cnn4_param * nchan, ks=3, stride=1),
            conv_layer(cnn4_param * nchan, cnn4_param * nchan, ks=3, stride=1),
            torch.nn.Conv2d(cnn4_param * nchan, cnn4_out_dim, kernel_size=3, padding=1, stride=1),
        )
        if self.plot_mem:
            utils_mem._add_hooks_to_sequential_model(self.cnn4, self.mem_log, exp, self.hr)

    def forward(self, inp):
        # print('KEYS=', inp.keys())
        img = inp['in_img']
        batch_size, n_in, im_h, im_w, im_ch = img.shape

        # img: B x Ni x H x W x 3
        # How do we implement batching?
        # We can batch the CNN layers, no prob with that
        # PSV is currently NOT batched
        # It can be changed, but we must modify plane_sweep_torch2() to support batched ref_intrin
        # Probably not worth it
        psvs = []

        for i_batch in range(batch_size):
            pose = torch.matmul(inp['in_cfw'][i_batch], inp['ref_wfc'][i_batch].unsqueeze(0))
            #psv = utils_model.plane_sweep_torch2(img[i_batch], inp['mpi_planes'][i_batch], pose, inp['in_intrin'][i_batch],
            #                                     inp['ref_intrin'][i_batch], im_h, im_w)
            psv = utils_model.plane_sweep_torch3(inp['in_img_base'][i_batch], inp['mpi_planes'][i_batch], pose, 
                inp['in_intrin_base'][i_batch], inp['ref_intrin'][i_batch], im_h, im_w)
            # psv dimensions: D x Nin x H x W x 3 , Nin = # of input images
            psvs.append(psv)
        psv = torch.cat(psvs, dim=0)

        if False:
            show_psv(psv)
            sys.exit(0)

        # From now on, depth_batches (= D*B) replace previous depths
        # We batch over true batch, D depths and also views (=4)

        depths_batches, views, height, width, channels = psv.shape       

        if self.iteration_num > 0:
            # inp['mpi']: [batch, depths, channels=8, height, width]
            pre_in = inp['mpi'].contiguous().view(depths_batches, 8, height, width)
            if self.iteration_num == 1:
                pre_out = checkpoint_sequential(self.pre_cnn, segments=2, input=pre_in) # pre_out: [depths_batches, channels=16, height//2, width//2]
            else:
                pre_out = pre_in
            rgba_mpis = utils_model.rgba_premultiply(torch.sigmoid(self.upsampler(pre_out[:,:4])))

            mpi_gradients = []
            for i_batch in range(batch_size):
                mpi_gradient = utils_model.calculate_mpi_gradient(
                    inp['mpi'][i_batch],
                    inp['ref_wfc'][i_batch],
                    inp['mpi_planes'][i_batch],
                    inp['in_cfw'][i_batch],
                    inp['in_intrin'][i_batch],
                    inp['in_img'][i_batch],
                    inp['ref_intrin'][i_batch]
                ) # [depths, views, channels=10, height, width]         
                gradient_channels = mpi_gradient.shape[2]
                mpi_gradients.append(mpi_gradient)

            mpi_gradients = torch.cat(mpi_gradients, dim=0)[:,:,:7]
            psv_gradients = psv.permute(0, 1, 4, 2, 3)
            mpi_gradients = torch.cat([mpi_gradients, psv_gradients], dim=2)

            cnn1_in = torch.cat([
                    mpi_gradients,
                    rgba_mpis.unsqueeze(1).expand(-1, views, -1, -1, -1)
                ],
                2
            ).contiguous().view(depths_batches * views, gradient_channels + 4, height, width)
        else:
            cnn1_in = psv.permute(0, 1, 4, 2, 3).contiguous().view(depths_batches * views, channels, height, width)

        cnn1_out = checkpoint_sequential(self.cnn1, segments=4, input=cnn1_in)  # [depths*views, channels=96, height, width]

        # compute max k over all views
        _, channels, height, width = cnn1_out.shape
        maxk0 = cnn1_out.contiguous().view(depths_batches, views, channels, height, width).max(dim=1)[0] \
            .unsqueeze(1).expand(-1, views, -1, -1, -1).contiguous().view(depths_batches * views, channels, height, width)

        cnn2_in = torch.cat([cnn1_out, maxk0], dim=1)
        cnn2_out = checkpoint_sequential(self.cnn2, segments=2, input=cnn2_in) 

        # compute max k over all views
        _, channels, height, width = cnn2_out.shape
        maxk1 = cnn2_out.contiguous().view(depths_batches, views, channels, height, width).max(dim=1)[0].unsqueeze(1) \
            .expand(-1, views, -1, -1, -1).contiguous().view(depths_batches * views, channels, height, width)
        cnn3_in = torch.cat([cnn2_out, maxk1], dim=1)
        cnn3_out = checkpoint_sequential(self.cnn3, segments=2, input=cnn3_in) 

        _, channels, height, width = cnn3_out.shape
        maxk2 = cnn3_out.contiguous().view(depths_batches, views, channels, height, width).max(dim=1)[0]

        # => For further DeepView blocks, cat() additional inputs Mn just here, and no TanH please !

        if self.iteration_num > 0:
            cnn4_in = torch.cat([maxk2, pre_out], 1)
        else:
            cnn4_in = maxk2
        # The final CNN, no more Nin images
        cnn4_out = checkpoint_sequential(self.cnn4, segments=3, input=cnn4_in) 

        # Turn depths_batches back to B x D
        _, channels, height, width = cnn4_out.shape
        cnn4_out = cnn4_out.view(batch_size, depths_batches // batch_size, channels, height, width)

        return cnn4_out

########################################################################################################################

from os import path

class DeepViewLargeModel(torch.nn.Module):
  def __init__(self, plot_mem = False):
    super().__init__()
    self.model0 = ModelDeepViewBlock(iteration_num = 0, plot_mem = plot_mem)
    self.model1 = ModelDeepViewBlock(iteration_num = 1, plot_mem = plot_mem)
    self.model2 = ModelDeepViewBlock(iteration_num = 2, is_final_iteration = True, plot_mem = plot_mem)

  def forward(self, input):
    m0 = self.model0(input)

    in1 = {'mpi': m0, **input}
    gradient0 = self.model1(in1)
    m1 = m0 + gradient0

    in2 = {'mpi': m1, **input}
    gradient1 = self.model2(in2)
    m2 = m1[:,:,:4] + gradient1

    return m2