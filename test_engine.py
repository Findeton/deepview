# Test engine used for model testing, currently SSIM

import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

import skimage.metrics

import misc


########################################################################################################################
class TestEngine:
    def __init__(self):
        self.photo_mode = 'ssim'
        self.scores = []
        self.scores_baseline = []

    def photo(self, img1, img2):
        """Photometric comparison of 2 RGB images"""
        if self.photo_mode == 'ssim':
            gray1 = img1.mean(axis=2).astype('uint8')
            gray2 = img2.mean(axis=2).astype('uint8')
            s = skimage.metrics.structural_similarity(gray1, gray2)
        elif self.photo_mode == 'ssim-color':
            s = skimage.metrics.structural_similarity(img1, img2, multichannel=True)
        else:
            raise ValueError(f'Wrong photo_mode={self.photo_mode}')
        return s

    def run_batch(self, x, outs, targets, borders):
        outs = misc.tens2rgb(outs)
        targets = misc.tens2rgb(targets)
        # print('OUTS', outs.shape, outs.dtype, outs.min(), outs.max())
        # print('TARGETS', targets.shape, targets.dtype, targets.min(), targets.max())
        nt = outs.shape[0]

        outs_cropped = misc.my_crop(outs, borders)
        targets_cropped = misc.my_crop(targets, borders)

        # Test outputs vs targets
        for i in range(nt):
            img1 = outs_cropped[i]
            img2 = targets_cropped[i]
            if False:  # Visualize each pair
                plt.subplot(1, 2, 1)
                plt.imshow(img1)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(img2)
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            s = self.photo(img1, img2)  # Photo score
            self.scores.append(s)

        # Baseline score: Compare input images to targets
        # A trained Neural net should perform better than this !
        inputs = misc.tens2rgb(x['in_img'])
        # print('INPUTS', inputs.shape, inputs.dtype, inputs.min(), inputs.max())
        nb, ni = inputs.shape[:2]
        # We want to run over all input images and targets,
        # but separately for each elements in a batch (if using batches)
        assert nt % nb == 0
        nt0 = nt // nb   # Targets per batch element
        for ib in range(nb):
            inputs_cropped = misc.my_crop(inputs[ib], borders)
            targets_cropped = misc.my_crop(targets[ib * nt0: (ib + 1) * nt0], borders)
            assert ni == inputs_cropped.shape[0] and nt0 == targets_cropped.shape[0]
            for ii in range(ni):
                for it in range(nt0):
                    img1 = inputs_cropped[ii]
                    img2 = targets_cropped[it]
                    s = self.photo(img1, img2)  # Photo score
                    self.scores_baseline.append(s)

    def print_stats(self):
        """Print the results"""
        print(f'PHOTO_MODE = {self.photo_mode}')
        print(f'AVG_SCORE = {np.mean(self.scores)}')
        print(f'AVG_BASELINE_SCORE = {np.mean(self.scores_baseline)}')

########################################################################################################################
