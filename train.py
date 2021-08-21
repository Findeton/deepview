# Train DeepView!

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data

import misc
import trainer_deepview
import os


########################################################################################################################
def main():
    print('train.py')

    do_training = os.environ.get('TRAIN', 'True') == 'True'

    dset_name = os.environ.get('DSET_NAME', 'spaces:1deterministic') 
    #dset_options = dict(tiny=True, layout='large_4_9')
    # dset_name = 're:1random'
    dset_options = {}

    dset_path_spaces = os.environ.get('SPACES_PATH', '/big/workspace/spaces_dataset/')
    dset_path_re = os.environ.get('RE_PATH', '/big/workspace/real-estate-10k-run0')
    dset_path_blender = os.environ.get('BLENDER_PATH', '/big/workspace/negatives-wupi/felix-london-july-74/blender0/')
    
    if dset_name.startswith('spaces'):
        dset_path = dset_path_spaces
    elif dset_name == 're:1random':
        dset_path = dset_path_re
    elif dset_name =="blender":
        dset_path = dset_path_blender
    
    device_type = os.environ.get('DEVICE', 'cuda')
    export_path = os.environ.get('EXPORT_MODEL', False)
    device = torch.device(device_type)
    batch_size = 1
    lr = 1e-4
    epochs = 1
    im_w, im_h = 200, 120

    trainer = trainer_deepview.TrainerDeepview(dset_dir=dset_path,
                                               dset_name=dset_name,
                                               dset_options=dset_options,
                                               device=device,
                                               lr=lr,
                                               batch_size=batch_size,
                                               im_w=im_w,
                                               im_h=im_h,
                                               borders=(im_w // 5, im_h // 5))
    
    # try loading the model
    trainer.load_model()

    if do_training:   # Train or load ?
        trainer.train_loop(n_epoch=epochs)
    if export_path:
        trainer.save_model(export_path)

    #trainer.demo_draw()
    trainer.create_html_viewer()

    # print('VAL LOSS=', trainer.val())


########################################################################################################################
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    main()
