
import sys
import pathlib
import time

import tqdm
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import matplotlib.pyplot as plt

import dset_spaces.dset1
import dset_realestate.dset1
import dset_blender.dset1
import model_deepview
import vgg
import misc
import utils_render
import test_engine


########################################################################################################################
class TrainerDeepview:
    """
    Trainer class for "mini_deep_view_no_lgd"
    """

    def __init__(self, dset_dir, dset_name, dset_options={}, device=torch.device('cuda'), lr=1.e-3, batch_size=1,
                 im_w=200, im_h=200, borders=(0, 0)):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.num_planes = 10
        self.num_workers = 4
        self.borders = borders
        # Model
        self.model = model_deepview.DeepViewLargeModel().to(device=device)
        # VGG loss
        self.vgg_loss = vgg.VGGPerceptualLoss1(resize=False, device=device)
        # Dataset+loaders
        print(
            f'TrainerDeepview: dset_dir={dset_dir}, dset_name={dset_name}, dset_options={dset_options}, device={device}')
        if dset_name == 'spaces:1deterministic':
            self.dset_train = dset_spaces.dset1.DsetSpaces1(dset_dir, False, im_w=im_w, im_h=im_h, **dset_options)
            self.dset_val = dset_spaces.dset1.DsetSpaces1(dset_dir, True, im_w=im_w, im_h=im_h, **dset_options)
        elif dset_name == 're:1random':
            self.dset_train = dset_realestate.dset1.DsetRealEstate1(dset_dir, False, im_w=im_w, im_h=im_h,
                                                                    **dset_options)
            self.dset_val = dset_realestate.dset1.DsetRealEstate1(dset_dir, True, im_w=im_w, im_h=im_h, **dset_options)
        elif dset_name =="blender":
            self.dset_train = dset_blender.dset1.DsetBlender(dset_dir, False, im_w=im_w, im_h=im_h,
                                                                    **dset_options)
            self.dset_val = dset_blender.dset1.DsetBlender(dset_dir, True, im_w=im_w, im_h=im_h, **dset_options)
        else:
            raise ValueError(f'Wrong dset_name={dset_name} !')

        print(f'Datasets : train: {len(self.dset_train)},  val: {len(self.dset_val)}')
        self.loader_train = torch.utils.data.DataLoader(self.dset_train, batch_size=self.batch_size,
                                                        num_workers=self.num_workers, shuffle=True)
        self.loader_val = torch.utils.data.DataLoader(self.dset_val, batch_size=self.batch_size,
                                                      num_workers=self.num_workers)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if False:
            b = next(iter(self.loader_train))
            print(('KEYS=', list(b.keys())))
            for k, v in list(b.items()):
                print(f'{k} : {v.shape} {v.dtype}')
            sys.exit(0)

    def process_one(self, out, x):
        """Process and render one network output+target"""
        out = torch.sigmoid(out)
        rgba_layers = out.permute(0, 3, 4, 1, 2)

        # print('SHAPE=', rgba_layers.shape)
        # print(f'MIN={rgba_layers.min().item()}, MAX={rgba_layers.max().item()}')
        n_targets = x['tgt_img'].shape[1]


        # t1 = time.time()
        batch_size = rgba_layers.shape[0]
        # i_batch = 0  # Replace later with a loop over batch
        out_images_batch = []
        # Can we fully batch the second version, I wonder?
        for i_batch in range(batch_size):
            if False:
                # Version with a loop over targets
                outs = []
                for i_target in range(n_targets):
                    rel_pose = torch.matmul(x['tgt_cfw'][i_batch, i_target, :, :], x['ref_wfc'][i_batch]).unsqueeze(0)
                    intrin_tgt = x['tgt_intrin'][i_batch, i_target, :, :].unsqueeze(0)
                    intrin_ref = x['ref_intrin'][i_batch].unsqueeze(0)
                    out_image = utils_render.mpi_render_view_torch(rgba_layers, rel_pose, x['mpi_planes'][i_batch],
                                                                   intrin_tgt, intrin_ref)
                    outs.append(out_image)
                out_images = torch.cat(outs, 0)
            else:
                # Version batched over targets, but we have to repeat a relatively large tensor rgba, which one is better?
                # Results are very close (but not to machine precision !)
                rel_pose = torch.matmul(x['tgt_cfw'][i_batch], x['ref_wfc'][i_batch])
                intrin_tgt = x['tgt_intrin'][i_batch]
                intrin_ref = x['ref_intrin'][i_batch].unsqueeze(0).repeat(n_targets, 1, 1)
                rgba = rgba_layers[i_batch].unsqueeze(0).repeat(n_targets, 1, 1, 1, 1)
                
                out_images = utils_render.mpi_render_view_torch(rgba, rel_pose, x['mpi_planes'][i_batch], intrin_tgt, intrin_ref)

            # t2 = time.time()
            # print('TIME RENDER', t2-t1)
            out_images_batch.append(out_images)
        out_images_batch = torch.cat(out_images_batch, dim=0)
        targets = x['tgt_img']
        targets = targets.reshape(batch_size * n_targets, *targets.shape[2:])
        return out_images_batch, targets

    def loss(self, out, x):
        """Calculate VGG loss"""
        output_image, target = self.process_one(out, x)
        output_image = misc.my_crop(output_image, self.borders)
        target = misc.my_crop(target, self.borders)
        loss = self.vgg_loss(output_image, target)
        return loss

    def val(self):
        """Validate, 1 run"""
        self.model.eval()
        loss_sum = 0
        with torch.no_grad():
            for batch in self.loader_val:
                x = misc.to_device(batch, self.device)
                out = self.model(x)
                loss = self.loss(out, x)
                loss_sum += loss.item()
        return loss_sum / len(self.loader_val)

    def train(self):
        """Train 1 epoch"""
        self.model.train()
        loss_sum = 0
        for batch in tqdm.tqdm(self.loader_train):
            x = misc.to_device(batch, self.device)
            # t1 = time.time()
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss(out, x)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            # t2 = time.time()
            # print('TIME TRAIN RUN', t2-t1)
        return loss_sum / len(self.loader_train)

    def train_loop(self, n_epoch=10):
        """Train the model"""
        for i_epoch in range(n_epoch):
            t1 = time.time()
            loss_train = self.train()
            loss_val = self.val()
            t2 = time.time()
            print(
                f'\nEpoch {i_epoch}/{n_epoch} : loss_train = {loss_train}, loss_val = {loss_val}, time = {t2 - t1:6.2f}s')
        # Save the trained model
        self.save_model()

    def save_model(self, filepath_s='./trained-models/dview.pt'):
        filepath = pathlib.Path(filepath_s)
        if not filepath.parent.exists():
            filepath.parent.mkdir()
        torch.save(self.model.state_dict(), str(filepath))

    def load_model(self, filepath_s='./trained-models/dview.pt'):
        filepath = pathlib.Path(filepath_s)
        if filepath.exists():
            self.model.load_state_dict(torch.load(str(filepath)))

    def demo_draw(self):
        """Draw predictions vs targets as a test"""
        self.model.eval()
        batch = next(iter(self.loader_val))
        x = misc.to_device(batch, self.device)
        with torch.no_grad():
            out = self.model(x)
        t1, t2 = self.process_one(out, x)
        nt = t1.shape[0]
        nt = min(nt, 1)
        for i in range(nt):
            plt.subplot(nt, 2, 2 * i + 1)
            plt.imshow(misc.tens2rgb(t1[i]))
            plt.axis('off')
            plt.subplot(nt, 2, 2 * i + 2)
            plt.imshow(misc.tens2rgb(t2[i]))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def create_html_viewer(self, scene_idx=0):
        """Infer a batch, and create an HTML viewer from the template"""
        self.model.eval()

        # Choose your scene_idx from the val set !
        # Dataloaders like this adds batch dimension to a chosen element !
        scene_loader = torch.utils.data.DataLoader(self.dset_val, batch_sampler=[[scene_idx]])
        x = misc.to_device(next(iter(scene_loader)), self.device)  # One batch

        with torch.no_grad():
            out = self.model(x)
        out = torch.sigmoid(out)
        rgba_layers = out.permute(0, 3, 4, 1, 2)

        # By now we have RGBA MPI in the [0, 1] range
        # Export them to the HTML
        p_viewer = pathlib.Path('generated-html')
        if not p_viewer.exists():
            p_viewer.mkdir()
        # print(rgba_layers.shape, rgba_layers.dtype)
        for i in range(self.num_planes):
            layer = i
            file_path = 'generated-html/mpi{}.png'.format(("0" + str(layer))[-2:])
            img = rgba_layers[0, :, :, layer, :]
            misc.save_image(img, file_path)

        image_srcs = [misc.get_base64_encoded_image('./generated-html/mpi{}.png'.format(("0" + str(i))[-2:])) for i in
                      range(self.num_planes)]

        with open("./deepview-mpi-viewer-template.html", "r") as template_file:
            template_str = template_file.read()

        MPI_SOURCES_DATA = ",".join(['\"' + img_src + '\"' for img_src in image_srcs])
        template_str = template_str.replace("const mpiSources = MPI_SOURCES_DATA;",
                                            "const mpiSources = [{}];".format(MPI_SOURCES_DATA))

        with open("./generated-html/deepview-mpi-viewer.html", "w") as output_file:
            output_file.write(template_str)

    def test(self):
        """Test on the val set with the test engine (SSIM)"""
        self.model.eval()
        engine = test_engine.TestEngine()
        with torch.no_grad():
            for batch in tqdm.tqdm(self.loader_val):
                x = misc.to_device(batch, self.device)
                out = self.model(x)
                outs, targets = self.process_one(out, x)
                engine.run_batch(x, outs, targets, self.borders)
        engine.print_stats()


########################################################################################################################
