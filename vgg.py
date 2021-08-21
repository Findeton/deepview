# By IT-JIM, 2021

import torch
import torch.nn.functional
import torchvision


class VGGPerceptualLoss1(torch.nn.Module):
    def __init__(self, resize=True, img_size=224, device=torch.device('cuda')):
        super(VGGPerceptualLoss1, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).to(device).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to(device).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to(device).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to(device).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate

        self.mean_const = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std_const = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        self.resize = resize
        self.img_size = img_size

    def forward(self, input, target):  # input, target
        input = input.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        input = (input - self.mean_const) / self.std_const
        target = (target - self.mean_const) / self.std_const
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(self.img_size, self.img_size), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(self.img_size, self.img_size), align_corners=False)
        x = input
        y = target
        loss = torch.nn.functional.l1_loss(x, y)
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y) / (1 + i)
        return loss
