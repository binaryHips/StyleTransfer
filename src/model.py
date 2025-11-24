import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms

import copy

from metrics import *
from image import *

# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.detach().clone().view(-1, 1, 1)
        self.std =std.detach().clone().view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class Model:
    
    content_image = None
    
    style_image = None
    
    normalization_mean = None
    normalization_std = None
    
    cnn = None
    device = None
    silent = False
    
    def load_images(self, content_path, style_path, target_size):
        self.content_image, self.style_image = load_images(content_path, style_path, target_size)
    

    def get_style_model_and_losses(self,
                                content_layers=content_layers_default,
                                style_layers=style_layers_default):
        # normalization module
        normalization = Normalization(self.normalization_mean, self.normalization_std)

        # just in order to have an iterable access to or list of content/style
        # losses
        content_losses = []
        style_losses = []

        # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(self.content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(self.style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer_custom_input(self, input_img, num_steps=300,
                        style_weight=1000000, content_weight=1, noise_strength = 0):
        
        if not self.silent:
            print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses()

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        # We also put the model in evaluation mode, so that specific layers
        # such as dropout or batch normalization layers behave correctly.
        model.eval()
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)
        if not self.silent:
            print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image

                with torch.no_grad():
                    noise = torch.zeros(input_img.size()[0], input_img.size()[1], input_img.size()[2], input_img.size()[3], dtype=torch.float64)
                    noise = noise + (0.1**0.5)*torch.randn(input_img.size()[0], input_img.size()[1], input_img.size()[2], input_img.size()[3])
                    input_img.add(noise, alpha=noise_strength * 100.0)
                    input_img.clamp_(0, 1)
                    
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if not self.silent and run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
    
    def run_style_transfer(self, num_steps=300,
                        style_weight=1000000, content_weight=1, noise_strength=0):
        input_img = self.content_image.clone()
        
        return self.run_style_transfer_custom_input(input_img, num_steps, style_weight, content_weight, noise_strength)
    
    
    def run_style_transfer_custom_input_gif(self, input_img, gif_step=10, num_steps=300,
                                            style_weight=1000000, content_weight=1, noise_strength=0):
        
        gif_images = []
        if not self.silent:
            print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses()

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        # We also put the model in evaluation mode, so that specific layers
        # such as dropout or batch normalization layers behave correctly.
        model.eval()
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)
        if not self.silent:
            print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                
                with torch.no_grad():
                    noise = torch.zeros(input_img.size()[0], input_img.size()[1], input_img.size()[2], input_img.size()[3], dtype=torch.float64)
                    noise = noise + (0.1**0.5)*torch.randn(input_img.size()[0], input_img.size()[1], input_img.size()[2], input_img.size()[3])
                    input_img.add(noise, alpha=noise_strength)
                    input_img.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()
                input_img.add(noise, alpha=noise_strength)
                run[0] += 1
                if not self.silent and run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                
                if run[0] % gif_step == 0:
                    with torch.no_grad():
                        image = input_img.cpu().clone()
                        image.clamp_(0, 1)
                        image = image.squeeze(0)     
                        image = unloader(image)
                        gif_images.append(image)

                return style_score + content_score

            optimizer.step(closure)
        
        return gif_images

    def save_style_transfer_gif(self, out_path, gif_step=10, num_steps=300,
                        style_weight=1000000, content_weight=1, noise_strength=0):
        input_img = self.content_image.clone()
        
        
        res = self.run_style_transfer_custom_input_gif(input_img, gif_step, num_steps, style_weight, content_weight, noise_strength)
        
        res[0].save(out_path, save_all=True, append_images=res[1:], duration=100, loop=0)

    def save_cnn(self, path):
        torch.save(self.cnn, path)
    
    def load_cnn(self, path):
        self.cnn = torch.load(path)