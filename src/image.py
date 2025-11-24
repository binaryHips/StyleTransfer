import torch
import torch.nn.functional as F

from PIL import Image

import torchvision.transforms as transforms


def image_loader(image, imsize):
    loader = transforms.Compose([
        # transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    
    image = image.resize(imsize).convert('RGB')
    
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(torch.get_default_device(), torch.float)

def load_images(content_path, style_path, target_size):
    
    content = Image.open(content_path)
    style = Image.open(style_path)
    
    max_coord = max(content.width, content.height)
    
    fac = max_coord / float(target_size)
    
    imsize = (int(content.width / fac), int(content.height / fac))
    
    content_tensor = image_loader(content, imsize)
    style_tensor = image_loader(style, imsize)
    
    return content_tensor, style_tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image
def imsave(tensor, path):
    image = tensor.cpu().clone()
    image = image.squeeze(0)     
    image = unloader(image)
    image.save(path)
