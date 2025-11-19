from image import *
from model import Model
import torch
from torchvision.models import vgg19, VGG19_Weights

# desired size of the long side of the output image
targetSize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device_name)
device = torch.device(device_name)
torch.set_default_device(device)

content_path = "../assets/lake.jpg"
style_path = "../assets/lsd_dream_emulator.jpg"
out_dir = "../output/"

model = Model()
model.load_images(content_path, style_path, targetSize)
model.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
model.device = device
model.normalization_mean = torch.tensor([0.485, 0.456, 0.406])
model.normalization_std = torch.tensor([0.229, 0.224, 0.225])

"""
output = model.run_style_transfer(
    num_steps=500,
    style_weight=1000000,
    content_weight=1
    )

imsave(output, "../output/test.jpg")
"""


model.save_style_transfer_gif(
    out_dir + "house2.gif",
    10,
    num_steps=500,
    style_weight=10000,
    content_weight=1
    )