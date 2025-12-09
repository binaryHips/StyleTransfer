from image import *
from model import Model
import torch
from torchvision.models import vgg19, VGG19_Weights
from finetune import *

# desired size of the long side of the output image
targetSize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device_name)
device = torch.device(device_name)
torch.set_default_device(device)

content_path = "../assets/drip.png"
style_path = "../assets/datasets/dataset_updated/training_set/engraving/122.jpg"
out_dir = "../output/"
model_souce = ""

model = Model()
model.load_images(content_path, style_path, targetSize)
model.device = device
model.normalization_mean = torch.tensor([0.485, 0.456, 0.406])
model.normalization_std = torch.tensor([0.229, 0.224, 0.225])
model.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

if model_souce != "":
    model.load_state_dict(model_souce)

finetune_pass = True
apply_pass = False
apply_gif_pass = False

## Fine tune using gan
if finetune_pass:
    style_dataset = load_dataset("../assets/datasets/dataset_updated/training_set/engraving", 64, 600)
    content_dataset = load_dataset("../assets/datasets/data/human", 64, 600)

    apply_gan(model, content_dataset, style_dataset, 100, path_of_backups= "../models/engravings", bench_content=content_path, bench_style= style_path)

    model.save_cnn("../models/vgg_finetuned_finished.pth")
    
if apply_pass:
    output = model.run_style_transfer(
    num_steps=700,
    style_weight=100000000,
    content_weight=1
    )
    
    imsave(output, "../output/timodog.jpg")

if apply_gif_pass:

    model.save_style_transfer_gif(
        out_dir + "jsp.gif",
        20,
        num_steps=1000,
        style_weight=1000000,
        content_weight=1,
        noise_strength=0.5
        )