import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from random import shuffle
from utils import *
from image import *
import glob



# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# initialize weights for discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_dataset(src, force_size = 512, max_n = 10000):
    images = []
    n = 0
    s = len(glob.glob(src + "/*"))
    s = min(s, max_n)
    for f in glob.iglob(src + "/*"):
        try:
            if n >= max_n:
                break
            n += 1
            images.append(image_loader(Image.open(f), (force_size, force_size)))
            progress_bar(n, s, prefix = 'Loading images ', suffix = f'{n} /  {s}', length = 50)
        except:
            pass
    if (s - n != 0):
        progress_bar(n, s, prefix = 'Loading images ', suffix = f'{n} /  {s} (Could not load {s-n} images)', length = 50)
    print()
    return images
# adapted from https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def apply_gan(model, content_images, style_images, num_epochs):
    
    # Create the Discriminator (we always have one gpu here)
    netD = Discriminator(1).to(model.device)

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)
    
    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(model.cnn.parameters(), lr=lr, betas=(beta1, 0.999))
    
    G_losses = []
    D_losses = []
    
    model.silent = True
    
    for epoch in range(num_epochs):
        shuffle(style_images)
        for i in range(len(content_images)):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = style_images[i].detach().to(model.device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)

            label = torch.full((output.dim(),), real_label, dtype=torch.float, device=model.device)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            ## Train with all-fake batch
            model.content_image = content_images[i]
            model.style_image = style_images[i]
            fake = model.run_style_transfer(
                num_steps=50,
                style_weight=10000000,
                content_weight=1
                )
        
            label.fill_(fake_label)
            # Classify all fake batch with D
            
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model.cnn.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()  
            
            
            progress_bar(i, len(content_images)-1,
                         prefix = f'Epoch {epoch:<3}/{num_epochs:<3} ',
                         suffix = f'errG {errG.detach().item():.2f}, errD {errD.detach().item():.2f}',
                         length = 50)
        print()
        # Save Losses for plotting later
        G_losses.append(errG.detach().item())
        D_losses.append(errD.detach().item())
    print(" (Finished)")

def save_model(model, name):
    # Save the fine-tuned model
    torch.save(model.state_dict(), f'models/{name}.pth')
    