"""
Reimplementation from scratch of the MNIST Experiments from 
``Pixel Recurrent Neural Networks``
https://arxiv.org/pdf/1601.06759.pdf

Author: Paul Wilson
"""

import torch 
import einops
from simple_parsing import parse, ArgumentParser
from tqdm import tqdm
from rich import print as pprint
from dataclasses import dataclass
import matplotlib.pyplot as plt 
import wandb 
from os.path import join 
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class Args: 
    exp_dir: str = 'logs'

    n_epochs: int = 100
    lr: float = 1e-4

    sample_image_every_n_epochs: int = 10
    temperature: float = 1

    n_layers: int = 4

    batch_size: int = 32 
    limit_batches: int | None = None 
    

def train(args: Args):
    os.makedirs(args.exp_dir, exist_ok=True)

    # wandb.init(project='image-autoregression', name='pixelrnn_debug')
    train_loader, eval_loader = create_dataset(args)
    sample_batch = next(iter(train_loader))

    from models import PixelRNN
    model = PixelRNN(in_channels=1, out_channels=1, hidden_dim = 32, n_layers=args.n_layers).to(DEVICE)
    from torchinfo import summary
    summary(model, input_data=sample_batch[0].to(DEVICE))

    from torch.optim import Adam 
    optimizer = Adam(model.parameters(), lr=args.lr)
    # wandb.watch(model, log='all', log_freq=1)

    for epoch in range(1, args.n_epochs): 
        print(f"======== EPOCH {epoch} ========")
        
        if epoch % args.sample_image_every_n_epochs == 0: 
            print('Sampling images: ')
            n_samples = 8
            
            fig, ax = plt.subplots(4, n_samples)
            for ax_ in ax.flatten(): 
                ax_: plt.Axes 
                ax_.set_axis_off()

            prompts = torch.stack([eval_loader.dataset[i][0] for i in range(n_samples)])

            for row, start_rowpos in enumerate([0, 4, 8, 16]): 
                samples = generate_examples(model, prompts=prompts, startpos=(start_rowpos, 0))
                for i in range(n_samples): 
                    sample = samples[i][0].numpy()
                    ax[row, i].imshow(sample)

            fig.tight_layout()
            plt.savefig(f'logs/epoch{epoch}.png')
            plt.close()

        train_metrics = run_epoch(args, model, train_loader, optimizer)
        val_metrics = run_epoch(args, model, eval_loader)
        pprint({
            'train': train_metrics, 
            'val': val_metrics
        })


@torch.no_grad()
def generate_examples(model, prompts, startpos=(0, 0), temperature=1): 
    B, C, H, W = prompts.shape 
    image = prompts.to(DEVICE)

    i_0, j_0 = startpos

    for i in range(i_0, H):
        for j in range(W):
            if i == i_0 and j < j_0: 
                continue  
            # breakpoint() 
            pixel_logits = model(image)
            pixel_logits = pixel_logits / temperature
            pixel_probs = pixel_logits.sigmoid()
            # find the correct probability distribution 
            prob = pixel_probs[:, 0, i, j]
            # sample the probability distribution
            dice = torch.rand((B,), device=DEVICE)
            pixel_sample = (dice < prob).long().float()
            # we need to scale this and make it into a float 
            # now we can set the corresponding pixel value in the image 
            image[:, 0, i, j] = pixel_sample 
            
    
    return image.cpu()


def run_epoch(args, model, loader, optimizer=None): 
    training = optimizer is not None 
    model.train() if training else model.eval()

    with torch.enable_grad() if training else torch.no_grad():
        loss_epoch = 0
        total_items = 0 
        for batch in tqdm(loader, leave=False): 
            # breakpoint()
            img, *_ = batch
            img = img.to(DEVICE)
            
            B, C, H, W = img.shape
            pixel_probs = model(img.to(DEVICE))
            pixel_probs = einops.rearrange(
                pixel_probs, 'b c h w -> (b h w) c',
            )
            pixel_probs = pixel_probs.squeeze(-1).sigmoid()

            target = img
            target = einops.rearrange(target, 'b c h w -> (b h w) c')
            target = target.squeeze(-1)
            
            loss_step = torch.nn.functional.binary_cross_entropy(pixel_probs, target, reduction='sum')
            total_items += B 

            if training: 
                loss_step.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss_epoch += loss_step.item()

    return {
        'nll': loss_epoch / total_items,
    }


def create_dataset(args: Args): 
    from torchvision.datasets import MNIST 
    from torchvision import transforms as T

    def transform(img): 
        img = T.ToTensor()(img)
        img = (img > 0.5).float() 
        return img

    train_ds = MNIST(root='data/mnist', download=True, train=True, transform=transform)
    val_ds = MNIST(root='data/mnist', download=True, train=False, transform=transform)

    if args.limit_batches: 
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(args.limit_batches * args.batch_size))
        val_ds = Subset(val_ds, range(args.limit_batches * args.batch_size))

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False
    )

    return train_loader, val_loader


def add_arguments(parser=None):
    parser = parser or ArgumentParser()
    parser.add_arguments(Args, dest='train')
    return parser 


if __name__ == "__main__": 
    parser = add_arguments()
    args = parser.parse_args()
    train(args.train)
