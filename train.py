"""
Reimplementation from scratch of the natural image generation experiments from 
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
import typing as t
import numpy as np 
import os 


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class Args: 
    exp_name: str = 'experiment'

    seed: int = 0
    n_epochs: int = 100
    lr: float = 1e-4

    sample_freq: int = 10
    temperature: float = 1
    occlusions: tuple[int] = (4, 8, 10, 16)

    n_layers: int = 4
    residual: bool = True
    hidden_features_per_channel: int = 32

    batch_size: int = 32 
    limit_batches: int | None = None 
    
    dataset: t.Literal['cifar10', 'imagenet_32', 'imagenet_64'] = 'cifar10'
    exp_dir: str | None = None

    wandb: bool = True
    wandb_id: str | None = None 


def train(args: Args):
    from datetime import datetime 

    # WORKDIR

    if args.exp_dir is None: 
        args.exp_dir = join('logs', args.exp_name, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    if 'SLURM_JOB_ID' in os.environ: 
        if not os.path.exists(args.exp_dir): # could exist if resuming a run 
            os.makedirs(os.path.dirname(args.exp_dir), exist_ok=True)
            os.symlink(
                join(
                    '/checkpoint', 
                    os.environ['USER'], 
                    os.environ['SLURM_JOB_ID']
                ), 
                args.exp_dir, 
                target_is_directory=True
            )
    else: 
        os.makedirs(args.exp_dir, exist_ok=True) 
    
    # LOGGING

    if args.wandb: 
        run = wandb.init(project='image-autoregression', name=args.exp_name, config=vars(args), id=args.wandb_id)
        log = wandb.log 
        args.wandb_id = run.id
    else: 
        log = lambda d: pprint(d)

    if 'experiment_checkpoint.pt' in os.listdir(args.exp_dir): 
        state_dict = torch.load(
            join(
                args.exp_dir, 
                'experiment_checkpoint.pt'
            )
        )
    else: 
        state_dict = None 

    # DATASET

    train_loader, eval_loader = create_dataset(args)
    sample_batch = next(iter(train_loader))

    # MODEL

    from models import PixelRNN
    model = PixelRNN(in_channels=3, out_channels=256 * 3, hidden_dim = args.hidden_features_per_channel * 3, n_layers=args.n_layers).to(DEVICE)
    if state_dict: 
        model.load_state_dict(state_dict['model'])

    from torchinfo import summary
    summary(model, input_data=sample_batch[0].to(DEVICE))

    # OPTIMIZER

    from torch.optim import Adam 
    optimizer = Adam(model.parameters(), lr=args.lr)
    if state_dict: 
        optimizer.load_state_dict(state_dict['optimizer'])

    start_epoch = state_dict['epoch'] if state_dict else 1 
    if state_dict: 
        torch.random.set_rng_state(state_dict['rng'])

    for epoch in range(start_epoch, args.n_epochs): 

        print(f"======== EPOCH {epoch} ========")
        
        if epoch % args.sample_freq == 0: 
            print('Sampling images: ')
            for split, loader in {
                'train': train_loader, 
                'val': eval_loader
            }.items():
                n_samples = 8 if len(loader.dataset) >= 8 else len(loader.dataset)

                fig, ax = plt.subplots(len(args.occlusions), n_samples)
                for ax_ in ax.flatten(): 
                    ax_: plt.Axes 
                    ax_.set_axis_off()

                prompts = torch.stack([loader.dataset[i][0] for i in range(n_samples)])

                for row, start_rowpos in enumerate(args.occlusions): 
                    samples = generate_examples(model, prompts=prompts, startpos=(start_rowpos, 0))
                    # samples shape is b x 3 x h x w long tensor: convert to b x h x w x 3 numpy uint8
                    samples = samples.numpy().transpose(0, 2, 3, 1).astype(np.uint8)
                    for i in range(n_samples): 
                        sample = samples[i]
                        ax[row, i].imshow(sample)

                fig.tight_layout()
                plt.savefig(join(args.exp_dir, f'{split}_epoch{epoch}_samples.png'))
                plt.close()
                if args.wandb: 
                    wandb.log(
                        {f'{split}_samples': wandb.Image(join(args.exp_dir, f'{split}_epoch{epoch}_samples.png')), 
                            'epoch': epoch}
                    )

        train_metrics = run_epoch(args, model, train_loader, optimizer)
        val_metrics = run_epoch(args, model, eval_loader)

        torch.save(model.state_dict(), join(args.exp_dir, f'epoch_{epoch}.pt'))

        log({
            **{f'train_{k}': v for k, v in train_metrics.items()}, 
            **{f'val_{k}': v for k, v in val_metrics.items()}, 
            'epoch': epoch 
        })

        checkpoint(args, model, optimizer, epoch + 1)


@torch.no_grad()
def generate_examples(model, prompts, startpos=(0, 0), temperature=1): 
    """
    Model: floating point pixel values tensor, shape B, 3, H, W -> floating point pixel logits tensor, 
        B, 256 * 3, H, W
    Prompts: 
        floating points pixel values tensor, shape B, 3, H, W
    Startpos: int, int, the pixel to start filling in at
    """
    
    # prompts will be floating point values between 0 and 1 which is the model's
    # expected input
    B, C, H, W = prompts.shape
    prompts = prompts.to(DEVICE)
    image = prompts
    image = (image * 255).long()

    i_0, j_0 = startpos

    for i in range(i_0, H):
        for j in range(W): 
            if i == i_0 and i < j_0: 
                break 
            else: 
                # zero out the pixels we need to fill in to avoid bugs later
                image[:, :, i, j] = 0
                prompts = (image / 255).float()

    pbar = tqdm(desc='Sampling pixels', total=H * W * 3)

    for i in range(i_0, H):
        for j in range(W):
            if i == i_0 and j < j_0: 
                continue  
            for c in range(3):
                pixel_logits = model(prompts)
                channel_logits = torch.split(pixel_logits, pixel_logits.shape[1] // 3, dim=1)
                pixel_logits = channel_logits[c]
                pixel_logits = pixel_logits / temperature
                pixel_probs = pixel_logits.softmax(1)

                # find the correct probability distribution 
                prob = pixel_probs[:, :, i, j] # shape B x 256 

                # sample the probability distribution
                pixel_samples = torch.multinomial(prob, num_samples=1).squeeze(-1) # B,
                
                # we need to scale this and make it into a float 
                # now we can set the corresponding pixel value in the image 
                image[:, c, i, j] = pixel_samples  
                # we also need to replace the pixel into the floating point 
                # version of the tensor for next model input
                prompts = (image / 255).float()

                pbar.update(1)
            
    return image.cpu()


def run_epoch(args, model, loader, optimizer=None): 
    training = optimizer is not None 
    model.train() if training else model.eval()

    with torch.enable_grad() if training else torch.no_grad():
        loss_epoch = 0
        total_items = 0 

        for batch in tqdm(loader, leave=False): 
            img, *_ = batch
            img = img.to(DEVICE)
            
            B, C, H, W = img.shape
            pixel_probs = model(img)
            
            pixel_probs = einops.rearrange(
                pixel_probs, 'b (c n_pixel_vals) h w -> b n_pixel_vals c h w', n_pixel_vals = 256
            )
            target = (img * 255).long()
            
            loss_step = torch.nn.functional.cross_entropy(pixel_probs, target, reduction='sum')
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
    from torchvision.datasets import CIFAR10
    from torchvision import transforms as T

    def transform(img): 
        img = T.ToTensor()(img)
        return img

    import json
    with open('data/dataset_paths.json') as f: 
        paths = json.load(f)

    if args.dataset == 'cifar10':
        root = paths['cifar10']         
        train_ds = CIFAR10(root=root, download=True, train=True, transform=transform)
        val_ds = CIFAR10(root=root, download=True, train=False, transform=transform)
    else: 
        raise NotImplementedError()

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


def checkpoint(args: Args, model, optimizer, epoch): 
    state_dict = {}
    state_dict['model'] = model.state_dict()
    state_dict['optimizer'] = optimizer.state_dict()
    state_dict['epoch'] = epoch 
    state_dict['rng'] = torch.random.get_rng_state()

    target_path = join(
        args.exp_dir, 'experiment_state.pt'
    )
    tmp_path = join(
        args.exp_dir, 
        'tmp.pt'
    )
    torch.save(state_dict, tmp_path)
    os.rename(tmp_path, target_path)


def add_arguments(parser): 
    parser.add_arguments(Args, dest='train')
    return parser


if __name__ == "__main__": 
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    train(args.train)
