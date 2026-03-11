import argparse
import os
import math
from pathlib import Path
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ignite.engine import Engine, Events
from ignite.metrics import FID, InceptionScore
import PIL.Image as Image


class SimpleImageFolder(Dataset):
    # Load grayscale images from a folder
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = (
            glob.glob(os.path.join(data_dir, '*.jpg')) +
            glob.glob(os.path.join(data_dir, '*.jpeg')) +
            glob.glob(os.path.join(data_dir, '*.png')) +
            glob.glob(os.path.join(data_dir, '*.JPG')) +
            glob.glob(os.path.join(data_dir, '*.PNG'))
        )
        self.image_paths = sorted(list(set(self.image_paths)))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {data_dir}")
        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img


class SinusoidalPositionEmbeddings(nn.Module):
    # Sinusoidal time embedding
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    # Basic U-Net block
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    # Lightweight U-Net
    def __init__(self, image_channels=1, down_channels=(64, 128, 256), up_channels=(256, 128, 64), time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])

        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []

        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


class DiffusionModel(nn.Module):
    # DDPM wrapper
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return (
            sqrt_alphas_cumprod_t.to(device) * x_0.to(device) +
            sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device),
            noise.to(device)
        )

    def compute_loss(self, x_0):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, x_0.device)
        noise_pred = self.model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    @torch.no_grad()
    def sample(self, num_samples, image_size, device):
        self.model.eval()
        x = torch.randn((num_samples, 1, image_size, image_size), device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            betas_t = self.get_index_from_list(self.betas, t, x.shape)
            sqrt_one_minus_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_t
            )

            if i > 0:
                posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = model_mean

        self.model.train()
        return x


def fid_preprocess(batch):
    # Convert [-1,1] grayscale to [0,1] RGB 299x299
    x = batch.detach().cpu()
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return x


def generate_batched_fakes(diffusion, num_images, save_dir, device, batch_size=32, img_size=64):
    # Save final images from the best DDPM checkpoint
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {num_images} final images")
    print("DDPM sampling is slow. Please wait...")

    generated = 0
    pbar = tqdm(total=num_images, desc="Generating")

    with torch.no_grad():
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            fake_batch = diffusion.sample(current_batch, img_size, device).detach().cpu()

            for j in range(current_batch):
                fake_img = torch.clamp(fake_batch[j], -1.0, 1.0)
                fake_img = (fake_img + 1.0) / 2.0
                save_path = save_dir / f'ddpm_pneumonia_{generated + j:04d}.png'
                vutils.save_image(fake_img, save_path)

            generated += current_batch
            pbar.update(current_batch)

    pbar.close()
    print(f"Saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Image folder')
    parser.add_argument('--out_dir', required=True, help='Output folder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_gen', type=int, default=3875)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unet = SimpleUNet(image_channels=1).to(device)
    diffusion = DiffusionModel(unet, timesteps=1000).to(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SimpleImageFolder(args.data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    losses = []
    best_fid = float('inf')
    best_epoch = -1

    def training_step(engine, batch):
        unet.train()
        real = batch.to(device)
        optimizer.zero_grad()
        loss = diffusion.compute_loss(real)
        loss.backward()
        optimizer.step()
        return {'Loss': loss.item()}

    trainer = Engine(training_step)

    @trainer.on(Events.ITERATION_COMPLETED)
    def store_losses(engine):
        losses.append(engine.state.output["Loss"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        epoch = engine.state.epoch
        torch.save(unet.state_dict(), out_dir / f'unet_ddpm_epoch_{epoch}.pth')
        print(f"Epoch [{epoch}/{args.epochs}] DDPM_L1_Loss: {losses[-1]:.4f}")

    def evaluation_step(engine, batch):
        unet.eval()
        with torch.no_grad():
            real = batch.to(device)
            b_size = min(real.size(0), 16)
            fake = diffusion.sample(b_size, 64, device)
            real = real[:b_size]
            fake = fid_preprocess(fake).to(torch.device('cpu'))
            real = fid_preprocess(real).to(torch.device('cpu'))
            return fake, real

    evaluator = Engine(evaluation_step)
    fid_metric = FID(device=torch.device('cpu'))
    is_metric = InceptionScore(device=torch.device('cpu'), output_transform=lambda x: x[0])
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        nonlocal best_fid, best_epoch
        epoch = engine.state.epoch

        if epoch > args.epochs - 5:
            print(f"\nEpoch [{epoch}/{args.epochs}] computing metrics...")
            evaluator.run(train_loader, max_epochs=1, epoch_length=10)
            metrics = evaluator.state.metrics
            current_fid = metrics['fid']
            print(f"Epoch [{epoch}/{args.epochs}] FID: {current_fid:.4f}, IS: {metrics['is']:.4f}")

            if current_fid < best_fid:
                best_fid = current_fid
                best_epoch = epoch
                torch.save(unet.state_dict(), out_dir / 'unet_ddpm_BEST_FID.pth')
                print("New best FID checkpoint saved")

    print("Starting lightweight DDPM training")
    trainer.run(train_loader, max_epochs=args.epochs)

    plt.figure()
    plt.plot(losses, label='DDPM L1 Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_dir / 'ddpm_loss_curve.png')
    plt.close()

    print(f"\nTraining finished. Best epoch: {best_epoch}, best FID: {best_fid:.4f}")
    best_model_path = out_dir / 'unet_ddpm_BEST_FID.pth'

    if best_model_path.exists():
        print(f"Loading best checkpoint from epoch {best_epoch}")
        unet.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    else:
        print("Best checkpoint not found. Using the last checkpoint.")

    generate_batched_fakes(
        diffusion,
        num_images=args.num_gen,
        save_dir=out_dir / 'ddpm_pneumonia_fakes_best',
        device=device,
        batch_size=args.batch_size,
        img_size=64
    )


if __name__ == '__main__':
    main()