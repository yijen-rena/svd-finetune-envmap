import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
import numpy as np
from typing import Tuple

from diffusers import AutoencoderKL
from encoders.envmap_autoencoder_unet import UNetEnvMapConditionModel

from utils import *

class EnvMapDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.envmap_files = list(self.data_dir.rglob("*.hdr"))  # Recursively find all .hdr files
        
    def __len__(self):
        return len(self.envmap_files)
    
    def __getitem__(self, idx):
        envmap = read_hdr(self.envmap_files[idx])
        
        # Normalize to [-1, 1]
        if envmap.max() > 1.0:
            envmap = envmap / 255.0
        envmap = (envmap * 2.0) - 1.0
        
        # Reshape to [C, H, W]
        envmap = envmap.permute(2, 0, 1)
        
        return envmap

def preprocess_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Preprocesses a batch of environment maps.
    Args:
        batch: Tensor of shape [B, C, H, W]
    Returns:
        Preprocessed batch
    """
    return batch  # Already preprocessed in dataset

class EnvMapTrainer:
    def __init__(
        self,
        unet: UNetEnvMapConditionModel,
        vae: AutoencoderKL,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        output_dir: str = "outputs",
        experiment_name: str = "unet_envmap_autoencoder",
        use_wandb: bool = True,
    ):
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.vae.eval()  # Freeze VAE
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(unet.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Setup output directory
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="unet_envmap_training",
                name=experiment_name,
                config={
                    "learning_rate": learning_rate,
                    "model_config": unet.config,
                    "batch_size": train_dataloader.batch_size,
                }
            )

    @torch.no_grad()
    def encode_envmaps(self, envmaps: torch.Tensor) -> torch.Tensor:
        """Encode environment maps using frozen VAE"""
        return self.vae.encode(envmaps).latent_dist.sample()

    def train_epoch(self, epoch: int) -> float:
        self.unet.train()
        total_loss = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")
        for batch_idx, envmaps in enumerate(progress_bar):
            envmaps = envmaps.to(self.device)
            
            # Encode with VAE
            with torch.no_grad():
                latents = self.encode_envmaps(envmaps)
            
            self.optimizer.zero_grad()
            
            # Forward pass through UNet
            output = self.unet(latents)
            loss = self.criterion(output.sample, latents)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.6f}"})
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "train_batch_loss": loss.item(),
                    "train_avg_loss": avg_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                })
        
        return total_loss / len(self.train_dataloader)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.unet.eval()
        total_loss = 0
        
        progress_bar = tqdm(self.val_dataloader, desc=f"Validation Epoch {epoch}")
        for batch_idx, envmaps in enumerate(progress_bar):
            envmaps = envmaps.to(self.device)
            
            # Encode with VAE
            latents = self.encode_envmaps(envmaps)
            
            # Forward pass through UNet
            output = self.unet(latents)
            loss = self.criterion(output.sample, latents)
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"val_loss": f"{avg_loss:.6f}"})
            
            # Log sample reconstructions periodically
            if batch_idx == 0 and self.use_wandb:
                # Decode both original and reconstructed latents
                with torch.no_grad():
                    original_decoded = self.vae.decode(latents).sample
                    reconstructed_decoded = self.vae.decode(output.sample).sample
                
                wandb.log({
                    "original_envmap": wandb.Image(original_decoded[0].cpu()),
                    "reconstructed_envmap": wandb.Image(reconstructed_decoded[0].cpu()),
                })
        
        val_loss = total_loss / len(self.val_dataloader)
        if self.use_wandb:
            wandb.log({"val_loss": val_loss})
        
        return val_loss

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pt')

    def train(self, num_epochs: int, save_every: int = 5):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            # Regular checkpoint saving
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_loss)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

def main():
    # Load pretrained VAE
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", # TODO!
        subfolder="vae",
        use_auth_token=False,
        torch_dtype=torch.float32
    )
    vae.requires_grad_(False)  # Freeze VAE
    
    # Initialize UNet
    unet = UNetEnvMapConditionModel(
        in_channels=4,  # VAE latent channels
        out_channels=4,  # VAE latent channels
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2
    )
    
    # Create dataloaders
    train_dataset = EnvMapDataset("../datasets/haven_envmaps/train")
    val_dataset = EnvMapDataset("../datasets/haven_envmaps/val")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,  # Adjust based on GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = EnvMapTrainer(
        unet=unet,
        vae=vae,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name="unet_envmap_autoencoder"
    )
    
    # Start training
    trainer.train(num_epochs=100, save_every=5)

if __name__ == "__main__":
    main()