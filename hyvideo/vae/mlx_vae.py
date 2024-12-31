import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Dict, Any

def to_mlx(tensor):
    """Convert PyTorch tensor to MLX array maintaining fp16"""
    if tensor is None:
        return None
    if hasattr(tensor, 'numpy'):
        # Handle PyTorch tensor - detach and move to CPU first
        tensor = tensor.detach().cpu()
        # Convert directly to float16
        return mx.array(tensor.numpy(), dtype=mx.float16)
    elif isinstance(tensor, np.ndarray):
        # Handle numpy array
        return mx.array(tensor, dtype=mx.float16)
    elif isinstance(tensor, list):
        # Handle list
        return mx.array(np.array(tensor, dtype=np.float16), dtype=mx.float16)
    # Handle other types
    return mx.array(np.array(tensor, dtype=np.float16), dtype=mx.float16)

class MLXVAE:
    """MLX-optimized VAE wrapper for HunyuanVideo."""
    
    def __init__(self, vae):
        """Initialize MLX VAE wrapper.
        
        Args:
            vae: PyTorch VAE model to wrap
        """
        self.vae = vae
        self.config = vae.config if hasattr(vae, 'config') else None
        
    def encode(self, x: mx.array) -> Dict[str, mx.array]:
        """Encode input using VAE encoder.
        
        Args:
            x: Input tensor to encode
            
        Returns:
            Dictionary containing latent representation
        """
        # Convert MLX array to PyTorch tensor for encoding
        import torch
        if not isinstance(x, torch.Tensor):
            # Convert MLX array to numpy array first
            if isinstance(x, mx.array):
                # Convert directly to float16
                np_array = np.array(x.tolist(), dtype=np.float16)
                x = torch.from_numpy(np_array).to(dtype=torch.float16)
            elif isinstance(x, list):
                np_array = np.stack([np.array(arr.tolist(), dtype=np.float16) for arr in x])
                x = torch.from_numpy(np_array).to(dtype=torch.float16)
            else:
                np_array = np.array(x, dtype=np.float16)
                x = torch.from_numpy(np_array).to(dtype=torch.float16)
            
            # Force CPU for operations not supported by MPS
            x = x.cpu()
        
        # Encode with PyTorch VAE on CPU
        with torch.no_grad():
            # Ensure VAE is on CPU for unsupported ops
            self.vae = self.vae.cpu()
            z = self.vae.encode(x)
            if isinstance(z, torch.Tensor):
                latents = z
            else:
                # Handle case where encoder returns dictionary
                latents = z["latent_dist"].sample()
            
            # Scale latents
            if hasattr(self.vae, 'config'):
                if hasattr(self.vae.config, 'scaling_factor'):
                    latents = latents * self.vae.config.scaling_factor
            
            # Convert to float16 after processing
            latents = latents.to(dtype=torch.float16)
        
        # Convert back to MLX
        return {"sample": to_mlx(latents)}
    
    def decode(self, z: mx.array, **kwargs) -> Dict[str, mx.array]:
        """Decode latent representation to image.
        
        Args:
            z: Latent representation to decode
            **kwargs: Additional arguments passed to decoder
            
        Returns:
            Dictionary containing decoded image
        """
        # Convert MLX array to PyTorch tensor for decoding
        import torch
        if not isinstance(z, torch.Tensor):
            # Convert MLX array to numpy array first
            if isinstance(z, mx.array):
                # Convert directly to float16
                np_array = np.array(z.tolist(), dtype=np.float16)
                z = torch.from_numpy(np_array).to(dtype=torch.float16)
            elif isinstance(z, list):
                np_array = np.stack([np.array(arr.tolist(), dtype=np.float16) for arr in z])
                z = torch.from_numpy(np_array).to(dtype=torch.float16)
            else:
                np_array = np.array(z, dtype=np.float16)
                z = torch.from_numpy(np_array).to(dtype=torch.float16)
            
            # Force CPU for operations not supported by MPS
            z = z.cpu()
        
        # Scale latents
        if hasattr(self.vae, 'config'):
            if hasattr(self.vae.config, 'scaling_factor'):
                z = z / self.vae.config.scaling_factor
        
        # Decode with PyTorch VAE on CPU
        with torch.no_grad():
            # Ensure VAE is on CPU for unsupported ops
            self.vae = self.vae.cpu()
            sample = self.vae.decode(z, **kwargs)
            if not isinstance(sample, torch.Tensor):
                sample = sample.sample
            # Convert back to float16 after processing
            sample = sample.to(dtype=torch.float16)
        
        # Convert back to MLX
        return {"sample": to_mlx(sample)}

    def __call__(self, *args, **kwargs):
        """Forward pass through VAE."""
        return self.decode(*args, **kwargs)
