import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Dict, Any

def to_mlx(tensor):
    """Convert PyTorch tensor to MLX array"""
    if tensor is None:
        return None
    if hasattr(tensor, 'numpy'):
        # Handle PyTorch tensor
        return mx.array(tensor.detach().cpu().numpy(), dtype=mx.float16)
    elif isinstance(tensor, np.ndarray):
        # Handle numpy array
        return mx.array(tensor, dtype=mx.float16)
    elif isinstance(tensor, list):
        # Handle list of tensors/arrays
        return mx.array(np.array(tensor, dtype=np.float16), dtype=mx.float16)
    # Handle other types by converting to numpy first
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
                # Use MLX's built-in numpy conversion
                x = torch.from_numpy(np.array(x.tolist(), dtype=np.float16))
            elif isinstance(x, list):
                x = torch.from_numpy(np.stack([np.array(arr.tolist(), dtype=np.float16) for arr in x]))
            else:
                x = torch.from_numpy(np.array(x, dtype=np.float16))
            if torch.backends.mps.is_available():
                x = x.to("mps")
        
        # Encode with PyTorch VAE
        with torch.no_grad():
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
                # Use MLX's built-in numpy conversion
                z = torch.from_numpy(np.array(z.tolist(), dtype=np.float16))
            elif isinstance(z, list):
                z = torch.from_numpy(np.stack([np.array(arr.tolist(), dtype=np.float16) for arr in z]))
            else:
                z = torch.from_numpy(np.array(z, dtype=np.float16))
            if torch.backends.mps.is_available():
                z = z.to("mps")
        
        # Scale latents
        if hasattr(self.vae, 'config'):
            if hasattr(self.vae.config, 'scaling_factor'):
                z = z / self.vae.config.scaling_factor
        
        # Decode with PyTorch VAE
        with torch.no_grad():
            sample = self.vae.decode(z, **kwargs)
            if isinstance(sample, torch.Tensor):
                sample = sample
            else:
                sample = sample.sample
        
        # Convert back to MLX
        return {"sample": to_mlx(sample)}

    def __call__(self, *args, **kwargs):
        """Forward pass through VAE."""
        return self.decode(*args, **kwargs)
