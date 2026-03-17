import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

from core.loss.SMAPEInspiredLoss import SMAPEInspiredLoss


# --------------------------
# RevIN (Reversible Instance Normalization)
# --------------------------
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode="norm", stats=None):
        # x: (B, T, C)
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)  # (B,1,C)
            var = x.var(dim=1, unbiased=False, keepdim=True)  # (B,1,C)
            std = torch.sqrt(var + self.eps)
            x_n = (x - mean) / std
            if self.affine:
                x_n = x_n * self.gamma + self.beta
            return x_n, (mean, std)
        elif mode == "denorm":
            assert stats is not None, "stats(mean,std) required for denorm"
            mean, std = stats
            if self.affine:
                x = (x - self.beta) / (self.gamma + 1e-8)
            return x * std + mean
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")


# --------------------------
# Patch Embedding
# --------------------------
class PatchEmbed(nn.Module):
    """
    Convert time series to patches and embed them
    Input: (B, T, D_in)
    Output: (B, N, d_model) where N is number of patches
    """
    def __init__(self, d_in, d_model, patch_len=16, stride=8, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(d_in * patch_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D_in)
        B, T, D = x.shape
        # Create patches using unfold: (B, N, patch_len, D)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        B, N, L, D = patches.shape
        patches = patches.reshape(B, N, L * D)  # (B, N, patch_len * D_in)
        tokens = self.proj(patches)  # (B, N, d_model)
        return self.drop(tokens)


# --------------------------
# Positional Embedding
# --------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        # x: (B, N, d_model)
        N = x.size(1)
        return x + self.pe[:, :N, :]


# --------------------------
# Masking Functions (from original PatchTST)
# --------------------------
def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    mask_value: int = 0,
):
    """
    Random masking for self-supervised pre-training
    Args:
        inputs: (B, N, patch_len * D_in) - patch tokens
        mask_ratio: ratio of patches to mask
        mask_value: value to fill masked patches
    Returns:
        masked_inputs, mask
    """
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    B, N, D = inputs.shape
    device = inputs.device

    len_keep = int(N * (1 - mask_ratio))

    # Generate random noise for each patch
    noise = torch.rand(B, N, device=device)  # (B, N)

    # Create mask: 1 is remove, 0 is keep
    mask = torch.ones(B, N, device=device)
    mask[:, :len_keep] = 0

    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)

    # Apply mask
    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1)  # (B, N, 1)

    # Mask the inputs
    masked_inputs = inputs.masked_fill(mask.bool(), mask_value)
    return masked_inputs, mask.squeeze(-1)


# --------------------------
# Pre-training Head for Reconstruction
# --------------------------
class PatchTSTPretrainHead(nn.Module):
    """
    Head for reconstructing masked patches during pre-training
    """
    def __init__(self, d_model, patch_len, d_in, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len * d_in)
        self.patch_len = patch_len
        self.d_in = d_in

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, N, d_model)
        Returns:
            reconstructed patches: (B, N, patch_len * d_in)
        """
        return self.linear(self.dropout(embedding))


# --------------------------
# Prediction Head for Fine-tuning
# --------------------------
class PatchTSTPredictionHead(nn.Module):
    """
    Head for forecasting during fine-tuning
    """
    def __init__(self, d_model, forecast_size, channels, target_idx=0, dropout=0.1):
        super().__init__()
        self.target_idx = target_idx
        self.channels = channels
        self.forecast_size = forecast_size
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_size)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, N, d_model)
        Returns:
            predictions: (B, forecast_size, channels)
        """
        # Global average pooling over patches
        pooled = embedding.mean(dim=1)  # (B, d_model)
        
        # Predict only target channel (sales)
        y_pred = self.head(pooled)  # (B, forecast_size)
        
        # Create output tensor with all channels
        B = embedding.size(0)
        out = embedding.new_zeros(B, self.forecast_size, self.channels)
        out[:, :, self.target_idx] = y_pred
        
        return out


# --------------------------
# Main CustomPatchTSTForPrediction Model
# --------------------------
class CustomPatchTSTForPrediction(nn.Module):
    """
    PatchTST model with self-supervised pre-training and fine-tuning capabilities
    Based on the original PatchTST paper and implementation
    
    Two-phase training:
    1. Self-supervised pre-training with masked patch reconstruction
    2. Fine-tuning for downstream forecasting task
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Model configuration
        self.window_size = kwargs.get('window_size', 28)
        self.forecast_size = kwargs.get('forecast_size', 7)
        self.channels = kwargs.get('feature_size', 31)
        self.target_idx = kwargs.get('target_idx', 0)
        
        # Architecture parameters
        self.d_model = kwargs.get('d_model', 256)
        self.n_heads = kwargs.get('n_heads', 8)
        self.num_layers = kwargs.get('num_layers', 3)
        self.patch_len = kwargs.get('patch_len', 16)
        self.stride = kwargs.get('stride', 8)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Pre-training parameters
        self.mask_ratio = kwargs.get('mask_ratio', 0.4)
        self.pretrain_epochs = kwargs.get('pretrain_epochs', 100)
        
        # Training mode: 'pretrain' or 'finetune'
        self.training_mode = kwargs.get('training_mode', 'finetune')
        
        # Calculate number of patches
        self.num_patches = (self.window_size - self.patch_len) // self.stride + 1
        assert self.num_patches > 0, "patch_len/stride configuration results in no patches"
        
        # 1) RevIN normalization
        self.revin = RevIN(num_features=self.channels, eps=1e-5, affine=False)
        
        # 2) Patch embedding
        self.patch_embed = PatchEmbed(
            d_in=self.channels, 
            d_model=self.d_model, 
            patch_len=self.patch_len, 
            stride=self.stride, 
            dropout=self.dropout
        )
        
        # 3) Positional embedding
        self.pos_embed = PositionalEmbedding(
            max_len=self.num_patches, 
            d_model=self.d_model
        )
        
        # 4) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 5) Task-specific heads
        # Pre-training head for reconstruction
        self.pretrain_head = PatchTSTPretrainHead(
            d_model=self.d_model,
            patch_len=self.patch_len,
            d_in=self.channels,
            dropout=self.dropout
        )
        
        # Fine-tuning head for prediction
        self.prediction_head = PatchTSTPredictionHead(
            d_model=self.d_model,
            forecast_size=self.forecast_size,
            channels=self.channels,
            target_idx=self.target_idx,
            dropout=self.dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following the original PatchTST approach"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def set_training_mode(self, mode: str):
        """Set training mode: 'pretrain' or 'finetune'"""
        assert mode in ['pretrain', 'finetune'], "Mode must be 'pretrain' or 'finetune'"
        self.training_mode = mode
    
    def forward_pretrain(self, x):
        """
        Forward pass for self-supervised pre-training
        Args:
            x: (B, window_size, channels)
        Returns:
            loss, reconstructed_patches, mask
        """
        B, T, C = x.shape
        
        # 1) RevIN normalization
        x_norm, stats = self.revin(x, mode="norm")
        
        # 2) Patch embedding
        patches = self.patch_embed(x_norm)  # (B, N, d_model)
        
        # 3) Apply random masking
        masked_patches, mask = random_masking(patches, self.mask_ratio)
        
        # 4) Add positional embedding
        masked_patches = self.pos_embed(masked_patches)
        
        # 5) Transformer encoding
        encoded = self.encoder(masked_patches)  # (B, N, d_model)
        
        # 6) Reconstruction
        reconstructed = self.pretrain_head(encoded)  # (B, N, patch_len * channels)
        
        # 7) Compute reconstruction loss (only on masked patches)
        # Reshape original patches for loss computation
        original_patches = self.patch_embed.proj.weight.new_zeros(B, self.num_patches, self.patch_len * C)
        
        # Extract original patches manually
        for i in range(self.num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            if end_idx <= T:
                patch = x_norm[:, start_idx:end_idx, :].reshape(B, -1)  # (B, patch_len * C)
                original_patches[:, i, :] = patch
        
        # Compute MSE loss only on masked patches
        print(original_patches.shape, reconstructed.shape)
        mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)  # (B, N, patch_len * C)
        loss = SMAPEInspiredLoss().forward(
            reconstructed,
            original_patches
        )
        
        return {
            'loss': loss,
            'reconstructed': reconstructed,
            'mask': mask,
            'original_patches': original_patches
        }
    
    def forward_finetune(self, x):
        """
        Forward pass for fine-tuning (forecasting)
        Args:
            x: (B, window_size, channels)
        Returns:
            predictions: (B, forecast_size, channels)
        """
        B, T, C = x.shape
        
        # 1) RevIN normalization
        x_norm, stats = self.revin(x, mode="norm")
        
        # 2) Patch embedding (no masking during fine-tuning)
        patches = self.patch_embed(x_norm)  # (B, N, d_model)
        
        # 3) Add positional embedding
        patches = self.pos_embed(patches)
        
        # 4) Transformer encoding
        encoded = self.encoder(patches)  # (B, N, d_model)
        
        # 5) Prediction
        y_norm = self.prediction_head(encoded)  # (B, forecast_size, channels)
        
        # 6) RevIN denormalization (only for target channel)
        mean, std = stats
        mu_target = mean[:, :, self.target_idx:self.target_idx+1]  # (B, 1, 1)
        std_target = std[:, :, self.target_idx:self.target_idx+1]  # (B, 1, 1)
        
        # Denormalize target channel predictions
        y_denorm = y_norm.clone()
        y_denorm[:, :, self.target_idx:self.target_idx+1] = (
            y_norm[:, :, self.target_idx:self.target_idx+1] * std_target + mu_target
        )
        
        return y_denorm
    
    def forward(self, x):
        """
        Main forward pass - routes to appropriate method based on training mode
        """
        if self.training_mode == 'pretrain':
            return self.forward_pretrain(x)
        else:
            return self.forward_finetune(x)
    
    def load_pretrained_weights(self, pretrained_path: str):
        """
        Load pre-trained weights for fine-tuning
        """
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Load only the encoder and embedding weights
        pretrained_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model_dict = self.state_dict()
        
        # Filter out prediction head weights, keep encoder weights
        filtered_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and 'prediction_head' not in k
        }
        
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded pre-trained weights from {pretrained_path}")
    
    def freeze_encoder(self):
        """
        Freeze encoder weights during fine-tuning
        """
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.pos_embed.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen for fine-tuning")
    
    def unfreeze_encoder(self):
        """
        Unfreeze encoder weights
        """
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        for param in self.pos_embed.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder weights unfrozen")


# --------------------------
# Example usage and configuration
# --------------------------
def create_custom_patchtst_model(**kwargs):
    """
    Factory function to create CustomPatchTSTForPrediction model
    """
    default_config = {
        'window_size': 28,
        'forecast_size': 7,
        'feature_size': 25,
        'target_idx': 0,
        'd_model': 256,
        'n_heads': 8,
        'num_layers': 3,
        'patch_len': 16,
        'stride': 8,
        'dropout': 0.1,
        'mask_ratio': 0.4,
        'training_mode': 'finetune'
    }
    
    # Update with provided kwargs
    config = {**default_config, **kwargs}
    
    return CustomPatchTSTForPrediction(**config)


if __name__ == "__main__":
    # Example usage
    model = create_custom_patchtst_model(
        window_size=28,
        forecast_size=7,
        feature_size=25,
        d_model=256,
        training_mode='pretrain'  # or 'finetune'
    )
    
    # Test input
    x = torch.randn(32, 28, 25)  # (batch_size, window_size, channels)
    
    # Pre-training phase
    model.set_training_mode('pretrain')
    pretrain_output = model(x)
    print(f"Pre-training loss: {pretrain_output['loss'].item():.4f}")
    
    # Fine-tuning phase
    model.set_training_mode('finetune')
    predictions = model(x)
    print(f"Predictions shape: {predictions.shape}")  # Should be (32, 7, 25)