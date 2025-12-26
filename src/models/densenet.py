"""
DenseNet-based classifiers for medical image classification.

Supports:
- DenseNet121, DenseNet169, DenseNet201 architectures
- Pretrained ImageNet weights
- Feature extraction and fine-tuning
- Uncertainty estimation capabilities
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class DenseNetClassifier(BaseModel):
    """
    DenseNet-based image classifier for medical diagnosis.
    
    Implements DenseNet architecture variants with support for:
    - Multiple depth variants (121, 169, 201)
    - Transfer learning from ImageNet
    - Feature extraction
    - Uncertainty estimation via MC Dropout
    
    Reference:
        Huang et al. (2017). Densely Connected Convolutional Networks.
        https://arxiv.org/abs/1608.06993
    """
    
    SUPPORTED_VARIANTS = {
        "densenet121": {"depth": 121, "growth_rate": 32},
        "densenet169": {"depth": 169, "growth_rate": 32},
        "densenet201": {"depth": 201, "growth_rate": 32},
    }
    
    def __init__(
        self,
        num_classes: int,
        variant: str = "densenet121",
        pretrained: bool = True,
        num_channels: int = 3,
        dropout_rate: float = 0.0,
        dropout_p_mc: float = 0.5,
        feature_dim: Optional[int] = None,
    ):
        """
        Initialize DenseNet classifier.
        
        Args:
            num_classes: Number of output classes
            variant: DenseNet variant - "densenet121", "densenet169", or "densenet201"
            pretrained: Whether to use ImageNet pretrained weights
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate in classifier head
            dropout_p_mc: Dropout probability for MC Dropout uncertainty estimation
            feature_dim: Dimension of feature space. If None, uses model defaults (1024 for all variants)
        
        Raises:
            ValueError: If variant is not supported
        """
        super().__init__(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            dropout_p_mc=dropout_p_mc,
        )
        
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported DenseNet variant '{variant}'. "
                f"Supported: {list(self.SUPPORTED_VARIANTS.keys())}"
            )
        
        self.variant = variant
        self.pretrained = pretrained
        self.num_channels = num_channels
        
        # Load pretrained model
        model_dict = {
            "densenet121": models.densenet121,
            "densenet169": models.densenet169,
            "densenet201": models.densenet201,
        }
        
        self.backbone = model_dict[variant](pretrained=pretrained)
        self.feature_dim = feature_dim or 1024
        
        logger.info(
            f"Initialized {variant} with pretrained={pretrained}, "
            f"num_classes={num_classes}"
        )
        
        # Handle input channel mismatch
        if num_channels != 3:
            self._adapt_input_channels(num_channels)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )
        
        # For MC Dropout
        self.mc_dropout = nn.Dropout(p=dropout_p_mc)
    
    def _adapt_input_channels(self, num_channels: int) -> None:
        """
        Adapt first convolutional layer to accept different number of input channels.
        
        Args:
            num_channels: Desired number of input channels
        """
        if num_channels == 3:
            return
        
        # Get the first convolution layer
        first_conv = self.backbone.features[0]
        
        if num_channels == 1:
            # Convert RGB weights to grayscale by averaging
            with torch.no_grad():
                original_weight = first_conv.weight.data
                first_conv.weight = nn.Parameter(
                    original_weight.mean(dim=1, keepdim=True)
                )
        else:
            # Create new layer with different input channels
            new_conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            
            # Initialize with original weights, repeating or averaging as needed
            with torch.no_grad():
                if num_channels < 3:
                    new_conv.weight[:, :, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
                else:
                    # Repeat weights
                    repeats = (num_channels // 3) + 1
                    new_conv.weight[:, :, :, :] = first_conv.weight.repeat(1, repeats, 1, 1)
                    new_conv.weight = nn.Parameter(new_conv.weight[:, :num_channels, :, :])
            
            self.backbone.features[0] = new_conv
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width)
        
        Returns:
            Tuple of (logits, features) where:
            - logits: Class predictions of shape (batch_size, num_classes)
            - features: Feature representations of shape (batch_size, feature_dim)
        """
        # Extract features from backbone
        features = self.backbone.features(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Apply classifier
        logits = self.backbone.classifier(features)
        
        return logits, features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations from input images.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width)
        
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        features = self.backbone.features(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 10,
        return_all_samples: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates using MC Dropout.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width)
            num_samples: Number of MC Dropout samples
            return_all_samples: If True, return all prediction samples
        
        Returns:
            Dictionary containing:
            - 'logits': Mean prediction logits (batch_size, num_classes)
            - 'probs': Mean prediction probabilities (batch_size, num_classes)
            - 'uncertainty': Predictive entropy (batch_size,)
            - 'samples': All prediction samples if return_all_samples=True
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits, _ = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        self.eval()  # Disable dropout
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, num_classes)
        
        # Compute mean and uncertainty
        mean_probs = predictions.mean(dim=0)
        
        # Entropy-based uncertainty
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        results = {
            "logits": torch.log(mean_probs + 1e-10),
            "probs": mean_probs,
            "uncertainty": entropy,
        }
        
        if return_all_samples:
            results["samples"] = predictions
        
        return results
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone parameters for transfer learning.
        
        Args:
            freeze: If True, freeze backbone. If False, unfreeze.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze
        
        logger.info(f"Backbone frozen: {freeze}")
    
    def set_dropout_rate(self, rate: float) -> None:
        """
        Update dropout rate in the classifier.
        
        Args:
            rate: New dropout rate
        """
        self.backbone.classifier[0].p = rate
    
    def get_model_summary(self) -> Dict:
        """
        Get a summary of the model architecture and parameters.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "variant": self.variant,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "feature_dimension": self.feature_dim,
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"DenseNetClassifier(\n"
            f"  variant={self.variant},\n"
            f"  num_classes={self.num_classes},\n"
            f"  pretrained={self.pretrained},\n"
            f")"
        )
