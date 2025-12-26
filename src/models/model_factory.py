"""
Model factory for creating models from configuration
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .resnet import ResNetClassifier
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any]) -> BaseModel:
    """
    Create model from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized model
    """
    model_config = config['model']
    model_name = model_config['name'].lower()
    
    # Common parameters
    num_classes = config['dataset']['num_classes']
    pretrained = model_config.get('pretrained', True)
    dropout_rate = model_config.get('dropout_rate', 0.5)
    freeze_backbone = model_config.get('freeze_backbone', False)
    freeze_layers = model_config.get('freeze_layers', 0)
    
    # Create model based on name
    if 'resnet' in model_name:
        model = ResNetClassifier(
            variant=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers
        )
    # Add other model types here (DenseNet, EfficientNet, ViT, etc.)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Log model info
    total_params = model.get_num_parameters()
    trainable_params = model.get_num_trainable_parameters()
    logger.info(f"Created {model_name} with {total_params:,} parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model