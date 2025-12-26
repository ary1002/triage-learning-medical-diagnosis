"""
Monte Carlo Dropout for uncertainty estimation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


class MCDropout:
    """
    Monte Carlo Dropout uncertainty estimator
    
    Args:
        model: PyTorch model with dropout layers
        num_samples: Number of forward passes
        device: torch device
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 30,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model
        self.num_samples = num_samples
        self.device = device
        
    def enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_all_predictions: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates
        
        Args:
            dataloader: DataLoader for inference
            return_all_predictions: Return all MC samples
        
        Returns:
            predictions: Predicted classes (N,)
            uncertainties: Uncertainty scores (N,)
            probabilities: Mean predicted probabilities (N, num_classes)
            all_predictions: All MC samples if requested (num_samples, N, num_classes)
        """
        self.model.eval()
        self.enable_dropout()
        
        all_predictions = []
        
        # Run multiple forward passes
        for _ in range(self.num_samples):
            batch_preds = []
            
            with torch.no_grad():
                for images, _ in tqdm(dataloader, desc="MC Dropout sampling"):
                    images = images.to(self.device)
                    logits = self.model(images)
                    probs = torch.softmax(logits, dim=1)
                    batch_preds.append(probs.cpu().numpy())
            
            all_predictions.append(np.concatenate(batch_preds, axis=0))
        
        # Stack predictions (num_samples, N, num_classes)
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Compute statistics
        mean_probs = all_predictions.mean(axis=0)  # (N, num_classes)
        predictions = mean_probs.argmax(axis=1)     # (N,)
        
        # Compute uncertainty metrics
        uncertainties = self._compute_uncertainty(all_predictions)
        
        if return_all_predictions:
            return predictions, uncertainties, mean_probs, all_predictions
        else:
            return predictions, uncertainties, mean_probs
    
    def _compute_uncertainty(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute uncertainty from MC predictions
        
        Args:
            predictions: Array of shape (num_samples, N, num_classes)
        
        Returns:
            uncertainties: Predictive entropy (N,)
        """
        mean_probs = predictions.mean(axis=0)
        
        # Predictive entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        
        return entropy
    
    def compute_mutual_information(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute mutual information (epistemic uncertainty)
        
        Args:
            predictions: Array of shape (num_samples, N, num_classes)
        
        Returns:
            mutual_info: Mutual information scores (N,)
        """
        # Predictive entropy
        mean_probs = predictions.mean(axis=0)
        predictive_entropy = -np.sum(
            mean_probs * np.log(mean_probs + 1e-10),
            axis=1
        )
        
        # Expected entropy
        expected_entropy = -np.mean(
            np.sum(predictions * np.log(predictions + 1e-10), axis=2),
            axis=0
        )
        
        # Mutual information = Predictive entropy - Expected entropy
        mutual_info = predictive_entropy - expected_entropy
        
        return mutual_info