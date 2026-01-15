"""
Differential Privacy utilities for synthetic data generation.

This module provides model-agnostic differential privacy mechanisms
that can be applied to any synthesizer output.

Author: SB
Date: 2026-01-15
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Literal, Optional, Tuple

logger = logging.getLogger('luigi-interface')


class DifferentialPrivacyMechanism:
    """
    Model-agnostic differential privacy mechanism for synthetic data.
    
    Implements post-processing approaches that work with any synthesizer.
    """
    
    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        mechanism: Literal['laplace', 'gaussian', 'none'] = 'laplace'
    ):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget (smaller = more privacy). Typical values: 0.1 to 10
                    - epsilon < 1: Strong privacy
                    - epsilon 1-10: Moderate privacy  
                    - epsilon > 10: Weak privacy
            delta: Probability of privacy breach (for Gaussian mechanism)
            sensitivity: Global sensitivity of the query/data
            mechanism: Type of noise mechanism ('laplace', 'gaussian', or 'none')
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        
        if mechanism not in ['laplace', 'gaussian', 'none']:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        logger.info(f"Initialized DP mechanism: {mechanism} with ε={epsilon}, δ={delta}")
    
    def add_noise_to_synthetic_data(
        self,
        synthetic_data: pd.DataFrame,
        original_data: Optional[pd.DataFrame] = None,
        clip_to_bounds: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Add differential privacy noise to synthetic data (post-processing).
        
        Args:
            synthetic_data: Generated synthetic data
            original_data: Original data for computing bounds (optional)
            clip_to_bounds: Whether to clip noisy values to original data bounds
            
        Returns:
            Tuple of (noisy_data, privacy_info_dict)
        """
        if self.mechanism == 'none':
            logger.info("No differential privacy mechanism applied (mechanism='none')")
            return synthetic_data, {
                'mechanism': 'none',
                'epsilon': None,
                'delta': None,
                'privacy_applied': False
            }
        
        logger.info(f"="*60)
        logger.info(f"APPLYING DIFFERENTIAL PRIVACY")
        logger.info(f"="*60)
        logger.info(f"Mechanism: {self.mechanism}")
        logger.info(f"Privacy budget (ε): {self.epsilon}")
        logger.info(f"Delta (δ): {self.delta}")
        
        noisy_data = synthetic_data.copy()
        privacy_info = {
            'mechanism': self.mechanism,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'sensitivity': self.sensitivity,
            'privacy_applied': True,
            'columns_modified': []
        }
        
        # Get numeric columns
        numeric_cols = noisy_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found. No noise added.")
            privacy_info['privacy_applied'] = False
            return noisy_data, privacy_info
        
        # Compute bounds from original data if provided
        bounds = {}
        if original_data is not None and clip_to_bounds:
            for col in numeric_cols:
                if col in original_data.columns:
                    bounds[col] = (
                        float(original_data[col].min()),
                        float(original_data[col].max())
                    )
        
        # Add noise to each numeric column
        for col in numeric_cols:
            n_samples = len(noisy_data)
            
            # Generate noise based on mechanism
            if self.mechanism == 'laplace':
                scale = self.sensitivity / self.epsilon
                noise = np.random.laplace(0, scale, size=n_samples)
                logger.info(f"  Column '{col}': Added Laplace noise (scale={scale:.4f})")
                
            elif self.mechanism == 'gaussian':
                # Gaussian mechanism for (ε, δ)-DP
                sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
                noise = np.random.normal(0, sigma, size=n_samples)
                logger.info(f"  Column '{col}': Added Gaussian noise (σ={sigma:.4f})")
            
            # Apply noise
            original_mean = noisy_data[col].mean()
            noisy_data[col] = noisy_data[col] + noise
            noisy_mean = noisy_data[col].mean()
            
            # Clip to bounds if requested
            if col in bounds:
                lower, upper = bounds[col]
                clipped = noisy_data[col].clip(lower, upper)
                n_clipped = (clipped != noisy_data[col]).sum()
                noisy_data[col] = clipped
                if n_clipped > 0:
                    logger.info(f"    Clipped {n_clipped} values to bounds [{lower:.2f}, {upper:.2f}]")
            
            privacy_info['columns_modified'].append({
                'column': col,
                'original_mean': float(original_mean),
                'noisy_mean': float(noisy_mean),
                'noise_std': float(np.std(noise)),
                'bounds': bounds.get(col)
            })
        
        logger.info(f"Privacy noise added to {len(numeric_cols)} numeric columns")
        logger.info(f"="*60)
        
        return noisy_data, privacy_info
    
    def compute_privacy_budget_allocation(
        self,
        n_operations: int
    ) -> float:
        """
        Compute per-operation epsilon using composition theorem.
        
        For sequential composition: ε_total = Σ ε_i
        This splits the budget equally across operations.
        
        Args:
            n_operations: Number of operations that will use the privacy budget
            
        Returns:
            Per-operation epsilon value
        """
        per_op_epsilon = self.epsilon / n_operations
        logger.info(f"Privacy budget allocation: {self.epsilon} / {n_operations} = {per_op_epsilon:.4f} per operation")
        return per_op_epsilon


def apply_differential_privacy(
    synthetic_data: pd.DataFrame,
    epsilon: float,
    mechanism: str = 'laplace',
    original_data: Optional[pd.DataFrame] = None,
    clip_to_bounds: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to apply differential privacy to synthetic data.
    
    Args:
        synthetic_data: Generated synthetic data
        epsilon: Privacy budget
        mechanism: 'laplace', 'gaussian', or 'none'
        original_data: Original data for computing bounds
        clip_to_bounds: Whether to clip noisy values to original bounds
        
    Returns:
        Tuple of (private_data, privacy_info)
        
    Example:
        >>> private_data, info = apply_differential_privacy(
        ...     synthetic_data, 
        ...     epsilon=1.0, 
        ...     mechanism='laplace',
        ...     original_data=train_data
        ... )
    """
    dp_mechanism = DifferentialPrivacyMechanism(
        epsilon=epsilon,
        mechanism=mechanism
    )
    
    return dp_mechanism.add_noise_to_synthetic_data(
        synthetic_data=synthetic_data,
        original_data=original_data,
        clip_to_bounds=clip_to_bounds
    )


# Privacy budget recommendations
PRIVACY_LEVELS = {
    'strong': {'epsilon': 0.1, 'description': 'Strong privacy protection (high noise)'},
    'moderate': {'epsilon': 1.0, 'description': 'Moderate privacy protection (balanced)'},
    'weak': {'epsilon': 10.0, 'description': 'Weak privacy protection (low noise)'},
}


def get_recommended_epsilon(privacy_level: str = 'moderate') -> float:
    """
    Get recommended epsilon value based on privacy level.
    
    Args:
        privacy_level: 'strong', 'moderate', or 'weak'
        
    Returns:
        Recommended epsilon value
    """
    if privacy_level not in PRIVACY_LEVELS:
        raise ValueError(f"Unknown privacy level: {privacy_level}. Choose from {list(PRIVACY_LEVELS.keys())}")
    
    config = PRIVACY_LEVELS[privacy_level]
    logger.info(f"Using {privacy_level} privacy: ε={config['epsilon']} - {config['description']}")
    return config['epsilon']
