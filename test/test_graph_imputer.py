"""
Unit tests for Graph Imputation Model

Author: SB
Date: 2025-10-29
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from pathlib import Path

from dpplgngr.models.graph_imputer import GraphImputationModel, EnhancedGNNImputer


@pytest.fixture
def synthetic_data():
    """Generate synthetic test data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Generate correlated features
    mean = np.zeros(n_features)
    cov = np.eye(n_features) * 0.5 + 0.5
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    return df


@pytest.fixture
def missing_data(synthetic_data):
    """Create data with missing values."""
    df = synthetic_data.copy()
    missing_mask = np.zeros(df.shape, dtype=bool)
    
    # Create random missing pattern (30% missing)
    for i in range(len(df)):
        n_missing = np.random.randint(1, 3)
        missing_cols = np.random.choice(df.shape[1], n_missing, replace=False)
        missing_mask[i, missing_cols] = True
    
    df_missing = df.copy()
    df_missing.values[missing_mask] = np.nan
    
    return df, df_missing, missing_mask


class TestGraphImputationModel:
    """Test suite for GraphImputationModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GraphImputationModel(
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            k_neighbors=5
        )
        
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert model.num_heads == 4
        assert model.k_neighbors == 5
        assert model.model is None  # Not trained yet
    
    def test_fit(self, synthetic_data):
        """Test model training."""
        model = GraphImputationModel(
            hidden_dim=32,
            num_layers=2,
            epochs=10  # Few epochs for fast testing
        )
        
        model.fit(synthetic_data, verbose=False)
        
        assert model.model is not None
        assert model.scaler is not None
        assert model.feature_names == synthetic_data.columns.tolist()
        assert model.feature_means is not None
        assert model.feature_stds is not None
    
    def test_impute(self, missing_data):
        """Test imputation on missing data."""
        df_complete, df_missing, missing_mask = missing_data
        
        # Train model
        model = GraphImputationModel(
            hidden_dim=32,
            num_layers=2,
            epochs=10
        )
        model.fit(df_complete, verbose=False)
        
        # Impute
        X_imputed, uncertainty_info = model.impute(
            df_missing.values,
            missing_mask,
            return_uncertainty=True
        )
        
        # Check outputs
        assert X_imputed.shape == df_complete.shape
        assert 'uncertainty_values' in uncertainty_info
        assert 'confidence_scores' in uncertainty_info
        assert 'mean_confidence' in uncertainty_info
        
        # Check that only missing values were imputed
        observed_mask = ~missing_mask
        np.testing.assert_array_almost_equal(
            X_imputed[observed_mask],
            df_missing.values[observed_mask],
            decimal=5
        )
    
    def test_impute_without_uncertainty(self, missing_data):
        """Test imputation without uncertainty estimates."""
        df_complete, df_missing, missing_mask = missing_data
        
        model = GraphImputationModel(hidden_dim=32, num_layers=2, epochs=10)
        model.fit(df_complete, verbose=False)
        
        X_imputed = model.impute(
            df_missing.values,
            missing_mask,
            return_uncertainty=False
        )
        
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == df_complete.shape
    
    def test_save_load(self, synthetic_data):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            
            # Train and save
            model1 = GraphImputationModel(hidden_dim=32, num_layers=2, epochs=10)
            model1.fit(synthetic_data, verbose=False)
            model1.save(model_path)
            
            assert os.path.exists(model_path)
            
            # Load
            model2 = GraphImputationModel.load(model_path)
            
            assert model2.hidden_dim == model1.hidden_dim
            assert model2.num_layers == model1.num_layers
            assert model2.feature_names == model1.feature_names
            assert model2.model is not None
    
    def test_imputation_quality(self, missing_data):
        """Test that imputation produces reasonable results."""
        df_complete, df_missing, missing_mask = missing_data
        
        model = GraphImputationModel(
            hidden_dim=64,
            num_layers=3,
            epochs=50  # More epochs for better quality
        )
        model.fit(df_complete, verbose=False)
        
        X_imputed, _ = model.impute(df_missing.values, missing_mask)
        
        # Calculate RMSE on imputed values
        from sklearn.metrics import mean_squared_error
        
        true_values = df_complete.values[missing_mask]
        imputed_values = X_imputed[missing_mask]
        
        rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
        
        # RMSE should be reasonable (less than 2 std devs of data)
        data_std = np.std(df_complete.values)
        assert rmse < 2 * data_std, f"RMSE {rmse} too high compared to data std {data_std}"
    
    def test_with_dataframe(self, synthetic_data):
        """Test that model works with DataFrame input."""
        model = GraphImputationModel(hidden_dim=32, num_layers=2, epochs=10)
        
        # Should accept DataFrame
        model.fit(synthetic_data, verbose=False)
        
        assert model.feature_names == synthetic_data.columns.tolist()
    
    def test_with_numpy(self, synthetic_data):
        """Test that model works with numpy array input."""
        model = GraphImputationModel(hidden_dim=32, num_layers=2, epochs=10)
        
        # Should accept numpy array
        X_array = synthetic_data.values
        feature_names = synthetic_data.columns.tolist()
        
        model.fit(X_array, feature_names=feature_names, verbose=False)
        
        assert model.feature_names == feature_names
    
    def test_device_handling(self):
        """Test device selection (CPU/GPU)."""
        # CPU model
        model_cpu = GraphImputationModel(device='cpu')
        assert model_cpu.device.type == 'cpu'
        
        # GPU model (if available)
        if torch.cuda.is_available():
            model_gpu = GraphImputationModel(device='cuda')
            assert model_gpu.device.type == 'cuda'
    
    def test_different_missing_rates(self, synthetic_data):
        """Test imputation with different missing rates."""
        model = GraphImputationModel(hidden_dim=32, num_layers=2, epochs=20)
        model.fit(synthetic_data, verbose=False)
        
        for missing_rate in [0.1, 0.3, 0.5]:
            # Create missing data
            missing_mask = np.random.rand(*synthetic_data.shape) < missing_rate
            df_missing = synthetic_data.copy()
            df_missing.values[missing_mask] = np.nan
            
            # Impute
            X_imputed, uncertainty_info = model.impute(
                df_missing.values,
                missing_mask
            )
            
            # Check that imputation was performed
            assert not np.any(np.isnan(X_imputed))
            assert uncertainty_info['mean_confidence'] > 0
            assert uncertainty_info['mean_confidence'] <= 1


class TestEnhancedGNNImputer:
    """Test suite for the underlying GNN model."""
    
    def test_model_forward(self):
        """Test forward pass of the GNN model."""
        n_samples = 20
        n_features = 5
        hidden_dim = 16
        
        model = EnhancedGNNImputer(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=2
        )
        
        # Create dummy input
        x = torch.randn(n_samples, n_features)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        missing_mask = torch.rand(n_samples, n_features) > 0.7
        
        # Forward pass
        predictions, uncertainties, confidence = model(x, edge_index, missing_mask)
        
        assert predictions.shape == (n_samples, n_features)
        assert uncertainties.shape == (n_samples, n_features)
        assert confidence.shape == (n_samples, 1)
    
    def test_model_training_mode(self):
        """Test model can switch between train and eval modes."""
        model = EnhancedGNNImputer(input_dim=5, hidden_dim=16)
        
        # Should start in training mode
        assert model.training
        
        # Switch to eval
        model.eval()
        assert not model.training
        
        # Switch back to train
        model.train()
        assert model.training


class TestGraphConstruction:
    """Test graph construction methods."""
    
    def test_adaptive_graph_creation(self, synthetic_data):
        """Test adaptive graph construction with missing data."""
        model = GraphImputationModel(hidden_dim=32, k_neighbors=5)
        
        # Create missing mask
        missing_mask = np.zeros(synthetic_data.shape, dtype=bool)
        missing_mask[:10, :2] = True  # First 10 samples missing first 2 features
        
        # Create graph (need to scale first)
        X_scaled = model.scaler.fit_transform(synthetic_data.values)
        edge_index, edge_weights = model._create_adaptive_graph(X_scaled, missing_mask)
        
        assert edge_index.shape[0] == 2  # Should have source and target nodes
        assert edge_index.shape[1] > 0  # Should have edges
        assert len(edge_weights) == edge_index.shape[1]


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self, synthetic_data):
        """Test complete training and imputation workflow."""
        # Split data
        train_size = int(0.8 * len(synthetic_data))
        df_train = synthetic_data.iloc[:train_size]
        df_test = synthetic_data.iloc[train_size:]
        
        # Train model
        model = GraphImputationModel(hidden_dim=32, num_layers=2, epochs=20)
        model.fit(df_train, verbose=False)
        
        # Create missing data in test set
        missing_mask = np.zeros(df_test.shape, dtype=bool)
        missing_mask[:, [0, 2]] = True  # Missing features 0 and 2
        
        df_test_missing = df_test.copy()
        df_test_missing.values[missing_mask] = np.nan
        
        # Impute
        X_imputed, uncertainty_info = model.impute(
            df_test_missing.values,
            missing_mask
        )
        
        # Verify results
        assert X_imputed.shape == df_test.shape
        assert not np.any(np.isnan(X_imputed))
        assert 0 < uncertainty_info['mean_confidence'] <= 1
        
        # Check imputation quality
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(
            df_test.values[missing_mask],
            X_imputed[missing_mask]
        ))
        
        # Should be better than random imputation
        assert rmse < np.std(df_test.values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
