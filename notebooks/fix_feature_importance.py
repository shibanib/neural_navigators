"""
Helper functions to fix feature importance analysis issues.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def permutation_feature_importance(model, X_sample, feature_names=None, num_features=None, sample_size=100, verbose=True):
    """
    Calculate feature importance using permutation importance method.
    This is a more reliable alternative to SHAP for TensorFlow models.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    X_sample : numpy.ndarray
        Sample input data
    feature_names : list, optional
        Names of features
    num_features : int, optional
        Number of features to analyze (defaults to all)
    sample_size : int, optional
        Number of samples to use for analysis
    verbose : bool, optional
        Whether to print progress
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing feature importance values
    """
    try:
        # Determine number of features
        if num_features is None:
            num_features = X_sample.shape[2]  # For 3D data (samples, time_steps, features)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(num_features)]
        
        # Ensure we have the right number of feature names
        feature_names = feature_names[:num_features]
        
        # Create a smaller sample if X_sample is large
        sample_size = min(sample_size, X_sample.shape[0])
        X_small = X_sample[:sample_size]
        
        if verbose:
            print(f"Calculating permutation importance using {sample_size} samples for {num_features} features...")
        
        # Get the baseline predictions
        baseline_preds = model.predict(X_small, verbose=0)
        baseline_preds_class = np.argmax(baseline_preds, axis=1)
        
        # Initialize importance scores
        importance_scores = np.zeros(num_features)
        
        # For each feature, permute its values and measure the drop in performance
        for i in range(num_features):
            # Create a copy of the data
            X_permuted = X_small.copy()
            
            # Permute the values for this feature across all samples and time steps
            permuted_values = X_permuted[:, :, i].copy()
            np.random.shuffle(permuted_values)
            X_permuted[:, :, i] = permuted_values
            
            # Get predictions with the permuted feature
            permuted_preds = model.predict(X_permuted, verbose=0)
            permuted_preds_class = np.argmax(permuted_preds, axis=1)
            
            # Calculate the drop in performance (importance)
            # Higher drop means more important feature
            baseline_acc = accuracy_score(baseline_preds_class, baseline_preds_class)  # Always 1.0
            permuted_acc = accuracy_score(baseline_preds_class, permuted_preds_class)
            importance = baseline_acc - permuted_acc
            
            # Store the importance score
            importance_scores[i] = importance
            
            if verbose:
                print(f"Feature {i+1}/{num_features} ({feature_names[i]}): Importance = {importance:.4f}")
        
        # Create a DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        print(f"Error calculating permutation importance: {e}")
        
        # Return empty DataFrame
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(num_features or 1)]
        
        return pd.DataFrame({
            'Feature': feature_names[:num_features],
            'Importance': np.zeros(num_features or 1)
        })

def weights_feature_importance(model, feature_names=None, num_features=None):
    """
    Calculate feature importance by analyzing model weights.
    This is a simpler alternative to SHAP for TensorFlow models.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    feature_names : list, optional
        Names of features
    num_features : int, optional
        Number of features to analyze
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing feature importance values
    """
    try:
        import tensorflow as tf
        
        # Find the first dense layer
        dense_layer = None
        for layer in model.layers:
            if 'dense' in layer.name.lower() and layer.name != model.layers[-1].name:
                dense_layer = layer
                break
        
        if dense_layer is None:
            # Try to find LSTM layer if dense layer not found
            for layer in model.layers:
                if 'lstm' in layer.name.lower():
                    dense_layer = layer
                    break
        
        if dense_layer is not None:
            # Get the weights
            weights = dense_layer.get_weights()[0]
            
            # For LSTM layers, reshape if needed
            if 'lstm' in dense_layer.name.lower() and len(weights.shape) > 2:
                weights = weights.reshape(-1, weights.shape[-1])
            
            # Calculate importance as the mean absolute weight value
            importance = np.mean(np.abs(weights), axis=1)
            
            # Determine number of features
            if num_features is None:
                num_features = len(importance)
            
            # Make sure we have the right number of importance values
            if len(importance) >= num_features:
                importance = importance[:num_features]
            else:
                # If we don't have enough values, pad with zeros
                padded_importance = np.zeros(num_features)
                padded_importance[:len(importance)] = importance
                importance = padded_importance
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"Feature_{i+1}" for i in range(num_features)]
            
            # Ensure we have the right number of feature names
            feature_names = feature_names[:num_features]
            
            # Create a DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError("Could not find appropriate layer for feature importance")
            
    except Exception as e:
        print(f"Error calculating weights importance: {e}")
        
        # Return empty DataFrame
        if feature_names is None and num_features is not None:
            feature_names = [f"Feature_{i+1}" for i in range(num_features)]
        elif feature_names is None:
            feature_names = ["Feature_1"]
            num_features = 1
        else:
            num_features = len(feature_names)
        
        return pd.DataFrame({
            'Feature': feature_names[:num_features],
            'Importance': np.zeros(num_features)
        })

def analyze_model_importance(model, X_sample, feature_names=None, num_features=None, method='permutation'):
    """
    Analyze feature importance using the specified method.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    X_sample : numpy.ndarray
        Sample input data
    feature_names : list, optional
        Names of features
    num_features : int, optional
        Number of features to analyze
    method : str, optional
        Method to use for importance analysis ('permutation' or 'weights')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing feature importance values
    """
    if method.lower() == 'permutation':
        return permutation_feature_importance(model, X_sample, feature_names, num_features)
    elif method.lower() == 'weights':
        return weights_feature_importance(model, feature_names, num_features)
    else:
        print(f"Unknown method: {method}. Using permutation importance.")
        return permutation_feature_importance(model, X_sample, feature_names, num_features)

# Example usage
if __name__ == "__main__":
    print("Import this module and use analyze_model_importance() to analyze feature importance.") 