# data_normalization.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(data):
    """
    Normalize numerical features in the dataset.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing features to be normalized.

    Returns:
        normalized_data (pd.DataFrame): DataFrame with normalized features.
    """
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['float64', 'int64'])
    normalized_features = scaler.fit_transform(numerical_features)
    normalized_data = pd.DataFrame(normalized_features, columns=numerical_features.columns)
    return normalized_data

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': [10, 20, 30],
        'feature2': [1.0, 0.5, 0.2]
    })
    normalized_data = normalize_data(sample_data)
    print("Normalized Data Sample:\n", normalized_data)
