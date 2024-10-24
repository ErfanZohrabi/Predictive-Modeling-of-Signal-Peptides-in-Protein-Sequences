# data_loading.py

import pandas as pd

def load_data(positive_path, negative_path):
    """
    Load positive and negative datasets.

    Parameters:
        positive_path (str): Path to the positive dataset (.tsv file).
        negative_path (str): Path to the negative dataset (.tsv file).

    Returns:
        positive_data (pd.DataFrame): DataFrame containing positive sequences.
        negative_data (pd.DataFrame): DataFrame containing negative sequences.
    """
    positive_data = pd.read_csv(positive_path, sep='\t')
    negative_data = pd.read_csv(negative_path, sep='\t')
    return positive_data, negative_data

if __name__ == "__main__":
    # Example usage
    positive_path = "./data/positive.tsv"
    negative_path = "./data/negative.tsv"
    positive_data, negative_data = load_data(positive_path, negative_path)
    print("Positive Data Sample:\n", positive_data.head())
    print("Negative Data Sample:\n", negative_data.head())
