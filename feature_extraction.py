# feature_extraction.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def extract_features(sequences, max_features=1000):
    """
    Extract features from sequences using CountVectorizer.

    Parameters:
        sequences (list of str): List of sequences from which features are extracted.
        max_features (int): Maximum number of features to extract.

    Returns:
        features (pd.DataFrame): DataFrame containing the extracted features.
    """
    vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"[A-Za-z]")
    features = vectorizer.fit_transform(sequences)
    features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
    return features_df

if __name__ == "__main__":
    # Example usage
    sample_sequences = [
        "MTEYKLVVVG",
        "AKTAA",
        "VVVVVV"
    ]
    features = extract_features(sample_sequences)
    print("Extracted Features Sample:\n", features)
