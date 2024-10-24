# model_testing.py

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def test_model(model, features, labels):
    """
    Test the trained SVM model on the test dataset.

    Parameters:
        model (SVC): The trained SVM model.
        features (pd.DataFrame): The input features for testing.
        labels (pd.Series or list): The labels corresponding to the input features.

    Returns:
        accuracy (float): The accuracy of the model on the test dataset.
    """
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f"Test Accuracy: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    # Example usage
    sample_features = pd.DataFrame({
        'feature1': [0.2, 0.8, -0.5],
        'feature2': [-1.5, 1.3, 0.2]
    })
    sample_labels = [1, 0, 1]
    # Assuming you have a trained model from the training script
    trained_model = SVC(kernel='linear').fit(sample_features, sample_labels)
    test_accuracy = test_model(trained_model, sample_features, sample_labels)
