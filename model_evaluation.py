# model_evaluation.py

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, features, labels):
    """
    Evaluate the trained model using metrics like precision, recall, and F1 score.

    Parameters:
        model (SVC): The trained SVM model.
        features (pd.DataFrame): The input features for evaluation.
        labels (pd.Series or list): The true labels corresponding to the input features.

    Returns:
        metrics (dict): Dictionary containing precision, recall, F1 score, and classification report.
    """
    predictions = model.predict(features)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    report = classification_report(labels, predictions)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }

    print("Classification Report:\n", report)
    return metrics

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.svm import SVC
    
    sample_features = pd.DataFrame({
        'feature1': [0.2, 0.8, -0.5],
        'feature2': [-1.5, 1.3, 0.2]
    })
    sample_labels = [1, 0, 1]
    # Assuming you have a trained model from the training script
    trained_model = SVC(kernel='linear').fit(sample_features, sample_labels)
    evaluation_metrics = evaluate_model(trained_model, sample_features, sample_labels)
