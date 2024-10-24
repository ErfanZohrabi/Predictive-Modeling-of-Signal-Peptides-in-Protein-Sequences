# model_training.py

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split


def train_model(features, labels):
    """
    Train an SVM model using cross-validation and hyperparameter tuning.

    Parameters:
        features (pd.DataFrame): The input features for model training.
        labels (pd.Series or list): The labels corresponding to the input features.

    Returns:
        best_model (SVC): The trained SVM model with the best parameters.
    """
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Define the SVM model and parameters for grid search
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Return the best model found by grid search
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    return best_model

if __name__ == "__main__":
    # Example usage
    sample_features = pd.DataFrame({
        'feature1': [0.5, -1.2, 3.3],
        'feature2': [1.0, 0.5, -0.2]
    })
    sample_labels = [1, 0, 1]
    best_model = train_model(sample_features, sample_labels)
    print("Trained model:", best_model)
