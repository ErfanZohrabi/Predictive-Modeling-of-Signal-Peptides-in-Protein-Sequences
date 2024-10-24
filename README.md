# Predictive-Modeling-of-Signal-Peptides-in-Protein-Sequences
This project uses machine learning to predict signal peptides in protein sequences. It features data normalization, SVM-based classification, and cross-validation to ensure reliable signal peptide detection, creating a streamlined, effective prediction pipeline

# Machine Learning Project for Sequence Analysis

## Overview

This project is a comprehensive machine-learning pipeline for analyzing biological sequences. It involves loading data, normalizing and scaling features, extracting relevant features, training an SVM model, testing the model, and evaluating its performance using key metrics such as accuracy, precision, recall, and F1 score.

The key components of the project include:
1. **Loading Data**: Reading and preprocessing data from `.tsv` files containing biological sequence data.
2. **Data Normalization**: Normalizing the dataset to prepare it for modeling.
3. **Feature Extraction**: Extracting features from sequence data using vectorization.
4. **Model Training**: Training an SVM model with cross-validation and hyperparameter tuning.
5. **Model Testing**: Testing the model on a separate test dataset.
6. **Model Evaluation**: Evaluating the model's performance using multiple metrics.

## Repository Structure

```
├── data_loading.py
├── data_normalization.py
├── feature_extraction.py
├── model_training.py
├── model_testing.py
├── model_evaluation.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Setup Instructions

### Requirements

- Python 3.7+
- Libraries specified in `requirements.txt`

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sequence-analysis-ml.git
   cd sequence-analysis-ml
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows use: env\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Loading**:
   Use `data_loading.py` to load positive and negative sequences from `.tsv` files.
   ```bash
   python data_loading.py
   ```

2. **Data Normalization**:
   Use `data_normalization.py` to normalize your features before proceeding to training.
   ```bash
   python data_normalization.py
   ```

3. **Feature Extraction**:
   Run `feature_extraction.py` to vectorize sequences and generate features.
   ```bash
   python feature_extraction.py
   ```

4. **Model Training**:
   Use `model_training.py` to train the SVM model using cross-validation.
   ```bash
   python model_training.py
   ```

5. **Model Testing**:
   Test your trained model using `model_testing.py`.
   ```bash
   python model_testing.py
   ```

6. **Model Evaluation**:
   Evaluate the performance of your model with `model_evaluation.py`.
   ```bash
   python model_evaluation.py
   ```

## Project Details

### Data
- The dataset consists of positive and negative sequences saved in `.tsv` files.
- Positive sequences contain known features such as cleavage sites, while negative sequences do not.

### Model
- The project uses an **SVM (Support Vector Machine)** model for classification.
- Cross-validation and hyperparameter tuning are used to optimize the model performance.

### Evaluation Metrics
- The model is evaluated using accuracy, precision, recall, F1 score, and a detailed classification report.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.

## Contact
If you have any questions or need further information, feel free to contact the project maintainer at [Erfanzohrabi.ez@gmail.com].

## Acknowledgements
This project was developed using data provided by [Unibo LB2]. We would like to thank the researchers and developers who contributed to the data collection and pre-processing.

