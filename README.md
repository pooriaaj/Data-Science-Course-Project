# Iris Classification Project

This project demonstrates a classification model using the Iris dataset. The model predicts the species of iris flowers based on their features. The project uses Logistic Regression and evaluates the model's performance with metrics such as accuracy and confusion matrix.

## Prerequisites

Ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these libraries using pip:

pip install pandas numpy matplotlib scikit-learn

## Project Structure
- **`iris_classification.py`**: The main Python script for loading the dataset, building the model, and evaluating it.

## Usage
Load the Dataset: The script loads the Iris dataset from scikit-learn.

Preprocess Data: The data is split into training and test sets, and features are normalized.

Build Model: A Logistic Regression model is trained on the training data.

Evaluate Model: The model's accuracy is calculated, and a classification report and confusion matrix are generated.

Plot Results: The confusion matrix is visualized using a heatmap.

To run the script, execute:
python iris_classification.py

## Code Details

- **Data Loading**: The Iris dataset is loaded using `load_iris()` from `scikit-learn`.

- **Data Splitting**: The dataset is split into training and test sets using `train_test_split()`.

- **Normalization**: Features are standardized using `StandardScaler()`.

- **Model Training**: A Logistic Regression model is trained.

- **Evaluation**: The model's accuracy is computed, and a classification report is generated.

- **Confusion Matrix**: A confusion matrix is computed and visualized with `matplotlib`.


## Results
The script outputs:

- **Accuracy of the model**
- **Classification report with precision, recall, and F1-score**
- **Confusion matrix as a heatmap**
