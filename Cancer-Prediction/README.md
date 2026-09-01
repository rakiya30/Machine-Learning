# 🩺 Cancer Prediction

## 📌 Project Overview

This project uses machine learning to classify cancer diagnoses as either **malignant** or **benign**.

The analysis uses a Decision Tree classifier and includes data inspection, preprocessing, model training, evaluation and cross-validation.

## 🎯 Objective

The objective is to build a classification model that can distinguish between malignant and benign diagnoses based on the available features in the dataset.

## 🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Decision Tree Classifier

## 🔍 Methodology

The project follows these main steps:

1. Load and inspect the dataset
2. Check for missing values and dataset dimensions
3. Explore the distribution of diagnoses
4. Convert diagnosis labels:
   - `M` = Malignant
   - `B` = Benign
5. Remove the ID column
6. Separate the features and target variable
7. Split the data into training and testing sets
8. Train a Decision Tree classifier
9. Evaluate the model using accuracy
10. Use 10-fold cross-validation
11. Test different Decision Tree depths from 1 to 20

## 🌳 Machine Learning Model

### Decision Tree Classifier

A Decision Tree classifier is used to predict whether a diagnosis is malignant or benign.

The project also evaluates different tree depths to compare training accuracy with cross-validation performance.

## 📊 Model Evaluation

The project evaluates the model using:

- Accuracy score
- Training accuracy
- 10-fold cross-validation score

The script prints the evaluation results when it is executed.

## 📁 Dataset

The Python script expects a dataset named:

`data.csv`

The dataset contains a `diagnosis` column used as the target variable.

## ▶️ How to Run

Install the required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
