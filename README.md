# 🤖 Machine Learning Projects

This repository contains a collection of practical machine learning projects demonstrating data preprocessing, exploratory analysis, model training, evaluation and hyperparameter tuning.

The projects cover both classification and regression problems using different machine learning algorithms.

## 📂 Projects

| Project | Problem Type | Models | Main Techniques |
|---|---|---|---|
| [Cancer Prediction](Cancer-Prediction/) | Classification | Decision Tree | Train/test split, accuracy, cross-validation |
| [Marketing Analysis](Marketing-Analysis/) | Classification | Decision Tree | Data wrangling, encoding, GridSearchCV, confusion matrix |
| [Stock Price Prediction](Stock-Price-Prediction/) | Regression | XGBoost Regressor | Time-series preparation, scaling, lag features, MAE, RMSE |
| [Weather Prediction](Weather-Prediction/) | Classification | XGBoost Classifier | Label encoding, normalization, GridSearchCV |
| [Wine Quality Prediction](Wine-Quality-Prediction/) | Classification | Random Forest, SVM | EDA, standardisation, GridSearchCV, cross-validation |
## 🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- Jupyter Notebook

## 🔍 Skills Demonstrated

- Data cleaning and preprocessing
- Exploratory data analysis
- Categorical encoding
- Feature scaling and normalization
- Train/test splitting
- Classification
- Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- XGBoost
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Confusion matrices
- Classification reports
- Accuracy, MAE and RMSE evaluation

## 📊 Project Overview

### 1. Cancer Prediction

A classification project using a Decision Tree to predict whether a diagnosis is malignant or benign.

The project includes data inspection, diagnosis encoding, feature selection, model training, accuracy evaluation and cross-validation across different tree depths.

### 2. Marketing Analysis

A bank marketing classification project focused on predicting whether a customer responds positively to a marketing campaign.

The workflow includes data wrangling, categorical encoding, feature engineering, Decision Tree classification, confusion matrix analysis, classification reports and hyperparameter tuning.

### 3. Stock Price Prediction

A regression project using Tesla historical closing prices.

The workflow prepares sequential data using 15-step windows, applies MinMax scaling, uses a chronological train/test split and trains an XGBoost regression model.

Model performance is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### 4. Weather Prediction

A classification project using Seattle weather data to predict weather categories from precipitation, temperature and wind-related variables.

The project includes label encoding, feature normalization, XGBoost classification and hyperparameter tuning.

### 5. Wine Quality Prediction

A binary classification project that groups wine quality into "bad" and "good" categories.

The project compares Random Forest and Support Vector Machine (SVM) models, applies feature standardization, evaluates classification performance, tunes SVM hyperparameters and uses cross-validation.

## 📈 Model Evaluation

The projects use evaluation techniques appropriate to the different machine learning problems, including:

- Accuracy
- Classification reports
- Confusion matrices
- Cross-validation scores
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The individual project scripts generate the evaluation results when executed.

## ▶️ Running the Projects

Each project requires its corresponding dataset file.

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
