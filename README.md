# Sentiment Analysis with MLflow Integration

## Overview

This project focuses on sentiment analysis using various machine learning models. The code is implemented in a Jupyter Notebook (`.ipynb`) and demonstrates the use of MLflow for experiment tracking, model management, and reproducibility.

## Project Structure

1. **Data Preparation**
   - Load and preprocess the dataset.
   - Handle missing values and duplicates.
   - Generate a sentiment label based on the rating.

2. **Exploratory Data Analysis**
   - Visualize the distribution of sentiments.
   - Create and display word clouds from review texts.

3. **Text Preprocessing**
   - Clean and preprocess review text data.
   - Tokenize, remove stopwords, and perform lemmatization.

4. **Model Training and Evaluation**
   - Define and train multiple machine learning models:
     - Naive Bayes
     - Decision Tree
     - Logistic Regression
     - Random Forest
     - Support Vector Classifier (SVC)
     - K-Nearest Neighbors (KNN)
   - Perform hyperparameter tuning using GridSearchCV.

5. **MLflow Integration**
   - Use MLflow to log experiments, including:
     - Model parameters
     - Metrics
     - Model artifacts
   - Track and manage experiments and models using the MLflow UI.
  
6. **Model Management**:
   - Saves and registers the best models for future use.


## Requirements

- Python 3.x
- Jupyter Notebook
- `pandas`
- `numpy`
- `matplotlib`
- `nltk`
- `scikit-learn`
- `mlflow`
- `wordcloud`
- `joblib`

## Installation

To set up the project, clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/sentiment-analysis-mlflow.git
cd sentiment-analysis-mlflow
pip install -r requirements.txt
