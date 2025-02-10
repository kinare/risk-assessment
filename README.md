# Taxpayer Risk Assessment

This application provides a simple risk assessment for taxpayers based on their industry, income, and number of tax filings. It uses a synthetic dataset to train a logistic regression model for predicting whether a taxpayer is at high risk of non-compliance or other tax-related issues.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Running the Application](#running-the-application)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project includes two main Python scripts:
- `train_model.py`: Generates a synthetic dataset and trains a logistic regression model.
- `predict_risk.py`: Uses the trained model to predict the risk assessment for individual taxpayers based on user input.

## Features
- Generates synthetic data mimicking Kenyan taxpayer information.
- Trains a logistic regression model on this data.
- Predicts risk assessment based on income, industry, and tax filing frequency.

## Prerequisites
- Python 3.x
- The following Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`

You can install these via pip:
```bash
pip install pandas numpy scikit-learn joblib
```

## Installation
Clone this repository or download the scripts directly.
bash
git clone <repository-url>
Navigate to the directory containing the scripts.

## Usage
Data Generation and Model Training
Use train_model.py to generate data and train the model:
This script generates 200 records of synthetic taxpayer data tailored to Kenyan contexts.
It trains a logistic regression model based on this data.

## Risk Prediction
Use predict_risk.py to predict risk for individual taxpayers:
This script prompts for user input regarding a taxpayer's details and uses the trained model to predict their risk.

## Running the Application
Train the Model
To generate data and train the model:
```bash
python train_model.py
```
This will create trained_model.joblib and scaler.joblib in the same directory.

## Predict Risk
Once you've trained the model:
```bash
python predict_risk.py
```
This will prompt you to enter taxpayer details for a risk assessment.
