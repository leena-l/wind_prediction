# Wind Speed & Direction Prediction

This project leverages Machine Learning to predict wind speed and direction based on historical weather data. By comparing multiple algorithms, the project identifies the most effective model for handling the complex, non-linear patterns found in atmospheric science.


# Project Overview

Accurate wind prediction is vital for optimizing renewable energy (wind farms) and improving local weather alerts. This project follows a full ML pipeline from data cleaning and Exploratory Data Analysis (EDA) to model selection and hyperparameter tuning.


# Machine Learning Strategy

I implemented and compared three distinct supervised learning algorithms to find the best fit for the dataset:

1 - Support Vector Machine (SVM): Chosen for its ability to define a clear hyperplane in high-dimensional space.

2 - Random Forest: An ensemble method used to reduce variance and capture complex feature interactions.

3 - Decision Tree: Used as a baseline model to understand the hierarchical structure of the data.


# Key Results

After rigorous training and testing, the SVM model emerged as the most reliable predictor.

Model	             Accuracy	         Note

SVM	               90%	             Best Performance

Random Forest	     87%	             Strong generalizer

Decision Tree	     82%	             Prone to slight overfitting


# Why SVM performed best?

The meteorological features exhibited clear margins of separation once the data was properly scaled. The SVM's kernel trick allowed the model to map the non-linear wind patterns more effectively than the tree-based models.


