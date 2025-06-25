# Introduction
This project focuses on analyzing and classifying the well-known Iris flower dataset using data preprocessing, feature engineering, visualization, and the K-Nearest Neighbors (KNN) algorithm. The objective is to explore the dataset thoroughly and build a machine learning model capable of accurately predicting the species of an iris flower based on its physical measurements.

## About the Dataset
The Iris dataset is a classic dataset in the fields of pattern recognition and machine learning. It includes 150 samples from three species of iris flowers:
Iris-setosa
Iris-versicolor
Iris-virginica

Each flower is described by four numerical features:
Sepal length (cm)
Sepal width (cm)
Petal length (cm)
Petal width (cm)

The goal is to predict the species of the flower using these features.


# Project Objectives
Perform exploratory data analysis (EDA) to understand the data distribution
Visualize feature relationships and class separability
Engineer new features to improve classification accuracy
Detect and handle outliers using the IQR method
Encode categorical variables for machine learning
Apply the K-Nearest Neighbors (KNN) algorithm for classification
Evaluate model performance and find the optimal value of k

## Key Tasks Performed
Data Cleaning: Removed unnecessary columns like Id
Checked and removed duplicate records
Verified the dataset had no missing values


# Data Visualization:
Count plots, boxplots, violin plots, histograms, and scatter plots
Pair plots to show class separability
Correlation heatmap to understand feature relationships


# Feature Engineering:
Created new ratio-based features: sepal_ratio and petal_ratio
Added a categorical feature petal_size based on petal length
## Outlier Detection:
Used the IQR method to detect and remove outliers in sepal width
## Label Encoding:
Converted Species and petal_size to numerical values using Label Encoding


# Model Building:
Split the data into training and testing sets
Trained KNN models with different values of k
Evaluated model performance using accuracy and classification report
Identified the best k value through visualization


# Results
Iris-setosa was clearly distinguishable based on feature values.
Iris-versicolor and Iris-virginica showed some overlap but were distinguishable using engineered features.
The KNN classifier achieved high accuracy, especially with an optimal value of k.


# Conclusion
This project demonstrates how combining exploratory data analysis, feature engineering, and simple machine learning models like KNN can lead to effective and interpretable classification results. It also emphasizes the importance of understanding the data before modeling.

