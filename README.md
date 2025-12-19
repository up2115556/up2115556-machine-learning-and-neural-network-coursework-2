# up2115556-machine-learning-and-neural-network-coursework-2
# Machine Learning and Neural Network Coursework 2

## Overview
This repository contains the solutions for Coursework 2 of the Machine Learning
and Neural Networks module. The task involves binary classification of exoplanet
candidates from NASA’s Kepler dataset using traditional machine learning models
and a neural network.

## Repository Structure
The project is organised into separate folders for each question:

- **Q1_traditional.ipynb**: Traditional machine learning models including Logistic
  Regression and Random Forest with evaluation metrics.
- **Q2_neural_network.ipynb**: Neural network implementation and analysis.
- **Q3_experiment.ipynb**: Experimental investigation of class imbalance using
  class weighting.
- **helpers/functions.py**: Helper functions for data preprocessing shared across
  notebooks.

## Dataset
The Kepler exoplanet dataset is automatically downloaded using the `kagglehub`
package within each notebook when run in Google Colab. No manual download is required.

## How to Run
Each notebook is self-contained and can be run from top to bottom in Google Colab.
Required packages are listed in `dependancies.txt` and can be installed as needed.

## Notes
All models use a consistent preprocessing pipeline to ensure fair comparison. Each
question’s notebook includes detailed Markdown explanation and evaluation of results.

