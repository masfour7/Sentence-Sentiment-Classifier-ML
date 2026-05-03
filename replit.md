# cs4563 ML Sentiment Classification Project

## Overview
A machine learning project for customer sentiment analysis. Uses word embeddings (Word2Vec, FastText, Bag of Words) and multiple classifiers (Logistic Regression, KNN, SVM) to classify Amazon customer review sentiment.

Originally built for Google Colab. Runs as a Jupyter Notebook server in the Replit environment.

## Project Structure
- `MLProject_trainedw2v.ipynb` - Main Jupyter notebook with ML pipeline
- `start_jupyter.sh` - Startup script for Jupyter server
- `README.md` - Project description

## Running the Project
The project runs as a Jupyter Notebook server on port 5000 via the "Start application" workflow.

## Dependencies
Python packages (managed via pip/requirements.txt):
- `jupyter`, `notebook` - Notebook server
- `numpy`, `pandas`, `matplotlib` - Data science stack
- `scikit-learn` - ML models (KNN, SVM, Logistic Regression, Naive Bayes)
- `gensim` - Word2Vec embeddings

## Notes
- The notebook was originally written for Google Colab and uses `google.colab.drive` for data loading. Those cells will need to be adapted to load data from local files when running outside of Colab.
- Jupyter runs with no authentication token/password and all origins allowed for Replit proxy compatibility.
