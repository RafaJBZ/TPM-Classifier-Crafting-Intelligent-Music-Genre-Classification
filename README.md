# TPM Classifier: Crafting Intelligent Music Genre Classification

## Overview

Welcome to the TPM (Text, Pipelines, and Models) Classifier project! This exciting endeavor focuses on constructing a robust music genre classification system utilizing the Free Music Archive ([FMA](https://github.com/mdeff/fma)) dataset and state-of-the-art machine-learning techniques. Our primary objectives involve crafting two efficient data pipelines and developing an intelligent model capable of accurately classifying music genres.

## Project Goals

1. **Data Wrangling Mastery:**
   - Demonstrate expertise in data wrangling by efficiently processing the FMA dataset.
   - Build a scalable PySpark pipeline (`MusicToFurier.py`) for extracting meaningful features from audio files.

2. **AI Prowess:**
   - Showcase advanced AI capabilities in genre classification.
   - Develop a TensorFlow/Keras convolutional neural network (CNN) model (`model.py`) for accurate and efficient genre prediction.

3. **API Magic:**
   - Implement a FastAPI application (`app.py`) that exposes an endpoint for classifying music genres using the trained model.
   - Create an API (`/classify`) that accepts audio files and returns the top predicted genres.

## Key Components

1. **Data Pipelines:**
   - **MusicToFurier.py:** PySpark script for processing audio data, extracting features, and preparing it for machine learning.
   - **app.py:** FastAPI application providing an API endpoint for genre classification.

2. **Machine Learning Models:**
   - **model.py:** TensorFlow/Keras model script for training a convolutional neural network on the FMA dataset.

## Customize and Contribute

- Feel free to fork the repository and submit pull requests for improvements or fixes.
- Share your ideas and suggestions to enhance the classification system.
