# Stock-Price-Prediction-project

This project contains a machine learning model to predict the movement of stock prices. It uses historical data to train a model which can then be used to forecast whether the stock price will go up or down.

**Files in this project:**
**Tesla.csv:** The dataset containing historical stock data for Tesla Inc.

**train.py:** A Python script to train the prediction model using the dataset.

**app.py:** A Python script that creates a web interface (using Gradio) to interact with the trained model.

**stock_model.pkl:** The saved, trained machine learning model.

**scaler.pkl**: A saved scaler object used to normalize the data before making predictions.

**README.md:** This file.

**Project Summary**
Developed a predictive model for stock price direction using an XGBoost Classifier.

Preprocessed historical stock data by performing feature engineering to create insightful variables such as open-close price differences, low-high volatility, and quarter-end markers.

Utilized StandardScaler from Scikit-learn to normalize feature data, preparing it for the machine learning model.

Engineered a target variable to classify whether the next day's closing price would be higher or lower, framing the problem as a binary classification task.

Leveraged Python libraries including Pandas and NumPy for data manipulation, Scikit-learn for preprocessing, and XGBoost for building the classification model.

Deployed the trained model in an interactive web application using Gradio and yfinance to fetch live data and make real-time predictions.

**How it works:**
The train.py script reads the Tesla.csv data.

It engineers features from the data (like the difference between open and close prices).

It trains an XGBoost classifier model to predict the next day's price movement.

The trained model and the data scaler are saved as .pkl files.

The app.py script loads these saved files and creates a simple web UI where you can input a stock symbol to get a prediction.


