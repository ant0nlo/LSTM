
# LSTM Stock Price Prediction

This project aims to predict the closing stock price of a company (in this case, Apple Inc.) using a Long Short-Term Memory (LSTM) neural network. The model uses historical stock prices to make future predictions.

## Project Description

This project follows a series of well-defined steps to predict stock prices:

### 1. Data Loading:
The historical stock price data is loaded from a CSV file. The data includes daily stock prices of Apple Inc., which are used for model training and evaluation.

### 2. Data Preprocessing:
The data is preprocessed for the model by:
- **Filtering the 'Close' price**: Only the 'Close' column is used for predicting future values.
- **Normalization**: The data is scaled to values between 0 and 1 using `MinMaxScaler` from `sklearn`. This normalization is essential for optimizing the performance of the LSTM model, as it works best when the input data is in a standardized range.
- **Train-Test Split**: The dataset is split into training and testing sets. Typically, 80% of the data is used for training, while the remaining 20% is used for testing. The training set helps the model learn patterns, and the test set evaluates its performance on unseen data.

### 3. Feature Creation:
To train the LSTM model, past stock prices are used as input to predict the next day’s price. The previous 60 days of closing prices are used to predict the 61st-day price. This sliding window of 60 days is used to create features (`x_train`) and the target values (`y_train`).

### 4. Model Building:
An LSTM model is built using Keras. The model consists of:
- **LSTM layers**: Two stacked LSTM layers, where the first LSTM layer returns sequences (used as input for the second LSTM layer).
- **Dense layers**: Fully connected layers that reduce the dimensionality to a single output, predicting the next closing price.
- **Loss function**: The model is trained to minimize the mean squared error (MSE) between predicted and actual stock prices.
- **Optimizer**: The Adam optimizer is used to adjust model weights and minimize the loss.

### 5. Model Training:
The model is trained using the training data. This involves:
- **Epochs**: The model goes through the entire training set multiple times (one epoch is one complete pass through the data).
- **Batch size**: The number of training samples used to update the model weights after each iteration.

### 6. Model Evaluation:
Once trained, the model is evaluated on the test dataset. It predicts stock prices for the test period, and the predictions are compared against the actual prices. A common metric for evaluating the model's accuracy is Root Mean Squared Error (RMSE). The smaller the RMSE value, the better the model fits the data.

### 7. Prediction:
The trained model is used to make future predictions based on the last 60 days of stock data. The predictions are rescaled back to their original values using the inverse transform of the scaler.

### 8. Visualization:
The actual stock prices and the predicted values are visualized using Matplotlib. This helps to easily compare how well the model’s predictions match the actual stock price trend.

## Prerequisites

Before you begin, ensure you have the following requirements installed:

- **Python 3.x**
- The following Python libraries:
  - numpy
  - pandas
  - sklearn
  - keras
  - matplotlib
  - pandas_datareader

## Installation

Follow these steps to set up the project on your local machine:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repository
   ```
3. Install the required libraries:
   ```bash
   pip install numpy pandas sklearn keras matplotlib pandas_datareader
   ```

## Usage

1. After installation, open the Python script where the LSTM model is defined (`lstm_combined.py`).
2. Run the script in your terminal or preferred Python environment:
   ```bash
   python lstm_combined.py
   ```
3. The script will:
   - Load and preprocess the stock data.
   - Train the LSTM model.
   - Predict the stock price and visualize the results.
   - Output the predicted prices and performance metrics.

## Contributing

If you'd like to contribute to this project, follow these steps:

1. Fork this repository.
2. Create a branch:
   ```bash
   git checkout -b <branch_name>
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m '<commit_message>'
   ```
4. Push to the branch:
   ```bash
   git push origin <branch_name>
   ```

5. Submit a pull request.

