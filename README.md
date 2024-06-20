
# LSTM Stock Price Prediction

This project aims to predict the closing stock price of a company (in this case, Apple Inc.) using a Long Short-Term Memory (LSTM) neural network. The model uses historical stock prices to make predictions.

## Project Description

The notebook uses the following steps to build and evaluate the LSTM model:
1. **Data Loading**: Load historical stock price data from a CSV file.
2. **Data Preprocessing**: Normalize the data and create training and testing datasets.
3. **Model Building**: Build an LSTM model using Keras.
4. **Model Training**: Train the model on the training dataset.
5. **Model Evaluation**: Evaluate the model on the testing dataset.
6. **Prediction**: Use the model to make predictions on future stock prices.
7. **Visualization**: Plot the results to visualize the predictions compared to actual stock prices.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x
- Jupyter Notebook
- The following Python libraries:
  - numpy
  - pandas
  - sklearn
  - keras
  - matplotlib
  - pandas_datareader

## Installation

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

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook lstm-predict-the-closing-stock-price.ipynb
   ```
2. Run the cells in the notebook to execute the code step by step.

## Contributing

To contribute to this project, please follow these steps:

1. Fork this repository.
2. Create a branch: 
   ```bash
   git checkout -b <branch_name>
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m '<commit_message>'
   ```
4. Push to the original branch:
   ```bash
   git push origin <project_name>/<location>
   ```
