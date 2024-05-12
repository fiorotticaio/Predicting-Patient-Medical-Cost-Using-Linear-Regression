# Predicting Patient Medical Cost Using Linear Regression

This project demonstrates the use of linear regression to predict medical insurance costs based on various factors such as age, sex, BMI, number of children, smoking habits, and region.

## Overview

The repository contains a Python script (`main.py`) that performs the following tasks:

- Loads the dataset from a CSV file (`insurance.csv`), available to download in: [https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
- Cleans the data by removing any null values.
- Converts categorical variables into numeric values.
- Converts the "region" column into one-hot encoding.
- Splits the data into training and testing sets.
- Initializes a linear regression model using scikit-learn.
- Trains the model on the training data.
- Makes predictions on the testing data.
- Calculates evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), R-squared Score (R2), Median Absolute Error, and Maximum Error.
- Prints out the evaluation metrics.

## Dependencies

- Python 3
- pandas
- scikit-learn

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/Predicting-Patient-Medical-Cost-Using-Linear-Regression.git
    ```
2. Navigate to the project directory:

   ```bash
   cd Predicting-Patient-Medical-Cost-Using-Linear-Regression
   ```
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Python script:

   ```bash
    python main.py
    ```

## Results

The script will print out the evaluation metrics for the linear regression model, providing insights into its performance in predicting insurance costs based on the given dataset.

## Contributors

Feel free to contribute to this project by forking the repository and submitting a pull request.