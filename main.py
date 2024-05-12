import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, mean_squared_error, median_absolute_error, r2_score

# Upload data from CSV file
data = pd.read_csv("insurance.csv")

# Remove null values
data = data.dropna() 

# Convert the "sex" column to numeric values (0 for male, 1 for female)
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Convert "smoker" column to numeric values (0 for no, 1 for yes)
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

# Convert the "region" column to one-hot encoding
data = pd.get_dummies(data, columns=['region'])

# Separate data into features (X) and target (y)
X = data.drop(columns=['charges'])
y = data['charges']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Checking if y_pred and y_test have the same size
assert y_pred.shape == y_test.shape

# Calculate assessment metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
median_ae = median_absolute_error(y_test, y_pred)
max_err = max_error(y_test, y_pred)

# Print metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2%}")
print(f"R^2 Score: {r2:.2f}")
print(f"Median Absolute Error: {median_ae:.2f}")
print(f"Max Error: {max_err:.2f}")