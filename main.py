import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, mean_squared_error, median_absolute_error, r2_score

# Carregar os dados do arquivo CSV
data = pd.read_csv("insurance.csv")

# Remove os valores nulos
data = data.dropna() 

# Converter a coluna "sex" para valores numéricos (0 para male, 1 para female)
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Converter a coluna "smoker" para valores numéricos (0 para no, 1 para yes)
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

# Converter a coluna "region" para one-hot encoding
data = pd.get_dummies(data, columns=['region'])

# Separar os dados em features (X) e target (y)
X = data.drop(columns=['charges'])
y = data['charges']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer predições nos dados de teste
y_pred = model.predict(X_test)

# Conferindo se y_pred e y_test possuem o mesmo tamanho
assert y_pred.shape == y_test.shape

# Calcular métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
median_ae = median_absolute_error(y_test, y_pred)
max_err = max_error(y_test, y_pred)

# Imprimir métricas
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2%}")
print(f"R^2 Score: {r2:.2f}")
print(f"Median Absolute Error: {median_ae:.2f}")
print(f"Max Error: {max_err:.2f}")