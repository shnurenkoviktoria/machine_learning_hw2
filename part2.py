import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np


data = pd.read_csv("kc_house_data.csv")
data = data.drop(["date"], axis=1)

selected_columns = [
    "sqft_living",
    "sqft_lot",
    "floors",
    "condition",
    "yr_built",
    "sqft_living15",
    "sqft_lot15",
]

data_norm = preprocessing.normalize(data)
data_norm = pd.DataFrame(data_norm, columns=data.columns)

X = data_norm[selected_columns]
y = data_norm["price"]


scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=50
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (Multiple Variables): {mse}")

joblib.dump(model, "linear_regression_model_scaled_multi_valued.joblib")

X_constant = np.column_stack((np.ones(X.shape[0]), X))

theta_analytical = np.linalg.inv(X_constant.T @ X_constant) @ X_constant.T @ y

y_pred_analytical = X_constant @ theta_analytical

plt.scatter(y_test, y_pred, label="Iterative Model", alpha=0.5, color="red")
plt.scatter(y, y_pred_analytical, label="Analytical Model", alpha=0.5, color="blue")
plt.xlabel("Actual Price (Normalized)")
plt.ylabel("Predicted Price (Normalized)")
plt.legend()
plt.title("Comparison of Predictions between Iterative and Analytical Models")
plt.show()


joblib.dump(model, "linear_regression_model_normalized.joblib")


def calculate_loss(y, h_x):
    loss = np.sqrt(np.sum((h_x - y) ** 2) / len(y))
    return loss


# Приклад використання функції
y_vector = np.random.rand(100)
h_x_vector = np.random.rand(100)
loss_result = calculate_loss(y_vector, h_x_vector)
print(f"Loss(calculate_loss): {loss_result}")
