import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("kc_house_data.csv")
print(data.info())

scaler = MinMaxScaler()

X = scaler.fit_transform(data[["sqft_living"]])
y = scaler.fit_transform(data[["price"]])

plt.scatter(X, y)
plt.xlabel("sqft_living")
plt.ylabel("Price")
plt.title(f"Linear Regression ")


model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")

plt.plot(X, y_pred, color="red")
plt.savefig("linear_regression_prediction.png")
plt.show()


joblib.dump(model, "linear_regression_model.joblib")
