import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dữ liệu giống file trước
data = {
    "Advertising": [10, 20, 30, 40, 50, 60, 70, 80],
    "Sales": [25, 45, 65, 75, 95, 105, 130, 150]
}
df = pd.DataFrame(data)

# Huấn luyện mô hình
X = df[["Advertising"]]
y = df["Sales"]
model = LinearRegression()
model.fit(X, y)

# Dự đoán
y_pred = model.predict(X)

# Vẽ biểu đồ
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, y_pred, color='red', linewidth=2, label='Đường hồi quy')
plt.title("Linear Regression: Advertising vs Sales")
plt.xlabel("Chi phí quảng cáo")
plt.ylabel("Doanh thu")
plt.legend()
plt.grid(True)
plt.show()
