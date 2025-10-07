# Import thư viện
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1️⃣ Tạo dữ liệu mẫu
data = {
    "Advertising": [10, 20, 30, 40, 50, 60, 70, 80],
    "Sales": [25, 45, 65, 75, 95, 105, 130, 150]
}
df = pd.DataFrame(data)

# 2️⃣ Chia dữ liệu thành biến độc lập (X) và phụ thuộc (y)
X = df[["Advertising"]]   # Biến giải thích
y = df["Sales"]            # Biến mục tiêu

# 3️⃣ Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4️⃣ Tạo mô hình và huấn luyện
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ Dự đoán
y_pred = model.predict(X_test)

# 6️⃣ Đánh giá mô hình
print("Hệ số góc (slope):", model.coef_[0])
print("Hệ số chặn (intercept):", model.intercept_)
print("R² score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 7️⃣ Dự đoán giá trị mới
new_value = [[90]]
predicted_sales = model.predict(new_value)
print(f"Dự đoán doanh thu khi chi 90 cho quảng cáo: {predicted_sales[0]:.2f}")
