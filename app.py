import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# تنظیم seed برای تکرارپذیری
np.random.seed(42)

# ایجاد دیتاست مصنوعی
n_samples = 1000
data = {
    'rainfall_mm': np.random.uniform(50, 1000, n_samples),
    'avg_temperature_c': np.random.uniform(10, 35, n_samples),
    'soil_nitrogen_kgha': np.random.uniform(20, 200, n_samples),
    'soil_phosphorus_kgha': np.random.uniform(10, 100, n_samples),
    'soil_potassium_kgha': np.random.uniform(10, 150, n_samples),
    'soil_moisture_percent': np.random.uniform(10, 80, n_samples)
}
data['wheat_yield_tonha'] = (
    0.005 * data['rainfall_mm'] +
    0.1 * data['avg_temperature_c'] +
    0.02 * data['soil_nitrogen_kgha'] +
    0.03 * data['soil_phosphorus_kgha'] +
    0.02 * data['soil_potassium_kgha'] +
    0.05 * data['soil_moisture_percent'] +
    np.random.normal(0, 0.5, n_samples)
)
data['wheat_yield_tonha'] = np.clip(data['wheat_yield_tonha'], 2, 8)

# تبدیل به DataFrame
df = pd.DataFrame(data)

# آموزش مدل
X = df.drop('wheat_yield_tonha', axis=1)
y = df['wheat_yield_tonha']
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# رابط کاربری Streamlit
st.title("محاسبه‌گر پیش‌بینی عملکرد گندم")
st.write("مقادیر پارامترهای کشاورزی را وارد کنید تا عملکرد گندم پیش‌بینی شود.")

# دریافت ورودی‌ها از کاربر
rainfall = st.slider("میزان بارندگی (میلی‌متر)", min_value=50.0, max_value=1000.0, value=600.0)
temperature = st.slider("دمای متوسط (°C)", min_value=10.0, max_value=35.0, value=25.0)
nitrogen = st.slider("نیتروژن خاک (kg/ha)", min_value=20.0, max_value=200.0, value=100.0)
phosphorus = st.slider("فسفر خاک (kg/ha)", min_value=10.0, max_value=100.0, value=50.0)
potassium = st.slider("پتاسیم خاک (kg/ha)", min_value=10.0, max_value=150.0, value=80.0)
moisture = st.slider("رطوبت خاک (%)", min_value=10.0, max_value=80.0, value=60.0)

# تابع پیش‌بینی
def predict_yield(rainfall, temperature, nitrogen, phosphorus, potassium, moisture):
    input_data = np.array([[rainfall, temperature, nitrogen, phosphorus, potassium, moisture]])
    prediction = rf_model.predict(input_data)
    return prediction[0]

# دکمه برای پیش‌بینی
if st.button("پیش‌بینی عملکرد"):
    predicted_yield = predict_yield(rainfall, temperature, nitrogen, phosphorus, potassium, moisture)
    st.success(f"عملکرد پیش‌بینی‌شده گندم: {predicted_yield:.2f} تن بر هکتار")

# نمایش اهمیت ویژگی‌ها
st.subheader("اهمیت ویژگی‌ها")
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(feature_importance)

# توضیحات اضافی
st.write("""
این محاسبه‌گر از مدل جنگل تصادفی استفاده می‌کند که بر اساس داده‌های مصنوعی آموزش دیده است.
ویژگی‌ها شامل بارندگی، دمای متوسط، نیتروژن، فسفر، پتاسیم و رطوبت خاک هستند.
""")
