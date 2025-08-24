import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# =============================
# Load Dataset
# =============================
# Replace with your file
data = pd.read_csv("crop_yield.csv")
print(data.columns)


# Example expected columns
# State, Crop, Year, Rainfall, Fertilizer, Pesticide, Temperature, Yield

# =============================
# Encode Categorical Variables
# =============================
# =====================================
# Encode categorical features
# =====================================
from sklearn.preprocessing import LabelEncoder

state_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

data["state_enc"] = state_encoder.fit_transform(data["State"])
data["crop_enc"] = crop_encoder.fit_transform(data["Crop"])

# =====================================
# Features & Target
# =====================================
X = data[["Crop_Year", "state_enc", "crop_enc", "Annual_Rainfall", "Fertilizer", "Pesticide"]]
y = data["Yield"]

# =============================
# Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# Model Training
# =============================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =============================
# Evaluation
# =============================
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# =============================
# Save Model
# =============================
joblib.dump(model, "crop_yield_model.pkl")
print("✅ Model saved as crop_yield_model.pkl")


import joblib

# Save model and encoders
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(state_encoder, "state_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")

print("✅ Model and encoders saved successfully!")

import joblib

model = joblib.load("crop_yield_model.pkl")
state_encoder = joblib.load("state_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
