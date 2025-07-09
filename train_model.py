import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Create synthetic dataset
np.random.seed(42)
data_size = 500

data = pd.DataFrame({
    "HeartRate": np.random.randint(50, 140, data_size),
    "Glucose": np.random.uniform(60, 200, data_size),
    "Temperature": np.random.uniform(35.0, 40.0, data_size),
    "Oxygen": np.random.uniform(85, 100, data_size),
    "Steps": np.random.randint(0, 20000, data_size),
})

# Rule-based classification
def classify_risk(row):
    if row["HeartRate"] > 120 or row["Glucose"] > 180 or row["Temperature"] > 39 or row["Oxygen"] < 90:
        return "Critical"
    elif row["HeartRate"] > 100 or row["Glucose"] > 140 or row["Temperature"] > 37.5 or row["Oxygen"] < 95:
        return "Needs Checkup"
    else:
        return "Healthy"

data["RiskLevel"] = data.apply(classify_risk, axis=1)

# Encode target
le = LabelEncoder()
data["RiskEncoded"] = le.fit_transform(data["RiskLevel"])

# Train model
X = data[["HeartRate", "Glucose", "Temperature", "Oxygen", "Steps"]]
y = data["RiskEncoded"]

model = XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric="mlogloss")
model.fit(X, y)

# Save model, encoder, and data
joblib.dump(model, "xgboost_health_model.pkl")
joblib.dump(le, "label_encoder.pkl")
data.to_csv("synthetic_health_data.csv", index=False)
