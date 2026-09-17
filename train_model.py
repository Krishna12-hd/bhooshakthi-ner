import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Starting BHOOSHAKTI NER AI model...")

# Create demonstration data
np.random.seed(42)
rows = 1000

rainfall_24h = np.random.uniform(0, 300, rows)
rainfall_7d = np.random.uniform(0, 1000, rows)
soil_moisture = np.random.uniform(20, 100, rows)
slope = np.random.uniform(0, 50, rows)
elevation = np.random.uniform(0, 2000, rows)
historical_landslide = np.random.randint(0, 2, rows)
ground_movement = np.random.uniform(0, 15, rows)

# Calculate demonstration risk score
risk_score = (
    rainfall_24h * 0.25
    + rainfall_7d * 0.015
    + soil_moisture * 0.30
    + slope * 0.40
    + historical_landslide * 15
    + ground_movement * 2
)

# Convert score into risk classes
risk = np.where(
    risk_score < 50,
    0,
    np.where(risk_score < 80, 1, 2)
)

# Create dataset
data = pd.DataFrame({
    "rainfall_24h": rainfall_24h,
    "rainfall_7d": rainfall_7d,
    "soil_moisture": soil_moisture,
    "slope": slope,
    "elevation": elevation,
    "historical_landslide": historical_landslide,
    "ground_movement": ground_movement,
    "risk": risk
})

# Features and target
X = data.drop("risk", axis=1)
y = data["risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("AI model training completed.")
print("Model accuracy:", round(accuracy * 100, 2), "%")

# Save model
with open("bhooshakthi_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as bhooshakthi_model.pkl")
print("BHOOSHAKTI NER AI model is ready!")