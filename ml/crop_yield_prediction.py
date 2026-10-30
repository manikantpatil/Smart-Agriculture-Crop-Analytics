
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "agriculture_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features used to predict crop yield
features = [
    "crop",
    "state",
    "season",
    "year",
    "soil_ph",
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "area_hectares",
]

target = "yield_tonnes_per_hectare"

X = df[features]
y = df[target]

# Identify categorical and numerical columns
categorical_features = ["crop", "state", "season"]
numerical_features = [
    "year",
    "soil_ph",
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "area_hectares",
]

# Preprocess categories and train a model
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numbers", "passthrough", numerical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        )),
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== CROP YIELD PREDICTION MODEL =====")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Mean Absolute Error (MAE): {mae:.3f} tonnes/hectare")
print(f"R² score: {r2:.3f}")

# Save the trained model
model_path = MODEL_DIR / "crop_yield_model.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
print("\nNote: The dataset is synthetic. These metrics do not establish")
print("real-world prediction accuracy.")