import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic dataset
np.random.seed(42)

# Number of agricultural records
n = 1000

# Agricultural categories
crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Ragi"]
states = ["Karnataka", "Maharashtra", "Punjab", "Tamil Nadu", "Andhra Pradesh"]
seasons = ["Kharif", "Rabi", "Summer"]

# Generate sample agricultural data
data = pd.DataFrame({
    "state": np.random.choice(states, n),
    "crop": np.random.choice(crops, n),
    "season": np.random.choice(seasons, n),
    "year": np.random.randint(2019, 2026, n),
    "soil_ph": np.round(np.random.uniform(5.0, 8.5, n), 2),
    "rainfall_mm": np.round(np.random.uniform(300, 1800, n), 2),
    "temperature_c": np.round(np.random.uniform(15, 40, n), 2),
    "humidity_pct": np.round(np.random.uniform(30, 95, n), 2),
    "area_hectares": np.round(np.random.uniform(1, 100, n), 2)
})

# Illustrative crop yield ranges, in tonnes per hectare
yield_ranges = {
    "Rice": (2.0, 6.0),
    "Wheat": (1.5, 5.0),
    "Maize": (2.0, 7.0),
    "Cotton": (0.5, 2.5),
    "Sugarcane": (40.0, 100.0),
    "Ragi": (0.8, 3.0)
}

data["yield_tonnes_per_hectare"] = data["crop"].apply(
    lambda crop: np.random.uniform(*yield_ranges[crop])
).round(2)

# Calculate estimated production
data["production_tonnes"] = (
    data["area_hectares"] * data["yield_tonnes_per_hectare"]
).round(2)

# Save the dataset
output_path = Path(__file__).parent / "agriculture_data.csv"
data.to_csv(output_path, index=False)

print("Agricultural dataset generated successfully!")
print(f"Total records: {len(data)}")
print(f"Saved to: {output_path}")
print("\nFirst 5 records:")
print(data.head())
print("\nDataset information:")
print(data.info())