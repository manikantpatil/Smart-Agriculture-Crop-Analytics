
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "agriculture_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

print("\n===== SMART AGRICULTURE ANALYTICS =====")
print(f"\nTotal records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Data quality checks
print("\n===== DATA QUALITY =====")
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Crop production analysis
crop_production = (
    df.groupby("crop")["production_tonnes"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== TOTAL PRODUCTION BY CROP =====")
print(crop_production.round(2))

# Average yield analysis
crop_yield = (
    df.groupby("crop")["yield_tonnes_per_hectare"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== AVERAGE YIELD BY CROP =====")
print(crop_yield.round(2))

# State production analysis
state_production = (
    df.groupby("state")["production_tonnes"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== TOTAL PRODUCTION BY STATE =====")
print(state_production.round(2))

# Environmental analysis
print("\n===== ENVIRONMENTAL SUMMARY =====")
print(
    df[["rainfall_mm", "temperature_c", "soil_ph", "humidity_pct"]]
    .describe()
    .round(2)
)

# Save analysis results
crop_production.to_csv(OUTPUT_DIR / "crop_production.csv")
crop_yield.to_csv(OUTPUT_DIR / "crop_yield.csv")
state_production.to_csv(OUTPUT_DIR / "state_production.csv")

# Chart 1: Crop production
plt.figure(figsize=(10, 6))
crop_production.sort_values().plot(kind="barh")
plt.title("Total Agricultural Production by Crop")
plt.xlabel("Production (tonnes)")
plt.ylabel("Crop")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "crop_production.png", dpi=150)
plt.close()

# Chart 2: State production
plt.figure(figsize=(10, 6))
state_production.sort_values().plot(kind="barh")
plt.title("Total Agricultural Production by State")
plt.xlabel("Production (tonnes)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "state_production.png", dpi=150)
plt.close()

# Chart 3: Rainfall vs crop yield
plt.figure(figsize=(10, 6))
plt.scatter(
    df["rainfall_mm"],
    df["yield_tonnes_per_hectare"],
    alpha=0.5
)
plt.title("Rainfall vs Crop Yield")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield (tonnes per hectare)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rainfall_vs_yield.png", dpi=150)
plt.close()

print("\nAnalysis completed successfully!")
print(f"Results saved in: {OUTPUT_DIR}")
print("Generated 3 charts and 3 CSV reports.")
