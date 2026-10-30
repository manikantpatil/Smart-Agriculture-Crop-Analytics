
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Agriculture Analytics",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "agriculture_data.csv"
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"


# --------------------------------------------------
# DASHBOARD TITLE
# --------------------------------------------------

st.title("🌱 Smart Agriculture & Crop Analytics")

st.caption(
    "Explore crop production, agricultural yield, rainfall, "
    "and state-wise agricultural trends."
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# SEASON DISPLAY NAMES
# --------------------------------------------------

season_names = {
    "Summer": "Summer Season",
    "Rabi": "Winter Season",
    "Kharif": "Rainy Season"
}

# Create a separate column for displaying season names.
# The original season column remains unchanged for the ML model.
df["season_display"] = (
    df["season"].map(season_names).fillna(df["season"])
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🔎 Data Filters")

states = sorted(df["state"].dropna().unique())
crops = sorted(df["crop"].dropna().unique())
years = sorted(df["year"].dropna().unique())

season_options = [
    season_names.get(season, season)
    for season in sorted(df["season"].dropna().unique())
]

selected_states = st.sidebar.multiselect(
    "Select states",
    options=states,
    default=states
)

selected_crops = st.sidebar.multiselect(
    "Select crops",
    options=crops,
    default=crops
)

selected_years = st.sidebar.multiselect(
    "Select years",
    options=years,
    default=years
)

selected_seasons = st.sidebar.multiselect(
    "Select seasons",
    options=season_options,
    default=season_options
)

# Convert the selected display names back to original season values.
selected_original_seasons = [
    original
    for original in df["season"].dropna().unique()
    if season_names.get(original, original) in selected_seasons
]


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = df[
    df["state"].isin(selected_states)
    & df["crop"].isin(selected_crops)
    & df["year"].isin(selected_years)
    & df["season"].isin(selected_original_seasons)
].copy()


if filtered_df.empty:
    st.warning("No records match your selected filters.")
    st.stop()


# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_production = filtered_df["production_tonnes"].sum()

average_yield = filtered_df[
    "yield_tonnes_per_hectare"
].mean()

total_area = filtered_df["area_hectares"].sum()

record_count = len(filtered_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Production",
    f"{total_production:,.0f} tonnes"
)

col2.metric(
    "Average Crop Yield",
    f"{average_yield:.2f} tonnes/ha"
)

col3.metric(
    "Total Cultivated Area",
    f"{total_area:,.0f} hectares"
)

col4.metric(
    "Records Selected",
    f"{record_count:,}"
)

st.divider()


# --------------------------------------------------
# CROP PRODUCTION ANALYSIS
# --------------------------------------------------

st.subheader("🌾 Crop Production Analysis")

crop_production = (
    filtered_df.groupby("crop", as_index=False)["production_tonnes"]
    .sum()
    .sort_values("production_tonnes", ascending=False)
)

fig_crop = px.bar(
    crop_production,
    x="crop",
    y="production_tonnes",
    color="crop",
    title="Total Production by Crop",
    labels={
        "crop": "Crop",
        "production_tonnes": "Production (tonnes)"
    },
    template="plotly_dark"
)

st.plotly_chart(fig_crop, use_container_width=True)


# --------------------------------------------------
# STATE-WISE PRODUCTION
# --------------------------------------------------

st.subheader("📍 State-wise Production")

state_production = (
    filtered_df.groupby("state", as_index=False)["production_tonnes"]
    .sum()
    .sort_values("production_tonnes", ascending=False)
)

fig_state = px.bar(
    state_production,
    x="state",
    y="production_tonnes",
    color="state",
    title="Agricultural Production by State",
    labels={
        "state": "State",
        "production_tonnes": "Production (tonnes)"
    },
    template="plotly_dark"
)

st.plotly_chart(fig_state, use_container_width=True)


# --------------------------------------------------
# RAINFALL AND CROP YIELD
# --------------------------------------------------

st.subheader("🌧️ Rainfall and Crop Yield")

fig_rainfall = px.scatter(
    filtered_df,
    x="rainfall_mm",
    y="yield_tonnes_per_hectare",
    color="crop",
    hover_data=[
        "state",
        "year",
        "temperature_c",
        "soil_ph"
    ],
    title="Rainfall vs Crop Yield",
    labels={
        "rainfall_mm": "Rainfall (mm)",
        "yield_tonnes_per_hectare": "Yield (tonnes/hectare)"
    },
    template="plotly_dark"
)

st.plotly_chart(fig_rainfall, use_container_width=True)


# --------------------------------------------------
# YEAR-WISE PRODUCTION
# --------------------------------------------------

st.subheader("📈 Year-wise Production")

year_production = (
    filtered_df.groupby("year", as_index=False)["production_tonnes"]
    .sum()
    .sort_values("year")
)

fig_year = px.line(
    year_production,
    x="year",
    y="production_tonnes",
    markers=True,
    title="Total Production by Year",
    labels={
        "year": "Year",
        "production_tonnes": "Production (tonnes)"
    },
    template="plotly_dark"
)

st.plotly_chart(fig_year, use_container_width=True)


# --------------------------------------------------
# AGRICULTURAL DATASET TABLE
# --------------------------------------------------

st.subheader("📋 Agricultural Dataset")

# Make a separate copy for display.
# Show the readable season names in the table.
display_df = filtered_df.copy()

display_df["season"] = display_df["season_display"]

# Remove the extra internal display column.
display_df = display_df.drop(columns=["season_display"])

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# DOWNLOAD FILTERED DATASET
# --------------------------------------------------

# Download the displayed dataset with readable season names.
csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Dataset (CSV)",
    data=csv_data,
    file_name="filtered_agriculture_data.csv",
    mime="text/csv"
)


# --------------------------------------------------
# PROJECT DISCLAIMER
# --------------------------------------------------

st.caption(
    "Project demonstration using synthetic agricultural data. "
    "Charts and predictions are not verified real-world "
    "agricultural findings."
)


# --------------------------------------------------
# AI CROP YIELD PREDICTION
# --------------------------------------------------

st.divider()

st.header("🤖 AI Crop Yield Prediction")


if not MODEL_PATH.exists():

    st.warning(
        "Trained model not found. Run "
        "python ml/crop_yield_prediction.py first."
    )

else:

    try:
        model = joblib.load(MODEL_PATH)

    except Exception as error:
        st.error(f"Could not load the trained model: {error}")
        st.stop()

    st.write(
        "Enter the agricultural conditions below to estimate "
        "crop yield using the trained Random Forest model."
    )

    with st.form("crop_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            pred_crop = st.selectbox(
                "Crop",
                [
                    "Rice",
                    "Wheat",
                    "Maize",
                    "Cotton",
                    "Sugarcane",
                    "Ragi"
                ]
            )

            pred_state = st.selectbox(
                "State",
                [
                    "Karnataka",
                    "Maharashtra",
                    "Punjab",
                    "Tamil Nadu",
                    "Andhra Pradesh"
                ]
            )

            pred_season = st.selectbox(
                "Season",
                ["Kharif", "Rabi", "Summer"],
                format_func=lambda season: season_names.get(
                    season, season
                )
            )

            pred_year = st.selectbox(
                "Year",
                list(range(2019, 2026)),
                index=6
            )

        with col2:

            pred_soil_ph = st.slider(
                "Soil pH",
                min_value=5.0,
                max_value=8.5,
                value=6.5,
                step=0.1
            )

            pred_rainfall = st.slider(
                "Rainfall (mm)",
                min_value=300.0,
                max_value=1800.0,
                value=1000.0,
                step=10.0
            )

            pred_temperature = st.slider(
                "Temperature (°C)",
                min_value=15.0,
                max_value=40.0,
                value=27.0,
                step=0.5
            )

            pred_humidity = st.slider(
                "Humidity (%)",
                min_value=30.0,
                max_value=95.0,
                value=60.0,
                step=1.0
            )

            pred_area = st.number_input(
                "Cultivated area (hectares)",
                min_value=1.0,
                max_value=10000.0,
                value=10.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "Predict Crop Yield"
        )


    if submitted:

        input_data = pd.DataFrame(
            [
                {
                    "crop": pred_crop,
                    "state": pred_state,
                    "season": pred_season,
                    "year": pred_year,
                    "soil_ph": pred_soil_ph,
                    "rainfall_mm": pred_rainfall,
                    "temperature_c": pred_temperature,
                    "humidity_pct": pred_humidity,
                    "area_hectares": pred_area
                }
            ]
        )

        try:

            predicted_yield = float(
                model.predict(input_data)[0]
            )

            st.success(
                f"Estimated crop yield: "
                f"{predicted_yield:.2f} tonnes/hectare"
            )

            st.caption(
                "Demonstration only: the model was trained on "
                "synthetic data. This estimate is not a verified "
                "agricultural recommendation."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}. "
                "Check that the model was trained using the "
                "same input columns."
            )
