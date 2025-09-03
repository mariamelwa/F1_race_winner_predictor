# 🏎️ F1 Race Winner Predictor

This project predicts Formula 1 race winner probabilities using **FastF1 timing data** and **machine learning (scikit-learn RandomForest)**.  
It trains on multiple past seasons, integrates **automatic weather forecasts** (via Open-Meteo), and learns simple **tyre-strategy priors** from historical laps.

---

##Features
- Multi-season training across past F1 seasons
- Automatic weather integration:
  - Forecast from Open-Meteo (geocoding + hourly)
  - Fallback to last year’s race weather if unavailable
- Tyre strategy priors: start compound distribution, pit stops, stint lengths
- RandomForest ML model with grouped cross-validation by race
- Interactive **Streamlit UI** to select training years and predict upcoming races

---

## Installation

Make sure you have **Python 3.9+** installed.

### 1. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

### 2. Install dependencies

```bash
python -m pip install streamlit fastf1 scikit-learn pandas numpy requests tzdata pyarrow
```

## Usage

### 1. Activate your virtual environment (Windows example):

```bash
.\.venv\Scripts\Activate
```

### 2. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

### 3. Open the local URL from the terminal (default: http://localhost:8501) in your browser


## Output

• Ranked table of drivers with win probabilities.

• Weather details used (forecast or fallback).

• Learned tyre strategy priors (compound distribution, pit stops, stint lengths)

## Notes

• If a target year’s qualifying session isn’t available yet, the app maps last year’s quali performance onto the projected lineup

• Weather forecasts depend on Open-Meteo API availability

• **Disclaimer:** This is a personal project. It is not an official Formula 1 predictor. Predictions are experimental and for educational purposes only.
