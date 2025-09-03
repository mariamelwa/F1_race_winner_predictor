
# Predict race winner probabilities with FastF1 + ML + automatic weather
# - Trains on multiple past seasons 
# - Auto weather:
#     • Open-Meteo geocoding + forecast 
#     • If forecast isn't available yet, uses last year's race-start weather
# - Tyre strategy prior learned from historical laps 



#  Imports 
import os  
import warnings  
warnings.filterwarnings("ignore")  

import math  
from datetime import datetime, timedelta  

import numpy as np  # numeric arrays and random choice
import pandas as pd  # data tables
import requests  
import streamlit as st  # web app UI
import fastf1  # F1 timing data access

from sklearn.model_selection import GroupKFold  # CV that keeps whole races together
from sklearn.compose import ColumnTransformer  # split numeric,categorical pipelines
from sklearn.pipeline import Pipeline  # chain preprocessing + model
from sklearn.impute import SimpleImputer  # fill missing values
from sklearn.preprocessing import OneHotEncoder  # encode categories
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import roc_auc_score  



#  Setup and cache 
os.makedirs("cache", exist_ok=True) 
fastf1.Cache.enable_cache("cache")  

st.set_page_config(page_title="F1 Race Winner Predictor", layout="wide")  
st.title("🏁 F1 Race Winner Predictor (FastF1 + ML)")  # page header



# Function: timedelta_to_seconds
# Converts a pandas/Python timedelta  to seconds (float)
def timedelta_to_seconds(lap_time):

    """Turn a pandas/py timedelta into plain seconds (float), or NaN if missing."""

    return lap_time.total_seconds() if pd.notnull(lap_time) else np.nan  # convert safely

# Cache this function’s results so repeated calls for the same year reuse the result
@st.cache_data(show_spinner=False)


# Function: get_schedule
# Fetch the season’s race schedule from FastF1 and return a trimmed DataFrame
# with key fields (round, event name, dates, location)
def get_schedule(year: int) -> pd.DataFrame:

    """Return a trimmed event schedule for a given season (columns can vary by year)."""
    sch = fastf1.get_event_schedule(year, include_testing=False).copy()  # grab schedule, ignore testing sessions
   
    # Keep only columns we actually need, and only if they exist for this season
    keep = [c for c in [
        "RoundNumber", "EventName", "EventDate", "Country", "Location",
        "Session1Date", "Session2Date", "Session3Date", "Session4Date", "Session5Date",
        "RaceDate"
    ] if c in sch.columns]  # avoid KeyErrors if FastF1 changes schema
    return sch[keep]  # trimmed schedule



# Function: race_dt_from_schedule_row
# Determine the most likely local race start time from a FastF1 schedule row
# Priority:
#   1) Use explicit session timestamps if present 
#   2) Otherwise, take EventDate and assume Sunday at 15:00 local time
#   3) If EventDate is missing, fall back to “today at 15:00”
# Returns a plain Python datetime
def race_dt_from_schedule_row(row: pd.Series) -> datetime:

    """Pick the best timestamp for race start from a schedule row; fall back to Sun 15:00."""
    
    # prefer explicit race/session fields
    for c in ("Session5Date", "RaceDate", "Session4Date"):  
        if c in row and pd.notnull(row[c]):  
            return pd.to_datetime(row[c]).to_pydatetime()  # normalize  pandas/ISO to  datetime

    # Otherwise use the general EventDate  
    base = pd.to_datetime(row.get("EventDate", None))  
    
     # if even that is missing, use “now at 15:00” as a last-resort
    if pd.isna(base):  
        return datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)
    
     # Assume Sunday 15:00 local if only EventDate is known
    base = base.to_pydatetime()  # pandas to python datetime
    dt = base.replace(hour=15, minute=0, second=0, microsecond=0)  
    
    # push forward to Sunday (6)
    while dt.weekday() != 6:  
        dt += timedelta(days=1)
    
    return dt  # best-guess race time



#  Weather helpers

# Function: _geocode_open_meteo
# Purpose: Given a place name, query Open-Meteo’s geocoder
# and return (latitude, longitude); returns (None, None) if no match or on error
def _geocode_open_meteo(name: str):

    """Return (lat, lon) for a place name using Open-Meteo geocoding, or (None, None) on failure."""
    
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"  # endpoint for place search
        r = requests.get(url, params={"name": name, "count": 1}, timeout=10)  # 1 best match, quick timeout
        
        if r.ok:  # only proceed on HTTP 2xx
            data = r.json()  # parse JSON body
            res = data.get("results", [])  # safely get 'results' 
            
            if res:  
                return float(res[0]["latitude"]), float(res[0]["longitude"])  # take first match
    
    except Exception:
        pass  
    return None, None  # signal failure


# Function: _forecast_open_meteo
# Purpose: Given (lat, lon) and a target local datetime, fetch Open-Meteo hourly data
# and return (air_temp_C, rain_flag, "Open-Meteo forecast"); on failure, return
# (None, None, reason)
def _forecast_open_meteo(lat: float, lon: float, when_local: datetime):

    """Return (air_temp_C, rain_flag, source_str) or (None, None, 'reason') if we can't."""
    
    try:
        url = "https://api.open-meteo.com/v1/forecast"  # hourly forecast API
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation",  
            "timezone": "auto"  # get local timestamps
        }
        r = requests.get(url, params=params, timeout=10)  # quick call
        
        if not r.ok:  
            return None, None, f"forecast HTTP {r.status_code}"
        js = r.json()  # parse JSON
        
        # pull arrays safely
        times = js.get("hourly", {}).get("time", [])
        temps = js.get("hourly", {}).get("temperature_2m", [])
        precs = js.get("hourly", {}).get("precipitation", [])

        if not times:  
            return None, None, "no hourly times"
        
        # find the forecast hour closest to our race start
        t_idx, t_min = None, 1e12  # keep the best index and delta
        
        for i, iso in enumerate(times):
            try:
                dt = datetime.fromisoformat(iso)  # ISO string to datetime

            except Exception:
                continue  # skip malformed timestamps
            
            # For each forecast hour, measure how far it is from the race start time
            d = abs((dt - when_local).total_seconds())  # time gap in seconds

            # If this hour is the closest we've seen so far, remember it
            if d < t_min:  
                t_min = d # update the smallest gap
                t_idx = i # and store the index of that best hour

        # After checking all hours, if we never found a valid index, give up
        if t_idx is None:  
            return None, None, "no matching hour"
        
        # Read the temperature at the chosen hour
        air_temp = float(temps[t_idx]) if t_idx < len(temps) else None  

        # Read the precipitation at the chosen hour
        precipitation = float(precs[t_idx]) if t_idx < len(precs) else 0.0  
        
        # Turn precipitation into a simple flag: 1 = likely rain, 0 = no rain
        rain_flag = 1 if (precipitation is not None and precipitation > 0.2) else 0  

        return air_temp, rain_flag, "Open-Meteo forecast" 
    
    except Exception as e:
        return None, None, f"forecast error: {e}"  # network / parse issue


# Get race-day weather
# Steps:
# 1) Look up the event (location and date) from the template year's schedule
# 2) Try to geocode the location  and pull an hourly forecast near race time
# 3) If a forecast isn’t available (too far in the future), fall back to last year’s race weather
# Returns: (AirTemp°C, TrackTemp°C, Rain(0/1), source_str)
def auto_weather_for_event(template_year: int, round_no: int):

    """Try forecast for the event location/time; else fall back to last year's race weather."""
    
    sch = get_schedule(template_year)  #find  schedule of our template year
    
    row = sch.loc[sch["RoundNumber"] == round_no]  # pick the row for the chosen round
    
    if row.empty:  # if the round number isn't found
        return 25.0, 35.0, 0, "No schedule row"  # return defaults so the app still works
    
    # Turn it into a Series for easy field access
    row = row.iloc[0]  #
    location = row.get("Location", None)  
    country = row.get("Country", None)  
    event = row.get("EventName", "Grand Prix")  

    race_dt = race_dt_from_schedule_row(row)  # best-guess race time

    # Try a few query strings to geocode the place
    if pd.notna(location) and pd.notna(country):
        queries = [f"{location}, {country}", f"{event} {country}", f"{location}"]
    
    elif pd.notna(location):
        queries = [f"{location}", f"{event}"]
    
    else:
        queries = [f"{event}"]

    # Try each query until we get latitude/longitude
    lat = lon = None
    for q in queries:
        lat, lon = _geocode_open_meteo(q)
        if lat is not None:
            break

    # If we got coordinates, try to fetch an hourly forecast near race time
    if lat is not None:
        air, rain, src = _forecast_open_meteo(lat, lon, race_dt)
        if air is not None:

            # Very rough track temp estimate from air temp:
            # add ~10°C if dry, ~5°C if raining
            track = float(air) + (5.0 if rain else 10.0)  

            return float(air), float(track), int(rain), src

    # Otherwise, fall back to last year's race weather snapshot from FastF1
    try:
        r = fastf1.get_session(template_year, int(round_no), "R")  # load the race
        r.load(weather=True, laps=False)  # we only need weather rows
        w = r.weather_data  # pandas table of weather samples

        if (w is not None) and (len(w) > 0):  # if anything came back
            
            # Use the first weather row as a proxy for race start
            air = float(w["AirTemp"].iloc[0]) if "AirTemp" in w else np.nan  

            trk = float(w["TrackTemp"].iloc[0]) if "TrackTemp" in w else (
                float(air) + 10.0 if not math.isnan(air) else np.nan  # rough fill if missing
            )
            rain = 0

            if "Rainfall" in w:
                rain = int((w["Rainfall"] > 0).any())  # did it rain at all?

            return air, trk, rain, "Last year's race weather (fallback)"
    
    except Exception:
        pass  # Any error: return defaults below

    # Hard default if nothing else worked
    return 25.0, 35.0, 0, "Default weather (fallback)"



#  Data building (quali + race context) 

# Build per-driver qualifying features (grid + best of Q1/Q2/Q3 in seconds)
def fetch_quali_features(year: int, round_no: int) -> pd.DataFrame:

    """Per-driver quali stats: grid position and best of Q1/Q2/Q3 in seconds."""

    q = fastf1.get_session(year, round_no, "Q")  # qualifying session
    q.load()  # download/use cache

    cols = ["FullName", "TeamName", "Position", "Q1", "Q2", "Q3"]  # columns we care about
    df = q.results[cols].rename(
        columns={"FullName": "Driver", "Position": "GridPosition"}
    ).copy()  

    for col in ["Q1", "Q2", "Q3"]:  # convert each quali segment time to seconds
        df[col] = df[col].apply(timedelta_to_seconds)

    df["BestQuali_sec"] = df[["Q1", "Q2", "Q3"]].min(axis=1, skipna=True)  # best of the three
    return df[["Driver", "TeamName", "GridPosition", "BestQuali_sec"]]  


# Build race labels (Winner) and context (race grid if present, start tyre, weather)
def fetch_race_labels_and_context(year: int, round_no: int) -> pd.DataFrame:

    """Per-driver pre-race context + label: grid (race if present), start tyre, weather, Winner."""
    r = fastf1.get_session(year, round_no, "R")  # race session
    r.load(laps=True, weather=True)  # we need laps (start tyre) + weather

    res_cols = ["FullName", "TeamName", "Position"]  # classification
    
    if "GridPosition" in r.results.columns:  # use race grid if available
        res_cols.append("GridPosition")
    
    res = r.results[res_cols].rename(columns={"FullName": "Driver"}).copy()
    
    res["Winner"] = (res["Position"] == 1).astype(int)  # label: 1 if P1
    
    if "GridPosition" in res.columns:  # rename race grid to avoid collision
        res = res.rename(columns={"GridPosition": "GridPosition_r"})

    # Starting compound from Lap 1 per driver
    laps = r.laps  # all laps
    start_comp = (
        laps.sort_values(["Driver", "LapNumber"])  # earliest lap first
            .groupby("Driver", as_index=False)  # per driver
            .first()[["Driver", "Compound"]]  # grab first lap's compound
            .rename(columns={"Compound": "StartCompound"})  
    )

    # Weather snapshot 
    air = track = np.nan  # init
    rain = 0

    try:
        w = r.weather_data  # weather samples

        if (w is not None) and (len(w) > 0):  
            air = float(w["AirTemp"].iloc[0]) if "AirTemp" in w else np.nan  # deg C
            track = float(w["TrackTemp"].iloc[0]) if "TrackTemp" in w else np.nan  # deg C
            
            if "Rainfall" in w:
                rain = int((w["Rainfall"] > 0).any())  # any sign of rain
    
    except Exception:
        pass  # keep NaNs if weather missing

    # Merge start compound into classification and add weather fields
    res = res.merge(start_comp, on="Driver", how="left")
    res["AirTemp"] = air
    res["TrackTemp"] = track
    res["Rain"] = rain

    # Return the fields the model needs for training
    return res[["Driver", "TeamName", "GridPosition_r", "StartCompound",
                "AirTemp", "TrackTemp", "Rain", "Winner"]]


# Join quali features + race context into one per-event training table
def build_event_table(year: int, round_no: int, event_name: str) -> pd.DataFrame:

    """Join quali features with race context for one event."""

    q = fetch_quali_features(year, round_no)  # grid + best quali
    r = fetch_race_labels_and_context(year, round_no)  # start tyre + weather + label
    df = q.merge(r, on=["Driver", "TeamName"], how="inner")  # inner join on driver+team
    df["GridPosition"] = df["GridPosition_r"].fillna(df["GridPosition"]).astype(float)  # prefer race grid
    df = df.drop(columns=["GridPosition_r"], errors="ignore")  
    df["EventName"] = event_name  # add event label
    df["Year"] = year  
    df["Round"] = round_no  
    df["EventKey"] = f"{year:04d}_{round_no:02d}"  # used for grouped CV
    return df  # final per-event table

# Cache the  multi-season build so it only recomputes when inputs change
@st.cache_data(show_spinner=True)


# Build a combined training dataset across multiple F1 seasons
# For each season/year provided:
#   - Load the official race schedule
#   - Iterate through every round (Grand Prix)
#   - Try to fetch event-level qualifying + race context
#   - Collect per-event driver tables into one DataFrame
def build_training_set(years: tuple[int, ...]) -> pd.DataFrame:
    
    """Build training rows across seasons; skip sessions that fail to load."""
    
    all_rows = []  # collect per-event tables
    
    for y in years:  
        sch = get_schedule(y)  # schedule for that season
        
        for rnd in sch["RoundNumber"].tolist():  # loop each round
            
            try:
                ev_name = sch.loc[sch["RoundNumber"] == int(rnd), "EventName"].iloc[0]  # label
                tbl = build_event_table(y, int(rnd), ev_name)  # join quali + race context
                all_rows.append(tbl)  
                st.write(f"✓ built {y} R{rnd:02d} ({len(tbl)} drivers)")  # progress log
            
            except Exception as e:
                st.write(f"✗ skip {y} R{rnd:02d}: {e}")  # skip broken sessions 

    if not all_rows:  # if nothing loaded at all
        raise RuntimeError("No training data could be built.")  # stop the app
    return pd.concat(all_rows, ignore_index=True)  # big training DataFrame

#  Learn simple tyre-strategy priors
@st.cache_data(show_spinner=True)


# Analyze historical races at a given circuit/round across multiple seasons
# to estimate simple tyre-strategy priors:
#   • Distribution of starting compounds (Soft/Medium/Hard/Inter/Wet)
#   • Typical number of pit-stops per driver (median)
#   • Median stint length (laps) for each compound
# Returns a dict of small DataFrames for use in modeling or visualization
def learn_circuit_strategy_prior(years: tuple[int, ...], round_no: int) -> dict:

    """
    Learn simple tyre-strategy priors for this circuit/round from history:
      - start_compound_dist: probability of SOFT/MEDIUM/HARD/INTERMEDIATE/WET at race start
      - typical_pitstops: median number of pit-stops per driver
      - median_stint_laps_by_compound: median laps per stint for each compound
    """

    all_starts, all_stops, all_stints = [], [], []  # gather across seasons
    
    for y in years:  # loop seasons
        try:
            r = fastf1.get_session(y, round_no, "R")  # race
            r.load(laps=True, weather=False)  # laps only needed here
            laps = r.laps.copy()  
            
            if laps.empty:  # nothing to learn from
                continue

            # determine Starting compound 
            first_lap = (laps.sort_values(["Driver", "LapNumber"])
                             .groupby("Driver", as_index=False)
                             .first())
            
            if "Compound" in first_lap:
                all_starts.extend(first_lap["Compound"].dropna().str.upper().tolist())

            # Infer number of pit-stops: prefer 'Stint' if present, else detect compound changes
            df = laps[["Driver", "LapNumber", "Compound"]].copy()
            df["Compound"] = df["Compound"].str.upper()

            if "Stint" in laps.columns and laps["Stint"].notna().any():
                # Count unique stints → infer number of stops
                stops = (laps.dropna(subset=["Stint"])
                              .groupby("Driver")["Stint"].nunique()
                              .rename("NumStints")
                              .reset_index())
                stops["PitStops"] = (stops["NumStints"] - 1).clip(lower=0)
            
            else:
                # No 'Stint' info → detect stints by compound changes
                df = df.sort_values(["Driver", "LapNumber"])
                df["new_stint_flag"] = (df.groupby("Driver")["Compound"]
                                          .apply(lambda s: s.ne(s.shift()).astype(int)))
                stints_per_driver = (df.groupby("Driver")["new_stint_flag"]
                                       .sum().rename("NumStints").reset_index())
                stints_per_driver["PitStops"] = (stints_per_driver["NumStints"] - 1).clip(lower=0)
                stops = stints_per_driver
            all_stops.extend(stops["PitStops"].tolist())

            # Median stint length per compound
            if "Stint" in laps.columns and laps["Stint"].notna().any():
                stint_grp = (laps.dropna(subset=["Stint", "Compound"])
                                  .groupby(["Driver", "Stint"]))
                stint_sizes = (stint_grp.size().reset_index(name="StintLen"))
                stint_comp = (stint_grp["Compound"].first().reset_index(name="Compound"))
                st = stint_sizes.merge(stint_comp, on=["Driver", "Stint"])
            
            else:
                # synthesize stints by changes in compound
                df2 = df.copy()
                df2["stint_id"] = (df2.groupby("Driver")["Compound"]
                                     .apply(lambda s: s.ne(s.shift()).cumsum()))
                st = (df2.groupby(["Driver", "stint_id"])
                        .agg(StintLen=("LapNumber", "count"),
                             Compound=("Compound", "first"))
                        .reset_index(drop=True))
            all_stints.append(st[["Compound", "StintLen"]])  # collect stint info
        
        except Exception:
            continue  # skip races that fail to load

    # Aggregate starting compounds into a probability distribution
    start_series = pd.Series([c for c in all_starts if isinstance(c, str)])
    start_dist = (start_series.value_counts(normalize=True)
                  .rename_axis("Compound").reset_index(name="Probability"))
    
    all_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]  # ensure all present
    start_dist = (start_dist.set_index("Compound")
                          .reindex(all_compounds, fill_value=0.0)
                          .reset_index())

    # Typical pit-stops = median across drivers/races
    stops_median = float(np.median(all_stops)) if all_stops else np.nan
    stops_df = pd.DataFrame({"Metric": ["MedianPitStops"], "Value": [stops_median]})

    # Median stint length by compound
    if all_stints:
        stints_df = pd.concat(all_stints, ignore_index=True)
        stint_med = (stints_df.dropna(subset=["Compound"])
                              .groupby(stints_df["Compound"].str.upper())["StintLen"]
                              .median()
                              .rename("MedianStintLaps")
                              .reset_index()
                              .rename(columns={"Compound": "Compound"}))
        stint_med = (stint_med.set_index("Compound")
                              .reindex(all_compounds)
                              .reset_index())
    else:
        stint_med = pd.DataFrame({"Compound": all_compounds,
                                  "MedianStintLaps": [np.nan]*len(all_compounds)})


    # Return small dataframes for easy use/display
    return {
        "start_compound_dist": start_dist,
        "typical_pitstops": stops_df,
        "median_stint_laps_by_compound": stint_med
    }

# Model training 
@st.cache_resource(show_spinner=True)

# Train a winner-probability model from race-level features
# Final: fit on all rows and return the trained pipeline + feature names
def train_winner_model(_train_df: pd.DataFrame):

    """
    Train a RandomForest on:
      - numeric: GridPosition, BestQuali_sec, AirTemp, TrackTemp, Rain
      - categorical: EventName, TeamName, StartCompound, Driver
    Grouped cross-validation by EventKey, then fit on all rows.
    (Leading underscore prevents Streamlit hashing this potentially big object.)
    """

    df = _train_df.copy()  
    num_feats = ["GridPosition", "BestQuali_sec", "AirTemp", "TrackTemp", "Rain"]  
    cat_feats = ["EventName", "TeamName", "StartCompound", "Driver"]  # categorical inputs

    # Separate features, labels, and grouping key
    X = df[num_feats + cat_feats]  # feature matrix
    y = df["Winner"].astype(int)  # binary target (0 = not winner, 1 = winner)
    groups = df["EventKey"]  # group identifier to keep races intact during CV

    # Build preprocessing: 
    # - numeric: impute missing values with median
    # - categorical: impute with most frequent, then one-hot encode
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_feats),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_feats),
        ],
        remainder="drop"
    )

    # RandomForest baseline classifier for mixed feature types
    clf = RandomForestClassifier(
        n_estimators=600, # number of trees
        max_depth=None, #no limit
        class_weight="balanced",  # handle class imbalance
        random_state=42,
        n_jobs=-1 # use all available CPU cores
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])  # full pipeline

    # Dynamic GroupKFold: if very few events, reduce splits to avoid errors
    n_groups = groups.nunique() # number of unique races
    n_splits = max(2, min(5, n_groups))  # choose between 2–5 folds depending on races
    aucs = []  # store per-fold AUCs

    if n_groups >= 2:

        gkf = GroupKFold(n_splits=n_splits)
        
        for tr, te in gkf.split(X, y, groups):  # split by EventKey
            pipe.fit(X.iloc[tr], y.iloc[tr]) # train on training fold
            proba = pipe.predict_proba(X.iloc[te])[:, 1] # predict probabilities on test fold
            aucs.append(roc_auc_score(y.iloc[te], proba)) # compute AUC score
        st.sidebar.success(f"CV AUC ({n_splits}-fold, by race): {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    
    else:
        st.sidebar.warning("Not enough events for CV; training on all rows.")

    pipe.fit(X, y)  # final fit on all available data
    feature_names = num_feats + cat_feats  # remember feature order
    return pipe, feature_names  # return trained model  and feature metadata

#  2025 lineup
PROJECTED_2025 = {
    "Red Bull Racing": ["Max Verstappen", "Yuki Tsunoda"],
    "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
    "McLaren": ["Lando Norris", "Oscar Piastri"],
    "Mercedes": ["George Russell", "Kimi Antonelli"],
    "Aston Martin": ["Fernando Alonso", "Lance Stroll"],
    "Williams": ["Alexander Albon", "Carlos Sainz"],
    "Kick Sauber": ["Nico Hulkenberg", "Gabriel Bortoleto"],
    "Racing Bulls": ["Liam Lawson", "Isack Hadjar"],
    "Haas": ["Esteban Ocon", "Oliver Bearman"],
    "Alpine": ["Pierre Gasly", "Franco Colapinto"],
}

# Convert a team → drivers mapping into a flat DataFrame.
# Each driver gets one row with columns: Driver, TeamName, EventName
def lineup_df(lineup_dict: dict, event_name: str) -> pd.DataFrame:

    """Turn {Team: [drivers]} into a flat table of Driver | TeamName | EventName."""
    
    rows = []  # collect rows

    for team, drivers in lineup_dict.items():  # loop through teams
        for d in drivers:   # loop through drivers in each team
            rows.append({"Driver": d, "TeamName": team, "EventName": event_name})  # one row per driver
    return pd.DataFrame(rows)  # final table


# Provide a fallback heuristic for choosing a starting tyre if no priors exist
# Rule: Top 10 start on SOFT, others on MEDIUM
def _default_start_tyre(pos: int) -> str:

    """Last-resort starting tyre heuristic if priors are empty."""
    
    return "SOFT" if int(pos) <= 10 else "MEDIUM"  


#  Sidebar controls 
st.sidebar.header("Target event")  # section title

target_year = st.sidebar.number_input(  # which future season to predict
    "Target year", min_value=2025, max_value=2030, value=2025, step=1
)

st.sidebar.header("Training seasons")  # section title
min_train = st.sidebar.number_input("From", min_value=2018, max_value=2024, value=2021, step=1)  # first season
max_train = st.sidebar.number_input("To", min_value=min_train, max_value=2024, value=2024, step=1)  # last season
TRAIN_YEARS = tuple(range(int(min_train), int(max_train) + 1))  # inclusive range as a tuple

# Use latest training year’s schedule as the template for round numbers/circuits
sch_template_year = max_train  # calendar structure anchor
sch = get_schedule(sch_template_year)  # fetch that schedule

# Build nice labels e.g "R21 — São Paulo Grand Prix" for the dropdown
labels = sch.apply(lambda r: f"R{int(r.RoundNumber):02d} — {r.EventName}", axis=1).tolist()
sel = st.sidebar.selectbox("Grand Prix (uses template year's round #)", options=labels)  # pick a GP
sel_round = int(sel.split("—")[0].strip()[1:])  # parse "Rxx" to  xx
event_name = sch.loc[sch["RoundNumber"] == sel_round, "EventName"].iloc[0]  

st.sidebar.caption(
    "Weather is automatic: tries Open-Meteo forecast. "
    "If not available yet, falls back to last year's race weather."
)

#  Build training data and model 
with st.spinner(f"Building training set for {TRAIN_YEARS}…"):  # show spinner while loading
    train_df = build_training_set(TRAIN_YEARS)  

with st.spinner("Training winner model…"):  # train the model
    model, feature_names = train_winner_model(train_df)  # RF pipeline and feature names

#  Learn strategy prior for this circuit 
with st.spinner("Learning strategy priors from history…"):  # compute tyres/pit priors
    priors = learn_circuit_strategy_prior(TRAIN_YEARS, sel_round)  # data-driven priors

start_probs = (priors["start_compound_dist"]
               .set_index("Compound")["Probability"].to_dict())  # dict: compound -> probability



# Randomly sample a starting tyre compound based on learned probabilities
# If the distribution is missing or invalid, default to "SOFT"
def draw_start_tyre(prob_dict: dict) -> str:

    """Sample a starting tyre using the learned distribution; fall back if empty."""
    
    if not prob_dict or sum(prob_dict.values()) == 0:  # nothing learned? use heuristic
        return "SOFT"
    compounds, probs = zip(*prob_dict.items())   # separate labels and weights
    probs = np.array(probs, dtype=float)   # ensure numeric
    probs = probs / probs.sum()  # normalize so sum= 1.0
    return np.random.choice(compounds, p=probs)  # sample one compound

# Build prediction rows 
have_target_quali = False  # tell the UI if we used real quali or an approximation
pred_rows = None  # will hold per-driver prediction features (grid and best quali)

# Try to use target-year actual qualifying (if the race is already loaded)
try:
    q_target = fetch_quali_features(target_year, sel_round)  # attempt real quali

    if not q_target.empty:
        have_target_quali = True  # we got it
        pred_rows = q_target[["Driver", "TeamName", "GridPosition", "BestQuali_sec"]].copy()  # use as-is

except Exception:
    pass  

# If target-year quali isn't available, map last year's quali profile onto 2025 lineups
if pred_rows is None:

    q_last = fetch_quali_features(sch_template_year, sel_round)  # template year's quali
    base = lineup_df(PROJECTED_2025, event_name)  # 2025 drivers by team for this event name
    pred_list = []  # collect rows

    for team, grp in q_last.groupby("TeamName"):  # per team
        g2 = grp.sort_values("BestQuali_sec")  # internal order by pace
        times = g2["BestQuali_sec"].tolist()  # their best times
        
        if team in PROJECTED_2025:  # only if that team exists in 2025 projection
            drivers = PROJECTED_2025[team]  # the two drivers
            
            for i, d in enumerate(drivers):  # assign last year's within-team distribution
                t = times[i % len(times)] if times else np.nan  # handle short lists
                pred_list.append({"Driver": d, "TeamName": team, "BestQuali_sec": t})
    
    pred_rows = pd.DataFrame(pred_list)  # make a table
    pred_rows = pred_rows.sort_values("BestQuali_sec").reset_index(drop=True)  # order by pace
    pred_rows["GridPosition"] = np.arange(1, len(pred_rows) + 1)  #  grid 1 to 20

#  Weather
air, track, rain, weather_src = auto_weather_for_event(sch_template_year, sel_round)  

#  Assemble features expected by the model 
pred_feats = pred_rows.copy()  # copy because we don't want to mutate the original

pred_feats["EventName"] = event_name  # categorical feature

# Use learned prior to pick a plausible start tyre per row; if the prior is empty, default
pred_feats["StartCompound"] = pred_feats["GridPosition"].apply(lambda _: draw_start_tyre(start_probs))
pred_feats["AirTemp"] = float(air) 
pred_feats["TrackTemp"] = float(track)  
pred_feats["Rain"] = int(rain)  

X = pred_feats[feature_names]  # select columns in the exact order the pipeline expects

proba = model.predict_proba(X)[:, 1]  # get P(win) from the classifier

#  Present results (sorted highest probability first)

out = pred_feats[["Driver", "TeamName", "GridPosition", "BestQuali_sec"]].copy()  # base columns
out["P(win)"] = proba  # attach probabilities

# stable sort 
out = out.sort_values(by="P(win)", ascending=False, kind="mergesort").reset_index(drop=True)

out.insert(0, "Rank", np.arange(1, len(out) + 1))  # 1..N ranks

out = out.rename(columns={"GridPosition": "Grid"}) 

out["P(win)"] = out["P(win)"].round(3)  # round for display

st.subheader(f"Predicted win probabilities — {target_year} • {event_name}")  # section title

st.dataframe(  # show the main table
    out[["Rank", "Driver", "TeamName", "Grid", "P(win)"]],
    hide_index=True,
    use_container_width=True
)

#  Extra context panels
left, right = st.columns(2)  # two columns

with left:
    st.markdown("**Quali source used for prediction**")  # small header
    st.write("Target year quali available:", have_target_quali)  # True/False
    st.write("If False, we used last year’s quali at this circuit to estimate the grid/time.")  # note


with right:
    st.markdown("**Weather (automatic)**")  # small header
    st.write(f"Air: {air:.1f} °C  |  Track: {track:.1f} °C  |  Rain flag: {rain}  |  Source: {weather_src}")  # quick summary

# Show the learned strategy priors so we can sanity-check them
st.markdown("### Strategy priors learned from history (for this circuit)")
c1, c2, c3 = st.columns(3)

with c1:
    st.write("**Start compound distribution**")  # header
    st.dataframe(priors["start_compound_dist"], hide_index=True, use_container_width=True)  # probs

with c2:
    st.write("**Typical pit-stops (median)**")  # header
    st.dataframe(priors["typical_pitstops"], hide_index=True, use_container_width=True)  # median

with c3:
    st.write("**Median stint laps by compound**")  # header
    st.dataframe(priors["median_stint_laps_by_compound"], hide_index=True, use_container_width=True)  # stint lens



#  Footer notes 
st.caption(
    "Notes: Model trains only on the seasons you pick. If the target year’s qualifying "
    "isn’t available yet, the app approximates the grid from last year at the same track. "
    "Weather is fetched automatically via Open-Meteo (free). Strategy priors are learned "
    "from historical FastF1 laps for this circuit — no scraping or paid feeds."
)
