import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_DEFAULT_PATH = Path(
    r"c:\Users\tumar\Desktop\infy internship\exoplanet_raw.csv"
)

TARGET_COLUMN = "habitability_label"

# Core feature names expected by the rest of the pipeline
NUMERIC_FEATURES: List[str] = [
    "planet_radius",
    "planet_mass",
    "orbital_period",
    "equilibrium_temp",
    "distance_from_star",
    "stellar_temp",
    "luminosity",
    "metallicity",
]
CAT_FEATURES: List[str] = ["star_type"]


def load_raw_data(path: Path | str = DATASET_DEFAULT_PATH) -> pd.DataFrame:
    """Load the exoplanet dataset from disk.

    The NASA Exoplanet Archive CSV starts with many commented metadata lines.
    We use the `comment` argument so pandas skips them and reads the true header.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Dataset file not found at '{path}'. Please check the path."
        raise FileNotFoundError(msg)

    df = pd.read_csv(path, comment="#")
    return df


def explore_data(df: pd.DataFrame, *, max_rows: int = 5) -> None:
    """Print basic exploratory information about the dataset."""
    print("\n=== First rows ===")
    print(df.head(max_rows))

    print("\n=== Shape (rows, cols) ===")
    print(df.shape)

    print("\n=== Data types ===")
    print(df.dtypes)

    print("\n=== Missing values per column ===")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n=== Basic statistics (numeric columns) ===")
    print(df.describe(include=[np.number]).T)


def _map_nasa_columns_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map NASA Exoplanet Archive column names to the project feature names.

    The raw table typically uses ps/pscomppars-style naming, e.g.:
    - pl_rade    -> planet_radius  (Earth radii)
    - pl_bmasse  -> planet_mass    (Earth masses)
    - pl_orbper  -> orbital_period (days)
    - pl_eqt     -> equilibrium_temp (K)
    - pl_orbsmax -> distance_from_star (AU, semi-major axis)
    - st_teff    -> stellar_temp (K)
    - st_lum     -> luminosity (log10(L/Lsun))
    - st_met     -> metallicity (dex)
    - st_spectype -> star_type (spectral classification)
    """
    column_map: dict[str, str] = {}

    if "pl_rade" in df.columns and "planet_radius" not in df.columns:
        column_map["pl_rade"] = "planet_radius"
    if "pl_bmasse" in df.columns and "planet_mass" not in df.columns:
        column_map["pl_bmasse"] = "planet_mass"
    if "pl_orbper" in df.columns and "orbital_period" not in df.columns:
        column_map["pl_orbper"] = "orbital_period"
    if "pl_eqt" in df.columns and "equilibrium_temp" not in df.columns:
        column_map["pl_eqt"] = "equilibrium_temp"

    # Prefer semi-major axis as planet–star distance if available.
    if "pl_orbsmax" in df.columns and "distance_from_star" not in df.columns:
        column_map["pl_orbsmax"] = "distance_from_star"
    elif "sy_dist" in df.columns and "distance_from_star" not in df.columns:
        # Fallback: system distance from Earth (pc). Not physically identical,
        # but still a distance-related proxy if semi-major axis is missing.
        column_map["sy_dist"] = "distance_from_star"

    if "st_teff" in df.columns and "stellar_temp" not in df.columns:
        column_map["st_teff"] = "stellar_temp"
    if "st_lum" in df.columns and "luminosity" not in df.columns:
        column_map["st_lum"] = "luminosity"
    if "st_met" in df.columns and "metallicity" not in df.columns:
        column_map["st_met"] = "metallicity"
    if "st_spectype" in df.columns and "star_type" not in df.columns:
        column_map["st_spectype"] = "star_type"

    if column_map:
        df = df.rename(columns=column_map)

    return df


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: numeric -> median, categorical -> most frequent."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) > 0:
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)

    if len(categorical_cols) > 0:
        modes = df[categorical_cols].mode().iloc[0]
        df[categorical_cols] = df[categorical_cols].fillna(modes)

    return df


def _clip_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Clip obvious outliers using the IQR rule on numeric columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col]
        if series.nunique(dropna=True) < 10:
            # Skip low-cardinality or quasi-categorical numerics.
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = series.clip(lower=lower, upper=upper)

    return df


def _score_component(value: float, ideal: float, tolerance: float) -> float:
    """Return a simple [0, 1] score for how close a value is to an ideal."""
    if pd.isna(value):
        return np.nan
    if tolerance <= 0:
        return 0.0
    score = 1.0 - abs(value - ideal) / tolerance
    return float(np.clip(score, 0.0, 1.0))


def _compute_habitability_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Create Habitability Score Index and Stellar Compatibility Index."""
    df = df.copy()

    # Habitability Score Index (HSI): combines temperature, radius, distance.
    temp_scores = df["equilibrium_temp"].apply(
        _score_component,
        ideal=288.0,
        tolerance=80.0,
    )
    radius_scores = df["planet_radius"].apply(
        _score_component,
        ideal=1.0,
        tolerance=1.5,
    )
    distance_scores = df["distance_from_star"].apply(
        _score_component,
        ideal=1.0,
        tolerance=1.0,
    )

    hsi = 0.4 * temp_scores + 0.3 * radius_scores + 0.3 * distance_scores
    df["habitability_score_index"] = hsi

    # Stellar Compatibility Index (SCI): sun-like temperature and luminosity.
    stellar_temp_scores = df["stellar_temp"].apply(
        _score_component,
        ideal=5778.0,
        tolerance=1500.0,
    )
    # st_lum is log10(L/Lsun); ideal sun-like star is 0.
    luminosity_scores = df["luminosity"].apply(
        _score_component,
        ideal=0.0,
        tolerance=0.75,
    )
    sci = 0.6 * stellar_temp_scores + 0.4 * luminosity_scores
    df["stellar_compatibility_index"] = sci

    return df


def _ensure_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a binary habitability label exists; create rule-based if absent.

    Rules for habitability_label = 1:
    - equilibrium_temp between 240 and 320 Kelvin (inclusive)
    - planet_radius between 0.5 and 2.5 Earth radii
    - distance_from_star between 0.75 and 2.0 (AU-like scale)
    Else: 0
    """
    df = df.copy()

    missing_core = [col for col in NUMERIC_FEATURES if col not in df.columns]
    if missing_core:
        msg = (
            "Missing required feature columns for target rule construction: "
            f"{missing_core}"
        )
        raise ValueError(msg)

    if TARGET_COLUMN in df.columns:
        # Ensure it is binary {0,1}
        unique_vals = sorted(df[TARGET_COLUMN].dropna().unique())
        if not set(unique_vals).issubset({0, 1}):
            warnings.warn(
                f"Target column '{TARGET_COLUMN}' has non-binary values: "
                f"{unique_vals}. Proceeding as-is.",
                stacklevel=1,
            )
        return df

    temp_ok = df["equilibrium_temp"].between(240.0, 320.0, inclusive="both")
    radius_ok = df["planet_radius"].between(0.5, 2.5, inclusive="both")
    distance_ok = df["distance_from_star"].between(0.75, 2.0, inclusive="both")

    habitable_mask = temp_ok & radius_ok & distance_ok
    df[TARGET_COLUMN] = habitable_mask.astype(int)

    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning + feature-engineering pipeline on the raw dataframe.

    Steps:
    - Map NASA-style column names onto the project feature schema.
    - Remove duplicate rows.
    - Fill missing values by type.
    - Clip obvious numeric outliers.
    - Derive habitability-related indices.
    - Ensure the binary target label exists (or create rule-based).
    """
    df = _map_nasa_columns_to_features(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df = _fill_missing_values(df)
    df = _clip_outliers_iqr(df)
    df = _compute_habitability_indices(df)
    df = _ensure_target_label(df)

    # Ensure required training columns exist
    missing = [col for col in NUMERIC_FEATURES + CAT_FEATURES if col not in df.columns]
    if missing:
        msg = (
            "The following required feature columns are missing after preprocessing: "
            f"{missing}"
        )
        raise ValueError(msg)

    # Normalise star_type to string/categorical
    df["star_type"] = df["star_type"].astype(str)

    return df


def get_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a preprocessed dataframe into feature matrix X and target y."""
    if TARGET_COLUMN not in df.columns:
        msg = f"Target column '{TARGET_COLUMN}' not present in dataframe."
        raise ValueError(msg)

    feature_cols = NUMERIC_FEATURES + CAT_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y


def build_preprocessing_transformer() -> ColumnTransformer:
    """Build the reusable sklearn ColumnTransformer for preprocessing.

    - Numeric features: median imputation + StandardScaler
    - Categorical features: most-frequent imputation + OneHotEncoder
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )

    return preprocessor


def build_model_pipeline(model) -> Pipeline:
    """Create a full sklearn Pipeline (preprocessing + model)."""
    preprocessor = build_preprocessing_transformer()
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def load_and_prepare_data(
    path: Path | str = DATASET_DEFAULT_PATH,
) -> Tuple[pd.DataFrame, pd.Series]:
    """High-level helper: load raw CSV, clean/engineer, and return (X, y)."""
    df_raw = load_raw_data(path)
    explore_data(df_raw)
    df_processed = clean_and_engineer_features(df_raw)
    X, y = get_features_and_target(df_processed)
    return X, y


__all__ = [
    "DATASET_DEFAULT_PATH",
    "TARGET_COLUMN",
    "NUMERIC_FEATURES",
    "CAT_FEATURES",
    "load_raw_data",
    "explore_data",
    "clean_and_engineer_features",
    "get_features_and_target",
    "build_preprocessing_transformer",
    "build_model_pipeline",
    "load_and_prepare_data",
]

