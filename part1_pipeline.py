
# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif

DATA_PATH = "hotel_bookings.csv"
TARGET = "is_canceled"
RANDOM_STATE = 42
K_FEATURES = 30
LEAKAGE_AND_DROP = ["reservation_status", "reservation_status_date", "assigned_room_type", "company",
                    "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month"]

cap_cols    = ["lead_time", "adr", "adults", "stays_in_weekend_nights", "stays_in_week_nights",
               "total_of_special_requests", "total_nights", "total_guests"]
skew_cols   = ["days_in_waiting_list", "previous_cancellations", "previous_bookings_not_canceled",
               "booking_changes", "required_car_parking_spaces", "children", "babies"]
binary_cols = ["is_repeated_guest", "has_prev_cancel", "has_children"]
cat_cols    = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
               "reserved_room_type", "deposit_type", "agent", "customer_type", "lead_time_group"]

# %%
def load_split(path=DATA_PATH):
    """Part 1 cleaning (duplicates, zero-guest rows, leakage columns) and the stratified 80/20 split."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    no_guests = (df["adults"] + df["children"].fillna(0) + df["babies"]) == 0
    df = df[~no_guests].reset_index(drop=True)
    df["agent"] = ("agent_" + df["agent"].astype("Int64").astype(str)).where(df["agent"].notna())
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET] + LEAKAGE_AND_DROP)
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# %%
def add_features(data):
    data = data.copy()
    data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data["total_guests"] = data["adults"] + data["children"] + data["babies"]
    data["has_children"] = ((data["children"] > 0) | (data["babies"] > 0)).astype(int)
    data["has_prev_cancel"] = (data["previous_cancellations"] > 0).astype(int)
    data["lead_time_group"] = pd.cut(
        data["lead_time"], bins=[-1, 7, 30, 90, 180, np.inf],
        labels=["0-7", "8-30", "31-90", "91-180", "181+"]
    ).astype(str)
    return data

class QuantileCapper(BaseEstimator, TransformerMixin):
    """Winsorise: clip each column to [low, high] percentiles learned in fit() (train data only)."""
    def __init__(self, low=0.01, high=0.99):
        self.low, self.high = low, high
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.low, axis=0)
        self.upper_ = np.nanquantile(X, self.high, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

# %%
def make_column_transformer():
    """Fresh (unfitted) ColumnTransformer every call, so no fitted state is shared between models."""
    cap_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cap",    QuantileCapper(0.01, 0.99)),
        ("scale",  StandardScaler()),
    ])
    skew_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log",    FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                 drop="if_binary", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("cap",  cap_pipe,  cap_cols),
        ("skew", skew_pipe, skew_cols),
        ("bin",  "passthrough", binary_cols),
        ("cat",  cat_pipe,  cat_cols),
    ])

def make_preprocess(k=K_FEATURES):
    """The Part 1 preprocessing pipeline: features -> columns -> select."""
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("columns",  make_column_transformer()),
        ("select",   SelectKBest(score_func=f_classif, k=k)),
    ])

def make_model_pipeline(model):
    """Preprocessing (Part 1) + any sklearn classifier, in one leakage-free Pipeline."""
    return Pipeline([("preprocess", make_preprocess()), ("model", model)])

def make_cv():
    """The one CV scheme used by every Part 2 member: 5-fold stratified, shuffled."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def selected_feature_names(fitted_preprocess):
    """Names of the features kept by SelectKBest, in the order the model sees them."""
    names = pd.Index(fitted_preprocess.named_steps["columns"].get_feature_names_out())
    names = names.str.replace(r"^(cap|skew|bin|cat)__", "", regex=True)
    return np.asarray(names)[fitted_preprocess.named_steps["select"].get_support()]