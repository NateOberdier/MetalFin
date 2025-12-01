!pip -q install ipywidgets

from google.colab import output
output.enable_custom_widget_manager()

import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)

from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "/content/fish_heavy_metals_master.csv"

df = pd.read_csv(DATA_PATH)
print("Raw data shape:", df.shape)
display(df.head())

df.columns = [c.strip().lower() for c in df.columns]
print("\nColumns:", df.columns.tolist())

production_map = {
    "atlantic salmon": "farmed",
    "coho salmon": "wild",
    "sockeye salmon": "wild",
    "salmon (generic)": "mixed",
    "pollock": "wild",
    "cod": "wild",
    "haddock": "wild",
    "whiting": "wild",
    "flounder": "wild",
    "sole": "wild",
    "plaice": "wild",
    "tilapia": "farmed",
    "catfish": "farmed",
    "trout": "mixed",
    "perch": "wild",
    "carp": "mixed",
    "walleye": "wild",
    "pike": "wild",
    "bluegill": "wild",
    "whitefish": "wild",
    "bass (freshwater)": "wild",
    "albacore tuna": "wild",
    "skipjack tuna": "wild",
    "bluefin tuna": "wild",
    "bigeye tuna": "wild",
    "yellowfin tuna": "wild",
    "mahi-mahi": "wild",
    "sea bass": "mixed",
    "snapper": "mixed",
    "grouper": "wild",
    "halibut": "wild",
    "sardine": "wild",
    "anchovy": "wild",
    "herring": "wild",
    "swordfish": "wild",
    "shark": "wild",
    "king mackerel": "wild",
    "tilefish (gulf)": "wild",
    "orange roughy": "wild",
    "marlin": "wild",
    "escolar": "wild",
    "opah": "wild",
    "shrimp": "mixed",
    "lobster (american)": "wild",
    "lobster (spiny)": "wild",
    "crab": "wild",
    "crayfish": "wild",
    "oyster": "wild",
    "mussel": "wild",
    "clam": "wild",
    "scallop": "wild",
    "squid": "wild",
    "octopus": "wild",
}

df["species_stripped"] = df["species"].str.lower().str.strip()
df["production"] = df["species_stripped"].map(production_map).fillna("wild")
df.drop(columns=["species_stripped"], inplace=True)

print("\nAdded 'production' column:")
display(df[["species", "production"]].head())

possible_metal_cols = {
    "mercury": ["mercury_mg_kg", "hg_mg_kg", "mercury", "total_mercury"],
    "lead":    ["lead_mg_kg", "pb_mg_kg", "lead"],
    "arsenic": ["arsenic_mg_kg", "as_mg_kg", "arsenic"],
    "cadmium": ["cadmium_mg_kg", "cd_mg_kg", "cadmium"],
}

def find_first_existing(col_candidates, df_cols):
    for c in col_candidates:
        if c in df_cols:
            return c
    return None

metal_cols = {}
for metal, candidates in possible_metal_cols.items():
    col = find_first_existing(candidates, df.columns)
    if col is not None:
        metal_cols[metal] = col

print("\nDetected metal columns:", metal_cols)
if not metal_cols:
    raise ValueError("No heavy metal columns detected. Adjust `possible_metal_cols` to match your CSV.")

metal_col_names = list(metal_cols.values())

for col in metal_col_names:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["has_any_metal"] = df[metal_col_names].notna().any(axis=1)
df = df[df["has_any_metal"]].copy().drop(columns=["has_any_metal"])
print("After dropping rows without any metal data:", df.shape)

potential_cat_features = ["species", "production", "location", "habitat", "diet"]
potential_num_features = ["length_cm", "weight_g"]

cat_features = [c for c in potential_cat_features if c in df.columns]
num_features = [c for c in potential_num_features if c in df.columns]

print("\nCategorical features:", cat_features)
print("Numeric features:    ", num_features)

df_target = df.copy()

Y_reg_all = df_target[metal_col_names].apply(lambda s: pd.to_numeric(s, errors="coerce"))
max_vals = Y_reg_all.max(axis=1)

s_num = pd.to_numeric(max_vals, errors="coerce")
valid_mask = s_num.notna()
s_num = s_num[valid_mask].astype(float)

df_target = df_target.loc[valid_mask].copy()
df_target["max_metal_value"] = s_num

print("\nmax_metal_value dtype:", df_target["max_metal_value"].dtype)

q1, q2 = s_num.quantile([1/3, 2/3])

df_target["risk_label"] = pd.cut(
    s_num,
    bins=[-np.inf, q1, q2, np.inf],
    labels=["low", "medium", "high"],
    include_lowest=True
)

risk_label_mapping = {"low": 0, "medium": 1, "high": 2}
df_target["risk_label_int"] = df_target["risk_label"].map(risk_label_mapping)

print("\nExample targets:")
display(df_target[["species", "production", "max_metal_value", "risk_label"]].head())

X = df_target[cat_features + num_features].copy()
y_reg = df_target["max_metal_value"].astype(float).copy()
y_cls = df_target["risk_label_int"].astype(int).copy()

num_feature_means = {c: float(df_target[c].mean()) for c in num_features}

def safe_train_test_split(X, y_reg, y_cls, test_size, random_state):
    unique_classes, counts = np.unique(y_cls, return_counts=True)
    can_stratify = (len(unique_classes) > 1) and (counts.min() >= 2)
    strat = y_cls if can_stratify else None
    return train_test_split(
        X, y_reg, y_cls,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

X_train_val, X_test, y_reg_train_val, y_reg_test, y_cls_train_val, y_cls_test = safe_train_test_split(
    X, y_reg, y_cls,
    test_size=0.15,
    random_state=RANDOM_STATE
)

X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = safe_train_test_split(
    X_train_val, y_reg_train_val, y_cls_train_val,
    test_size=0.175,
    random_state=RANDOM_STATE
)

print("\nSplit sizes:")
print("  Train:", X_train.shape)
print("  Val:  ", X_val.shape)
print("  Test: ", X_test.shape)

transformers = []
if cat_features:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))
if num_features:
    transformers.append(("num", StandardScaler(), num_features))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

regressor = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
regression_pipeline = Pipeline(
    steps=[("preprocess", preprocessor),
           ("model", regressor)]
)

print("\nTraining regression model (max metal concentration)...")
regression_pipeline.fit(X_train, y_reg_train)

y_reg_val_pred = regression_pipeline.predict(X_val)
mae_val = mean_absolute_error(y_reg_val, y_reg_val_pred)
mse_val = mean_squared_error(y_reg_val, y_reg_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_reg_val, y_reg_val_pred)

print("\n[Regression] Validation metrics:")
print(f"  MAE  (mg/kg): {mae_val:.4f}")
print(f"  RMSE (mg/kg): {rmse_val:.4f}")
print(f"  R²          : {r2_val:.4f}")

classifier = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
classification_pipeline = Pipeline(
    steps=[("preprocess", preprocessor),
           ("model", classifier)]
)

print("\nTraining classification model (risk label low/medium/high)...")
classification_pipeline.fit(X_train, y_cls_train)

y_cls_val_pred = classification_pipeline.predict(X_val)
print("\n[Classification] Validation report:")
print(classification_report(
    y_cls_val, y_cls_val_pred,
    zero_division=0
))

classes_val = np.unique(np.concatenate([y_cls_val, y_cls_val_pred]))
cm_val = confusion_matrix(y_cls_val, y_cls_val_pred, labels=classes_val)
sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues")
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\n=== FINAL TEST EVALUATION ===")

y_reg_test_pred = regression_pipeline.predict(X_test)
mae_test = mean_absolute_error(y_reg_test, y_reg_test_pred)
mse_test = mean_squared_error(y_reg_test, y_reg_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_reg_test, y_reg_test_pred)

print("\n[Regression] Test metrics:")
print(f"  MAE  (mg/kg): {mae_test:.4f}")
print(f"  RMSE (mg/kg): {rmse_test:.4f}")
print(f"  R²          : {r2_test:.4f}")

y_cls_test_pred = classification_pipeline.predict(X_test)
print("\n[Classification] Test report:")
print(classification_report(
    y_cls_test, y_cls_test_pred,
    zero_division=0
))

classes_test = np.unique(np.concatenate([y_cls_test, y_cls_test_pred]))
cm_test = confusion_matrix(y_cls_test, y_cls_test_pred, labels=classes_test)
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens")
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

species_stats = df_target.groupby("species")[metal_col_names + ["max_metal_value"]].mean()
species_stats = species_stats.reset_index().set_index("species")
for col in metal_col_names + ["max_metal_value"]:
    species_stats[col] = pd.to_numeric(species_stats[col], errors="coerce")

ratio_cols = {}
for metal, col in metal_cols.items():
    ratio_col = f"{col}_ratio_to_max"
    species_stats[ratio_col] = species_stats[col] / species_stats["max_metal_value"]
    species_stats[ratio_col] = species_stats[ratio_col].clip(lower=0, upper=1)
    ratio_cols[metal] = ratio_col

print("\nSpecies-level stats and ratios ready for interactive tool.")
