import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder

ROOT = Path(__file__).resolve().parent
NUM = ["OverallQual", "GrLivArea", "FirstFlrSF", "TotalBsmtSF", "GarageCars",
       "YearBuilt", "YrSold", "FullBath", "TotRmsAbvGrd"]
CAT = ["Neighborhood", "KitchenQual", "CentralAir"]
FEATURES = NUM + CAT
CONFIG = yaml.safe_load((ROOT / "configs/config.yaml").read_text())


def load_data():
    frame = pd.read_csv(ROOT / "data/ames.csv").rename(columns={"1stFlrSF": "FirstFlrSF"})
    frame = frame[FEATURES + ["SalePrice"]]
    train, rest = train_test_split(frame, test_size=0.30, random_state=CONFIG["random_state"])
    validation, test = train_test_split(rest, test_size=0.50, random_state=CONFIG["random_state"])
    return train, validation, test


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self, added=True):
        self.added = added

    def fit(self, X, y=None):
        self.medians_ = X[NUM].median().fillna(0)
        counts = X["Neighborhood"].value_counts() / len(X)
        self.areas_ = counts[counts >= CONFIG["rare_threshold"]].index
        return self

    def transform(self, X):
        frame = X.copy().replace({None: np.nan})
        for name in NUM:
            frame[name] = frame[name].fillna(0 if name in ["TotalBsmtSF", "GarageCars"] else self.medians_[name])
        rare = frame["Neighborhood"].notna() & ~frame["Neighborhood"].isin(self.areas_)
        frame.loc[rare, "Neighborhood"] = "Other"
        if self.added:
            frame["TotalSF"] = frame["TotalBsmtSF"] + frame["FirstFlrSF"]
            frame["HouseAge"] = frame["YrSold"] - frame["YearBuilt"]
        return frame


class Factory:
    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "none": FunctionTransformer}
    models = {"dummy": lambda: DummyRegressor(strategy="median"),
              "ridge": lambda: Ridge(alpha=CONFIG["alpha"]),
              "forest": lambda: RandomForestRegressor(n_estimators=CONFIG["n_estimators"],
                    random_state=CONFIG["random_state"], max_depth=CONFIG["max_depth"], n_jobs=1)}

    @classmethod
    def create_scaler(cls, name):
        return cls.scalers[name]()

    @classmethod
    def create_model(cls, name):
        return cls.models[name]()


def make_pipeline(model, scaler, added=True, k="all"):
    numeric = NUM + (["TotalSF", "HouseAge"] if added else [])
    categorical = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    processor = ColumnTransformer([("num", "passthrough", numeric), ("cat", categorical, CAT)])
    return Pipeline([("features", FeatureEngineer(added)), ("prepare", processor),
                     ("select", SelectKBest(f_regression, k=k)),
                     ("scale", Factory.create_scaler(scaler)), ("model", Factory.create_model(model))])


class Repository:
    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else ROOT / "artifacts/v1"

    def save(self, model, schema, metadata):
        self.folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.folder / "model.joblib")
        for name, data in [("feature_schema", schema), ("metadata", metadata)]:
            (self.folder / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False))

    def load(self):
        model = joblib.load(self.folder / "model.joblib")
        schema = json.loads((self.folder / "feature_schema.json").read_text())
        metadata = json.loads((self.folder / "metadata.json").read_text())
        return model, schema, metadata


class PredictionService:
    def __init__(self, repository):
        self.model, self.schema, self.metadata = repository.load()

    def predict_one(self, data):
        frame = pd.DataFrame([data])[self.schema["feature_order"]]
        value = float(self.model.predict(frame)[0])
        if not np.isfinite(value):
            raise ValueError("Некоректний прогноз")
        return round(value, 2)
