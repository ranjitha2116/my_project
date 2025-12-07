"""Train the irrigation regression model on the generated dataset (no growth stage)."""

from __future__ import annotations

import os
from typing import Dict, Any

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from .dataset_generator import generate_dataset


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def train_and_save(
    models_path: str | None = None, n_estimators: int = 150, random_state: int = 42
) -> Dict[str, Any]:
    df = generate_dataset()

    X = df[["crop", "soil_moisture", "temperature", "humidity"]]
    y = df["water_lpm2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("crop", OneHotEncoder(handle_unknown="ignore"), ["crop"]),
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # feature names after one-hot
    enc: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["crop"]
    crop_feature_names = enc.get_feature_names_out(["crop"])
    numeric_features = ["soil_moisture", "temperature", "humidity"]
    feature_names = list(crop_feature_names) + numeric_features

    rf: RandomForestRegressor = pipe.named_steps["model"]
    importances = rf.feature_importances_

    model_info = {
        "pipeline": pipe,
        "r2": float(r2),
        "feature_names": feature_names,
        "importances": importances,
    }

    if models_path is None:
        models_path = MODEL_PATH

    os.makedirs(os.path.dirname(models_path), exist_ok=True)
    joblib.dump(model_info, models_path)

    return model_info


def load_model(models_path: str | None = None) -> Dict[str, Any]:
    if models_path is None:
        models_path = MODEL_PATH
    if not os.path.exists(models_path):
        raise FileNotFoundError(
            f"Model not found at {models_path}. Run train_and_save() first."
        )
    return joblib.load(models_path)
