"""Leakage-aware direct multi-horizon baselines for hourly realized rider fares.

At origin t the newest accessible pickup-hour bin is t - embargo - 1 hour.
All forecasts use the same bounded history. Outcome values in this retrospective
panel are used for fitting/scoring only, never as future model features.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 168)
WINDOWS = (6, 24, 168)
NAIVE_NAMES = ("Persistence", "Daily seasonal", "Weekly seasonal", "Four-week seasonal")
HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class FeatureConfig:
    embargo_hours: int = 3
    context_hours: int = 672


def _check_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"series_id", "timestamp", "fare", "count"}
    if not required.issubset(panel):
        raise ValueError(f"Missing panel columns: {required - set(panel.columns)}")
    result = panel[list(required)].copy()
    result["series_id"] = result["series_id"].astype(str)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    if result.duplicated(["series_id", "timestamp"]).any():
        raise ValueError("Duplicate series-hour rows")
    if not result.timestamp.eq(result.timestamp.dt.floor("h")).all():
        raise ValueError("Timestamps must identify whole hourly bins")
    if (result.fare.dropna() < 0).any():
        raise ValueError("Fares must be nonnegative or missing")
    return result.sort_values(["series_id", "timestamp"]).reset_index(drop=True)


class CausalFeatures:
    """Precompute trailing transforms, then retrieve them at a causal cutoff."""

    def __init__(self, panel: pd.DataFrame, config: FeatureConfig = FeatureConfig()):
        if config.embargo_hours < 0 or config.context_hours < max(LAGS) + 1:
            raise ValueError("Invalid embargo/context")
        self.config = config
        self.panel = _check_panel(panel)
        self.series_ids = sorted(self.panel.series_id.unique().tolist())
        self.start = self.panel.timestamp.min()
        self.end = self.panel.timestamp.max() + HOUR
        self.frames: dict[str, pd.DataFrame] = {}
        self.transforms: dict[str, pd.DataFrame] = {}
        grid = pd.date_range(self.start, self.end - HOUR, freq="h")
        for sid, group in self.panel.groupby("series_id", sort=True):
            frame = group.set_index("timestamp")[["fare", "count"]].reindex(grid)
            self.frames[sid] = frame
            values = frame.fare
            transform = pd.DataFrame(index=grid)
            for lag in LAGS:
                transform[f"fare_lag_{lag}"] = values.shift(lag)
                transform[f"count_lag_{lag}"] = frame["count"].shift(lag)
            for window in WINDOWS:
                roll = values.rolling(window, min_periods=1)
                transform[f"fare_mean_{window}"] = roll.mean()
                transform[f"fare_std_{window}"] = roll.std(ddof=0)
                transform[f"fare_min_{window}"] = roll.min()
                transform[f"fare_max_{window}"] = roll.max()
                transform[f"fare_observations_{window}"] = roll.count()
                transform[f"count_mean_{window}"] = frame["count"].rolling(window, min_periods=1).mean()
            # Bounded forward fill is for a PREDICTOR, not fabrication of labels.
            transform["last_valid_fare"] = values.ffill(limit=config.context_hours - 1)
            self.transforms[sid] = transform

    def cases(self, origins: Iterable, horizons=(1, 3, 6)) -> pd.DataFrame:
        origins = pd.DatetimeIndex(origins)
        if len(origins) == 0:
            raise ValueError("At least one origin is required")
        if not origins.equals(origins.floor("h")):
            raise ValueError("Origins must be on the hourly grid")
        if min(horizons) < 1:
            raise ValueError("Forecast horizons must be positive")
        pieces = []
        for sid in self.series_ids:
            latest = origins - pd.Timedelta(hours=self.config.embargo_hours + 1)
            historical = self.transforms[sid].reindex(latest).reset_index(drop=True)
            for horizon in horizons:
                target = origins + pd.Timedelta(hours=horizon - 1)
                item = historical.copy()
                item["series_id"] = sid
                item["origin"] = origins
                item["horizon"] = int(horizon)
                item["target_timestamp"] = target
                item["last_available_timestamp"] = latest
                # Stored timestamps are UTC-naive; demand seasonality is NYC-local.
                local_target = target.tz_localize("UTC").tz_convert("America/New_York")
                item["hour_sin"] = np.sin(2 * np.pi * local_target.hour / 24)
                item["hour_cos"] = np.cos(2 * np.pi * local_target.hour / 24)
                item["weekday_sin"] = np.sin(2 * np.pi * local_target.dayofweek / 7)
                item["weekday_cos"] = np.cos(2 * np.pi * local_target.dayofweek / 7)
                item["hour"] = local_target.hour.astype(float)
                item["weekday"] = local_target.dayofweek.astype(float)
                item["weekend"] = (local_target.dayofweek >= 5).astype(float)
                item["elapsed_day"] = (target - self.start).total_seconds() / 86400.0
                # Target-aligned seasonality matters more than latest-bin seasonality.
                for lag in (24, 48, 168, 336, 504, 672):
                    reference = target - pd.Timedelta(hours=lag)
                    oldest = latest - pd.Timedelta(hours=self.config.context_hours - 1)
                    permitted = (reference <= latest) & (reference >= oldest)
                    retrieved = self.frames[sid].fare.reindex(reference).to_numpy()
                    item[f"target_lag_{lag}"] = np.where(permitted, retrieved, np.nan)
                # Labels are carried alongside features only; model_columns excludes them.
                item["y_true"] = self.frames[sid].fare.reindex(target).to_numpy()
                item["target_count"] = self.frames[sid]["count"].reindex(target).to_numpy()
                pieces.append(item)
        return pd.concat(pieces, ignore_index=True)


EXCLUDED_COLUMNS = {
    "origin", "target_timestamp", "last_available_timestamp", "y_true", "target_count"
}


def causal_feature_row(panel: pd.DataFrame, series_id: str, origin, horizon: int,
                       config: FeatureConfig = FeatureConfig()) -> dict:
    """Convenient audit helper. Returned mapping deliberately excludes labels."""
    engine = CausalFeatures(panel[panel.series_id.astype(str).eq(str(series_id))], config)
    row = engine.cases([pd.Timestamp(origin)], [horizon]).iloc[0]
    return row.drop(labels=list(EXCLUDED_COLUMNS)).to_dict()


def naive_predictions(cases: pd.DataFrame) -> dict[str, np.ndarray]:
    fallback = cases.last_valid_fare
    weekly = cases.target_lag_168.fillna(cases.target_lag_24).fillna(fallback)
    daily = cases.target_lag_24.fillna(weekly).fillna(fallback)
    recent = cases[[f"target_lag_{x}" for x in (168, 336, 504, 672)]].mean(axis=1)
    return {
        "Persistence": fallback.to_numpy(),
        "Daily seasonal": daily.to_numpy(),
        "Weekly seasonal": weekly.to_numpy(),
        "Four-week seasonal": recent.fillna(weekly).fillna(fallback).to_numpy(),
    }


def _records(cases: pd.DataFrame, model: str, prediction: np.ndarray) -> pd.DataFrame:
    output = cases[["origin", "series_id", "horizon", "target_timestamp",
                    "last_available_timestamp", "y_true", "target_count"]].copy()
    output["model"] = model
    output["prediction"] = np.maximum(0, np.asarray(prediction, dtype=float))
    return output


class BaselineSuite:
    """Train once, tune on chronological validation, freeze before final test.

    Fit is available at ``train_end``: training outcome bins must finish before
    train_end - embargo. Validation labels are not used in coefficient fitting.
    Fit returns chosen-model validation predictions plus JSON-safe metadata.
    With refit=True, chosen models are then refitted using labels available at
    val_end, ready for a strictly later final test. Stored validation forecasts
    always come from the original training-only fits.
    """

    def __init__(self, panel: pd.DataFrame, horizons=(1, 3, 6), embargo_hours=3,
                 context_hours=672):
        self.features = CausalFeatures(panel, FeatureConfig(embargo_hours, context_hours))
        self.horizons = tuple(horizons)
        self.models: dict[str, Pipeline] = {}
        self.metadata: dict = {}

    def _pipeline(self, cases: pd.DataFrame, regressor, scale: bool) -> Pipeline:
        numeric = [c for c in self.model_columns if c != "series_id"]
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True,
                                          keep_empty_features=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        preprocess = ColumnTransformer([
            ("numeric", Pipeline(steps), numeric),
            ("route", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["series_id"]),
        ])
        return Pipeline([("features", preprocess), ("regressor", regressor)])

    def fit(self, train_end, val_start, val_end, origin_stride=3,
            min_training_history_hours=168, quick=False, refit=True,
            training_origin_stride=1):
        train_end, val_start, val_end = map(pd.Timestamp, (train_end, val_start, val_end))
        if not self.features.start < train_end <= val_start < val_end:
            raise ValueError("Require data_start < train_end <= val_start < val_end")
        if origin_stride < 1 or training_origin_stride < 1:
            raise ValueError("Training and validation origin strides must be >= 1")
        begin = self.features.start + pd.Timedelta(hours=min_training_history_hours)
        train_origins = pd.date_range(begin, train_end - HOUR, freq=f"{training_origin_stride}h")
        val_origins = pd.date_range(val_start, val_end - HOUR, freq=f"{origin_stride}h")
        training = self.features.cases(train_origins, self.horizons)
        label_cutoff = train_end - pd.Timedelta(hours=self.features.config.embargo_hours)
        training = training.loc[training.target_timestamp.lt(label_cutoff) & training.y_true.notna()].copy()
        validation = self.features.cases(val_origins, self.horizons)
        validation_label_cutoff = val_end - pd.Timedelta(hours=self.features.config.embargo_hours)
        validation = validation.loc[validation.target_timestamp.lt(validation_label_cutoff)].copy()
        if len(training) < 100 or validation.y_true.notna().sum() < 10:
            raise ValueError("Insufficient observed training or validation labels")
        self.model_columns = [c for c in training.columns if c not in EXCLUDED_COLUMNS]
        x_train, y_train = training[self.model_columns], np.log1p(training.y_true)
        x_val = validation[self.model_columns]
        val_valid = validation.y_true.notna().to_numpy()
        truth = validation.loc[val_valid, "y_true"].to_numpy()
        configs = [
            ("Ridge", {"alpha": alpha}, Ridge(alpha=alpha), True)
            for alpha in ((10.0,) if quick else (1.0, 10.0, 100.0))
        ]
        for leaves in ((15,) if quick else (15, 31)):
            for loss in (("squared_error",) if quick else ("squared_error", "absolute_error")):
                settings = {"max_leaf_nodes": leaves, "max_iter": 60 if quick else 220,
                            "learning_rate": .06, "l2_regularization": 1., "min_samples_leaf": 25,
                            "loss": loss, "early_stopping": False, "random_state": 42}
                configs.append(("Gradient boosting", settings,
                                HistGradientBoostingRegressor(**settings), False))
        candidates, winners, timings = [], {}, {}
        for name, settings, regressor, scale in configs:
            start = time.perf_counter()
            model = self._pipeline(training, regressor, scale)
            with threadpool_limits(limits=4):
                model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            with threadpool_limits(limits=4):
                pred = np.maximum(0, np.expm1(model.predict(x_val)))
            pred_seconds = time.perf_counter() - start
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite predictions from {name}")
            mae = float(np.mean(np.abs(pred[val_valid] - truth)))
            candidates.append({"model": name, "parameters": settings, "validation_mae_usd": mae,
                               "fit_seconds": fit_seconds, "validation_predict_seconds": pred_seconds})
            if name not in winners or mae < winners[name]["mae"]:
                winners[name] = {"mae": mae, "settings": settings, "prediction": pred,
                                 "model": model, "fit_seconds": fit_seconds}
        records = [_records(validation, name, pred) for name, pred in naive_predictions(validation).items()]
        for name, winner in winners.items():
            self.models[name] = winner["model"]
            records.append(_records(validation, name, winner["prediction"]))
        refit_metadata = {}
        if refit:
            refit_origins = pd.date_range(begin, val_end - HOUR, freq=f"{training_origin_stride}h")
            refit_cases = self.features.cases(refit_origins, self.horizons)
            refit_cutoff = val_end - pd.Timedelta(hours=self.features.config.embargo_hours)
            refit_cases = refit_cases.loc[refit_cases.target_timestamp.lt(refit_cutoff) & refit_cases.y_true.notna()]
            for name, model in self.models.items():
                start = time.perf_counter()
                with threadpool_limits(limits=4):
                    model.fit(refit_cases[self.model_columns], np.log1p(refit_cases.y_true))
                refit_metadata[name] = {"fit_seconds": time.perf_counter() - start,
                                        "rows": len(refit_cases),
                                        "label_cutoff_exclusive": refit_cutoff.isoformat()}
        self.metadata = {
            "train_end": train_end.isoformat(), "training_label_cutoff_exclusive": label_cutoff.isoformat(),
            "validation_start": val_start.isoformat(), "validation_end_exclusive": val_end.isoformat(),
            "validation_label_cutoff_exclusive": validation_label_cutoff.isoformat(),
            "training_rows": len(training), "validation_cases": len(validation),
            "validation_observed_labels": int(val_valid.sum()),
            "training_origin_stride_hours": training_origin_stride,
            "refit_origin_stride_hours": training_origin_stride,
            "validation_origin_stride_hours": origin_stride,
            "embargo_hours": self.features.config.embargo_hours,
            "context_hours": self.features.config.context_hours,
            "model_features": self.model_columns,
            "target_transform": "log1p; point forecasts back-transformed with expm1",
            "selection_metric": "pooled validation MAE in USD, equal route-origin-horizon weight",
            "candidates": candidates,
            "selected": {name: {"parameters": winner["settings"], "validation_mae_usd": winner["mae"],
                                 "fit_seconds": winner["fit_seconds"]} for name, winner in winners.items()},
            "refit_after_validation": refit,
            "refit": refit_metadata,
            "timestamp_convention": "UTC-naive; calendar features converted to America/New_York",
            "seasonal_lag_convention": "Fixed elapsed hours (24/168); NYC calendar covariates separately handle DST",
            "fallbacks": "Seasonal missing references fall back to other seasonality, then last valid value within context.",
        }
        self.validation_predictions_ = pd.concat(records, ignore_index=True)
        return self.validation_predictions_, self.metadata

    def predict(self, origins: Iterable) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Call fit before predict")
        cases = self.features.cases(origins, self.horizons)
        records = [_records(cases, name, pred) for name, pred in naive_predictions(cases).items()]
        self.last_predict_timings_ = {}
        for name, model in self.models.items():
            start = time.perf_counter()
            with threadpool_limits(limits=4):
                pred = np.maximum(0, np.expm1(model.predict(cases[self.model_columns])))
            self.last_predict_timings_[name] = time.perf_counter() - start
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite predictions from {name}")
            records.append(_records(cases, name, pred))
        return pd.concat(records, ignore_index=True)
