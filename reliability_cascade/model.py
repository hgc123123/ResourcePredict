from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from reliability_cascade.features import CascadeFeatures, post_features


@dataclass(frozen=True)
class CascadeConfig:
    seed: int = 42
    require_exact_script_for_bypass: bool = True
    reliability_error_cap: float = 0.10
    pre_n_estimators: int = 100
    pre_learning_rate: float = 0.03
    pre_max_depth: int = 2
    post_n_estimators: int = 120
    post_learning_rate: float = 0.03
    post_max_depth: int = 2
    post_blend_enabled: bool = True
    trust_region_enabled: bool = True
    trust_region_base: float = 0.025
    trust_region_mad_scale: float = 2.0
    trust_region_max: float = 0.10

    def validate(self) -> None:
        if self.reliability_error_cap <= 0.0 or not np.isfinite(
            self.reliability_error_cap
        ):
            raise ValueError("reliability_error_cap must be finite and positive")
        for name in ("pre_n_estimators", "post_n_estimators"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("pre_learning_rate", "post_learning_rate"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("pre_max_depth", "post_max_depth"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "trust_region_base",
            "trust_region_mad_scale",
            "trust_region_max",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.trust_region_base > self.trust_region_max:
            raise ValueError("trust_region_base cannot exceed trust_region_max")


@dataclass
class MedianImputer:
    medians: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "MedianImputer":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("Cannot fit an imputer on an empty/non-matrix input")
        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(finite, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        return cls(medians.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.medians):
            raise ValueError("Feature matrix differs from the fitted imputer schema")
        output = np.where(np.isfinite(matrix), matrix, self.medians)
        if not np.isfinite(output).all():
            raise ValueError("Imputed feature matrix contains non-finite values")
        return output


@dataclass
class ConstantProbabilityClassifier:
    probability: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        probability = np.full(len(values), float(self.probability), dtype=float)
        return np.column_stack([1.0 - probability, probability])


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), float(self.value), dtype=float)


@dataclass
class ResourceCascade:
    resource: str
    candidate_name: str
    threshold: float
    config: CascadeConfig
    pre_names: list[str]
    post_names: list[str]
    pre_imputer: MedianImputer
    post_imputer: MedianImputer
    pre_model: object
    post_model: object
    fit_rows: int
    fit_reliable_rate: float

    @classmethod
    def fit(
        cls,
        features: CascadeFeatures,
        *,
        resource: str,
        candidate_name: str,
        fit_mask: np.ndarray,
        config: CascadeConfig,
        threshold: float = 0.5,
    ) -> "ResourceCascade":
        config.validate()
        if resource not in {"memory", "vram"}:
            raise ValueError(f"Unknown resource: {resource}")
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if len(fit_mask) != len(features.frame):
            raise ValueError("fit_mask length mismatch")
        resource_mask = features.resource(resource)
        valid = features.frame["valid"].astype(bool).to_numpy()
        history = features.frame["user_available"].astype(bool).to_numpy()
        exact = features.frame["exact_available"].astype(bool).to_numpy()
        target = features.frame["target"].to_numpy(float)
        qwen = features.frame["qwen_prediction"].to_numpy(float)
        post_values, post_names, history_value = post_features(
            features, candidate_name
        )

        eligible = fit_mask & resource_mask & valid & history
        if config.require_exact_script_for_bypass:
            eligible &= exact
        if not eligible.any():
            raise ValueError(
                f"No eligible {resource} rows can fit the pre-inference reliability gate"
            )
        history_error = np.abs(history_value[eligible] - target[eligible])
        qwen_error = np.abs(qwen[eligible] - target[eligible])
        reliable = (history_error <= qwen_error + 1e-12) & (
            history_error <= config.reliability_error_cap
        )
        reliability_weight = 1.0 + np.minimum(
            np.abs(history_error - qwen_error)
            / max(config.reliability_error_cap, 1e-8),
            5.0,
        )
        pre_imputer = MedianImputer.fit(features.pre_values[eligible])
        pre_fit = pre_imputer.transform(features.pre_values[eligible])
        if len(np.unique(reliable)) == 1:
            pre_model: object = ConstantProbabilityClassifier(float(reliable[0]))
        else:
            pre_model = GradientBoostingClassifier(
                n_estimators=config.pre_n_estimators,
                learning_rate=config.pre_learning_rate,
                max_depth=config.pre_max_depth,
                subsample=1.0,
                random_state=config.seed,
            )
            pre_model.fit(pre_fit, reliable.astype(int), sample_weight=reliability_weight)

        post_fit_mask = fit_mask & resource_mask & valid & history
        if not post_fit_mask.any():
            raise ValueError(f"No {resource} history rows can fit the post-Qwen gate")
        qwen_fit = qwen[post_fit_mask]
        history_fit = history_value[post_fit_mask]
        target_fit = target[post_fit_mask]
        denominator = history_fit - qwen_fit
        stable = np.abs(denominator) > 1e-8
        gamma = np.full(len(denominator), 0.5, dtype=float)
        gamma[stable] = np.clip(
            (target_fit[stable] - qwen_fit[stable]) / denominator[stable],
            0.0,
            1.0,
        )
        qwen_fit_error = np.abs(qwen_fit - target_fit)
        history_fit_error = np.abs(history_fit - target_fit)
        post_weight = 1.0 + np.minimum(
            np.maximum(
                np.abs(qwen_fit_error - history_fit_error),
                np.abs(denominator),
            )
            / max(config.reliability_error_cap, 1e-8),
            5.0,
        )
        post_imputer = MedianImputer.fit(post_values[post_fit_mask])
        post_fit = post_imputer.transform(post_values[post_fit_mask])
        post_model: object
        if not config.post_blend_enabled:
            post_model = ConstantRegressor(0.0)
        elif np.allclose(gamma, gamma[0], rtol=0.0, atol=1e-12):
            post_model = ConstantRegressor(float(gamma[0]))
        else:
            post_model = GradientBoostingRegressor(
                loss="huber",
                n_estimators=config.post_n_estimators,
                learning_rate=config.post_learning_rate,
                max_depth=config.post_max_depth,
                subsample=1.0,
                random_state=config.seed + 1009,
            )
            post_model.fit(post_fit, gamma, sample_weight=post_weight)

        return cls(
            resource=resource,
            candidate_name=candidate_name,
            threshold=float(threshold),
            config=config,
            pre_names=list(features.pre_names),
            post_names=list(post_names),
            pre_imputer=pre_imputer,
            post_imputer=post_imputer,
            pre_model=pre_model,
            post_model=post_model,
            fit_rows=int(np.sum(post_fit_mask)),
            fit_reliable_rate=float(np.mean(reliable)),
        )

    def with_threshold(self, threshold: float) -> "ResourceCascade":
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("Route threshold must be finite and non-negative")
        return ResourceCascade(
            **{**self.__dict__, "threshold": float(threshold)}
        )

    def predict(self, features: CascadeFeatures) -> Dict[str, np.ndarray]:
        if features.pre_names != self.pre_names:
            raise ValueError("Pre-route feature schema differs from the fitted model")
        post_values, post_names, history_value = post_features(
            features, self.candidate_name
        )
        if post_names != self.post_names:
            raise ValueError("Post-route feature schema differs from the fitted model")
        frame = features.frame
        resource_mask = features.resource(self.resource)
        history = frame["user_available"].astype(bool).to_numpy()
        exact = frame["exact_available"].astype(bool).to_numpy()
        applicable = frame["resource_applicable"].astype(bool).to_numpy()
        qwen = frame["qwen_prediction"].to_numpy(float)

        probability = np.zeros(len(frame), dtype=float)
        eligible = resource_mask & applicable & history
        if self.config.require_exact_script_for_bypass:
            eligible &= exact
        if eligible.any():
            values = self.pre_imputer.transform(features.pre_values[eligible])
            probability[eligible] = np.asarray(
                self.pre_model.predict_proba(values), dtype=float
            )[:, 1]

        gamma = np.zeros(len(frame), dtype=float)
        post_rows = resource_mask & applicable & history
        if post_rows.any():
            values = self.post_imputer.transform(post_values[post_rows])
            gamma[post_rows] = np.clip(
                np.asarray(self.post_model.predict(values), dtype=float), 0.0, 1.0
            )
        safe_history = np.where(history, history_value, qwen)
        post_prediction = gamma * safe_history + (1.0 - gamma) * qwen

        trust_radius = np.full(len(frame), np.nan, dtype=float)
        trust_applied = np.zeros(len(frame), dtype=bool)
        if self.config.post_blend_enabled and self.config.trust_region_enabled:
            exact_mad = pd_to_float(frame["exact_mad5"], default=0.0)
            radius = np.minimum(
                self.config.trust_region_max,
                self.config.trust_region_base
                + self.config.trust_region_mad_scale * np.maximum(exact_mad, 0.0),
            )
            trust_applied = resource_mask & applicable & history & exact
            trust_radius[trust_applied] = radius[trust_applied]
            post_prediction[trust_applied] = np.clip(
                post_prediction[trust_applied],
                safe_history[trust_applied] - radius[trust_applied],
                safe_history[trust_applied] + radius[trust_applied],
            )

        bypass = eligible & (probability >= self.threshold)
        prediction = np.where(bypass, safe_history, post_prediction)
        prediction = np.clip(prediction, 0.0, 1.0)
        qwen_required_resource = resource_mask & applicable & ~bypass
        return {
            "resource_mask": resource_mask,
            "prediction": prediction,
            "history_prediction": history_value,
            "reliability_probability": probability,
            "history_bypass": bypass,
            "qwen_required_resource": qwen_required_resource,
            "post_gamma": gamma,
            "post_prediction": np.clip(post_prediction, 0.0, 1.0),
            "trust_region_applied": trust_applied,
            "trust_region_radius": trust_radius,
        }

    def feature_importance(self) -> list[Dict[str, object]]:
        rows: list[Dict[str, object]] = []
        for stage, model, names in (
            ("pre_route", self.pre_model, self.pre_names),
            ("post_blend", self.post_model, self.post_names),
        ):
            values = getattr(model, "feature_importances_", None)
            if values is None:
                values = np.zeros(len(names), dtype=float)
            for name, value in zip(names, values):
                rows.append(
                    {
                        "resource": self.resource,
                        "history_expert": self.candidate_name,
                        "stage": stage,
                        "feature": name,
                        "importance": float(value),
                    }
                )
        return rows

    def policy(self) -> Dict[str, object]:
        return {
            "resource": self.resource,
            "history_expert": self.candidate_name,
            "reliability_threshold": float(self.threshold),
            "fit_rows": int(self.fit_rows),
            "fit_reliable_rate": float(self.fit_reliable_rate),
            "configuration": asdict(self.config),
            "pre_route_features": list(self.pre_names),
            "post_route_features": list(self.post_names),
            "route_decision_uses_qwen_output": False,
        }


def pd_to_float(values: object, *, default: float) -> np.ndarray:
    # Import locally to keep the sklearn model module's public surface compact.
    import pandas as pd

    return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)


__all__ = ["CascadeConfig", "ResourceCascade"]
