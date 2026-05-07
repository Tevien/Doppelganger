"""
SCORE2 / SCORE2-OP risk score.

Implements the published SCORE2 equations for adults < 70 years and the
SCORE2-OP equations for adults >= 70 years, with optional recalibration by
European risk region.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import logging
import numpy as np
import pandas as pd

from dpplgngr.scores.base import BaseScoreCalculator, ScoreRegistry

logger = logging.getLogger(__name__)


SCORE2_FEATURES = [
    "age",
    "sex",
    "smoking",
    "sbp",
    "diabetes",
    "total_chol",
    "hdl",
]

_SCALE_PARAMS = {
    "under_70": {
        "low": {"male": (-0.5699, 0.7476), "female": (-0.7380, 0.7019)},
        "moderate": {"male": (-0.1565, 0.8009), "female": (-0.3143, 0.7701)},
        "high": {"male": (0.3207, 0.9360), "female": (0.5710, 0.9369)},
        "very_high": {"male": (0.5836, 0.8294), "female": (0.9412, 0.8329)},
    },
    "over_70": {
        "low": {"male": (-0.34, 1.19), "female": (-0.52, 1.01)},
        "moderate": {"male": (0.01, 1.25), "female": (-0.10, 1.10)},
        "high": {"male": (0.08, 1.15), "female": (0.38, 1.09)},
        "very_high": {"male": (0.05, 0.70), "female": (0.38, 0.69)},
    },
}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _coerce_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        normalized = series.astype("string").str.strip().str.lower()
        mapped = normalized.replace(
            {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "y": 1,
                "n": 0,
            }
        )
        return pd.to_numeric(mapped, errors="coerce").astype("float64")

    return pd.to_numeric(series, errors="coerce").astype("float64")


def _normalize_sex(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.Series(
            np.where(numeric == 1, "female", np.where(numeric == 0, "male", pd.NA)),
            index=series.index,
            dtype="string",
        )

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.replace(
        {
            "f": "female",
            "female": "female",
            "woman": "female",
            "m": "male",
            "male": "male",
            "man": "male",
        }
    )
    valid = mapped.where(mapped.isin(["female", "male"]))
    return valid.astype("string")


def _normalize_region(value: Optional[str]) -> str:
    if value is None:
        return "moderate"

    normalized = str(value).strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    aliases = {
        "low": "low",
        "moderate": "moderate",
        "medium": "moderate",
        "high": "high",
        "very high": "very_high",
        "veryhigh": "very_high",
    }
    if normalized not in aliases:
        raise ValueError(
            "SCORE2 risk_region must be one of: Low, Moderate, High, Very high"
        )
    return aliases[normalized]


def _resolve_region(
    data: pd.DataFrame,
    column_mapping: Dict[str, str],
    default_region: str,
) -> pd.Series:
    region_spec = column_mapping.get("risk_region")
    if region_spec is None:
        return pd.Series(default_region, index=data.index, dtype="string")

    if region_spec in data.columns:
        return data[region_spec].astype("string")

    return pd.Series(region_spec, index=data.index, dtype="string")


def _recalibrate(base_risk: np.ndarray, scale1: float, scale2: float) -> np.ndarray:
    clipped = np.clip(base_risk, 1e-12, 1 - 1e-12)
    return 1 - np.exp(-np.exp(scale1 + scale2 * np.log(-np.log(1 - clipped))))


def _classify_score2(age: pd.Series, risk: pd.Series) -> pd.Series:
    conditions = [
        (age < 50) & (risk < 2.5),
        (age < 50) & (risk >= 2.5) & (risk < 7.5),
        (age < 50) & (risk >= 7.5),
        (age >= 50) & (age <= 69) & (risk < 5.0),
        (age >= 50) & (age <= 69) & (risk >= 5.0) & (risk < 10.0),
        (age >= 50) & (age <= 69) & (risk >= 10.0),
        (age >= 70) & (risk < 7.5),
        (age >= 70) & (risk >= 7.5) & (risk < 15.0),
        (age >= 70) & (risk >= 15.0),
    ]
    choices = [
        "Low risk",
        "Moderate risk",
        "High risk",
        "Low risk",
        "Moderate risk",
        "High risk",
        "Low risk",
        "Moderate risk",
        "High risk",
    ]
    return pd.Series(
        np.select(conditions, choices, default=None),
        index=age.index,
        dtype="object",
    )


@ScoreRegistry.auto
class SCORE2Score(BaseScoreCalculator):
    """SCORE2 / SCORE2-OP 10-year cardiovascular risk calculator."""

    name = "score2"

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        risk_region: str = "Moderate",
    ):
        super().__init__(column_mapping)
        self.risk_region = _normalize_region(risk_region)

    def required_features(self) -> List[str]:
        return list(SCORE2_FEATURES)

    def output_columns(self) -> List[str]:
        return ["score2", "score2_class"]

    def _calculate(
        self,
        data: pd.DataFrame,
        column_mapping: Dict[str, str],
    ) -> pd.DataFrame:
        age = _coerce_numeric(data[column_mapping["age"]])
        sex = _normalize_sex(data[column_mapping["sex"]])
        smoking = _coerce_binary(data[column_mapping["smoking"]])
        sbp = _coerce_numeric(data[column_mapping["sbp"]])
        diabetes = _coerce_binary(data[column_mapping["diabetes"]])
        total_chol = _coerce_numeric(data[column_mapping["total_chol"]])
        hdl = _coerce_numeric(data[column_mapping["hdl"]])
        region_raw = _resolve_region(data, column_mapping, self.risk_region)
        region = region_raw.map(lambda value: _normalize_region(value) if pd.notna(value) else pd.NA)

        risk = pd.Series(np.nan, index=data.index, dtype="float64")

        valid_common = ~(age.isna() | sex.isna() | smoking.isna() | sbp.isna() | diabetes.isna() | total_chol.isna() | hdl.isna() | region.isna())

        under_70 = age < 70
        over_70 = ~under_70

        for age_group_name, age_mask in [("under_70", under_70), ("over_70", over_70)]:
            for sex_name in ("male", "female"):
                base_mask = valid_common & age_mask & (sex == sex_name)
                if not base_mask.any():
                    continue

                if age_group_name == "under_70":
                    age_term = (age - 60) / 5
                    if sex_name == "male":
                        linear_predictor = (
                            0.3742 * age_term
                            + 0.6012 * smoking
                            + 0.2777 * ((sbp - 120) / 20)
                            + 0.6457 * diabetes
                            + 0.1458 * (total_chol - 6)
                            - 0.2698 * ((hdl - 1.3) / 0.5)
                            - 0.0755 * age_term * smoking
                            - 0.0255 * age_term * ((sbp - 120) / 20)
                            - 0.0281 * age_term * (total_chol - 6)
                            + 0.0426 * age_term * ((hdl - 1.3) / 0.5)
                            - 0.0983 * age_term * diabetes
                        )
                        base_risk = 1 - np.power(0.9605, np.exp(linear_predictor))
                    else:
                        linear_predictor = (
                            0.4648 * age_term
                            + 0.7744 * smoking
                            + 0.3131 * ((sbp - 120) / 20)
                            + 0.8096 * diabetes
                            + 0.1002 * (total_chol - 6)
                            - 0.2606 * ((hdl - 1.3) / 0.5)
                            - 0.1088 * age_term * smoking
                            - 0.0277 * age_term * ((sbp - 120) / 20)
                            - 0.0226 * age_term * (total_chol - 6)
                            + 0.0613 * age_term * ((hdl - 1.3) / 0.5)
                            - 0.1272 * age_term * diabetes
                        )
                        base_risk = 1 - np.power(0.9776, np.exp(linear_predictor))
                else:
                    age_term = age - 73
                    if sex_name == "male":
                        linear_predictor = (
                            0.0634 * age_term
                            + 0.4245 * diabetes
                            + 0.3524 * smoking
                            + 0.0094 * (sbp - 150)
                            + 0.0850 * (total_chol - 6)
                            - 0.3564 * (hdl - 1.4)
                            - 0.0174 * age_term * diabetes
                            - 0.0247 * age_term * smoking
                            - 0.0005 * age_term * (sbp - 150)
                            + 0.0073 * age_term * (total_chol - 6)
                            + 0.0091 * age_term * (hdl - 1.4)
                        )
                        base_risk = 1 - np.power(0.7576, np.exp(linear_predictor - 0.0929))
                    else:
                        linear_predictor = (
                            0.0789 * age_term
                            + 0.6010 * diabetes
                            + 0.4921 * smoking
                            + 0.0102 * (sbp - 150)
                            + 0.0605 * (total_chol - 6)
                            - 0.3040 * (hdl - 1.4)
                            - 0.0107 * age_term * diabetes
                            - 0.0255 * age_term * smoking
                            - 0.0004 * age_term * (sbp - 150)
                            - 0.0009 * age_term * (total_chol - 6)
                            + 0.0154 * age_term * (hdl - 1.4)
                        )
                        base_risk = 1 - np.power(0.8082, np.exp(linear_predictor - 0.2290))

                for region_name in ("low", "moderate", "high", "very_high"):
                    region_mask = base_mask & (region == region_name)
                    if not region_mask.any():
                        continue
                    scale1, scale2 = _SCALE_PARAMS[age_group_name][region_name][sex_name]
                    recalibrated = _recalibrate(base_risk[region_mask], scale1, scale2)
                    risk.loc[region_mask] = np.round(recalibrated * 100.0, 1)

        score_class = _classify_score2(age, risk)
        score_class.loc[risk.isna()] = np.nan

        return pd.DataFrame({"score2": risk, "score2_class": score_class}, index=data.index)