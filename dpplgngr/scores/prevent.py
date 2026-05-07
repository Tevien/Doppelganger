"""
PREVENT cardiovascular risk equations.

Implements the American Heart Association PREVENT equations using coefficient
tables embedded in ``prevent_coefficients.json``. By default the calculator
selects the richest supported model variant available from the mapped inputs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import json
import logging
import numpy as np
import pandas as pd

from dpplgngr.scores.base import BaseScoreCalculator, ScoreRegistry

logger = logging.getLogger(__name__)


PREVENT_BASE_FEATURES = [
    "age",
    "sex",
    "sbp",
    "bp_tx",
    "total_chol",
    "hdl",
    "statin",
    "diabetes",
    "smoking",
    "egfr",
    "bmi",
]

PREVENT_OPTIONAL_FEATURES = ["hba1c", "uacr", "sdi"]
PREVENT_OUTCOMES = ["total_cvd", "ascvd", "heart_failure", "chd", "stroke"]
PREVENT_MODELS = ["base", "hba1c", "uacr", "sdi", "full"]
PREVENT_TIMES = ["10yr", "30yr"]


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
        values = np.where(numeric == 1, "female", np.where(numeric == 0, "male", pd.NA))
        return pd.Series(values, index=series.index, dtype="string")

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
    return mapped.where(mapped.isin(["female", "male"])).astype("string")


def _normalize_model(value: Optional[str]) -> str:
    if value is None:
        return "auto"

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "auto":
        return normalized
    if normalized not in PREVENT_MODELS:
        raise ValueError(
            f"PREVENT model must be one of {PREVENT_MODELS} or 'auto', got {value!r}."
        )
    return normalized


def _normalize_chol_unit(value: Optional[str]) -> str:
    if value is None:
        return "mmol/L"

    normalized = str(value).strip().lower()
    if normalized in {"mmol/l", "mmol", "mmol l-1"}:
        return "mmol/L"
    if normalized in {"mg/dl", "mg", "mg dl-1"}:
        return "mg/dL"
    raise ValueError("cholesterol_unit must be either 'mmol/L' or 'mg/dL'.")


def _resolve_optional_series(
    data: pd.DataFrame,
    column_mapping: Dict[str, str],
    key: str,
) -> Optional[pd.Series]:
    mapped = column_mapping.get(key)
    if mapped is None:
        return None
    if mapped in data.columns:
        return data[mapped]
    try:
        scalar = float(mapped)
    except (TypeError, ValueError):
        return None
    return pd.Series(scalar, index=data.index, dtype="float64")


def _resolve_literal_or_default(
    data: pd.DataFrame,
    column_mapping: Dict[str, str],
    key: str,
    default: str,
) -> str:
    mapped = column_mapping.get(key)
    if mapped is None:
        return default
    if mapped in data.columns:
        values = data[mapped].dropna().astype(str)
        if values.empty:
            return default
        return values.iloc[0]
    return str(mapped)


def _chol_to_mmol(values: pd.Series, unit: str) -> pd.Series:
    if unit == "mg/dL":
        return values * 0.02586
    return values


@lru_cache(maxsize=1)
def _load_prevent_coefficients() -> Dict[str, Dict[str, Dict[str, float]]]:
    path = Path(__file__).with_name("prevent_coefficients.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _choose_model(column_mapping: Dict[str, str], requested_model: str) -> str:
    if requested_model != "auto":
        return requested_model

    has_hba1c = "hba1c" in column_mapping
    has_uacr = "uacr" in column_mapping
    has_sdi = "sdi" in column_mapping

    if has_hba1c and has_uacr and has_sdi:
        return "full"
    if has_hba1c:
        return "hba1c"
    if has_uacr:
        return "uacr"
    if has_sdi:
        return "sdi"
    return "base"


def _prepare_terms(
    data: pd.DataFrame,
    column_mapping: Dict[str, str],
    cholesterol_unit: str,
) -> pd.DataFrame:
    age = _coerce_numeric(data[column_mapping["age"]])
    sbp = _coerce_numeric(data[column_mapping["sbp"]])
    total_chol = _coerce_numeric(data[column_mapping["total_chol"]])
    hdl = _coerce_numeric(data[column_mapping["hdl"]])
    bmi = _coerce_numeric(data[column_mapping["bmi"]])
    egfr = _coerce_numeric(data[column_mapping["egfr"]])
    dm = _coerce_binary(data[column_mapping["diabetes"]])
    smoking = _coerce_binary(data[column_mapping["smoking"]])
    bp_tx = _coerce_binary(data[column_mapping["bp_tx"]])
    statin = _coerce_binary(data[column_mapping["statin"]])

    hba1c_series = _resolve_optional_series(data, column_mapping, "hba1c")
    uacr_series = _resolve_optional_series(data, column_mapping, "uacr")
    sdi_series = _resolve_optional_series(data, column_mapping, "sdi")

    non_hdl_c = _chol_to_mmol(total_chol - hdl, cholesterol_unit) - 3.5
    hdl_c_term = (_chol_to_mmol(hdl, cholesterol_unit) - 1.3) / 0.3

    terms = pd.DataFrame(index=data.index)
    terms["age"] = (age - 55.0) / 10.0
    terms["age_squared"] = terms["age"] ** 2
    terms["non_hdl_c"] = non_hdl_c
    terms["hdl_c"] = hdl_c_term
    terms["sbp_lt_110"] = (np.minimum(sbp, 110.0) - 110.0) / 20.0
    terms["sbp_gte_110"] = (np.maximum(sbp, 110.0) - 130.0) / 20.0
    terms["dm"] = dm
    terms["smoking"] = smoking
    terms["bmi_lt_30"] = (np.minimum(bmi, 30.0) - 25.0) / 5.0
    terms["bmi_gte_30"] = (np.maximum(bmi, 30.0) - 30.0) / 5.0
    terms["egfr_lt_60"] = (np.minimum(egfr, 60.0) - 60.0) / -15.0
    terms["egfr_gte_60"] = (np.maximum(egfr, 60.0) - 90.0) / -15.0
    terms["bp_tx"] = bp_tx
    terms["statin"] = statin
    terms["bp_tx_sbp_gte_110"] = bp_tx * terms["sbp_gte_110"]
    terms["statin_non_hdl_c"] = statin * terms["non_hdl_c"]
    terms["age_non_hdl_c"] = terms["age"] * terms["non_hdl_c"]
    terms["age_hdl_c"] = terms["age"] * terms["hdl_c"]
    terms["age_sbp_gte_110"] = terms["age"] * terms["sbp_gte_110"]
    terms["age_dm"] = terms["age"] * dm
    terms["age_smoking"] = terms["age"] * smoking
    terms["age_bmi_gte_30"] = terms["age"] * terms["bmi_gte_30"]
    terms["age_egfr_lt_60"] = terms["age"] * terms["egfr_lt_60"]

    if sdi_series is None:
        sdi = pd.Series(np.nan, index=data.index, dtype="float64")
    else:
        sdi = _coerce_numeric(sdi_series)
    terms["sdi_4_to_6"] = sdi.between(4, 6, inclusive="both").astype("float64")
    terms["sdi_7_to_10"] = sdi.between(7, 10, inclusive="both").astype("float64")
    terms["missing_sdi"] = sdi.isna().astype("float64")

    if uacr_series is None:
        uacr = pd.Series(np.nan, index=data.index, dtype="float64")
    else:
        uacr = _coerce_numeric(uacr_series)
        uacr = uacr.where(uacr > 0)
    terms["ln_uacr"] = np.where(uacr.notna(), np.log(uacr), 0.0)
    terms["missing_uacr"] = uacr.isna().astype("float64")

    if hba1c_series is None:
        hba1c = pd.Series(np.nan, index=data.index, dtype="float64")
    else:
        hba1c = _coerce_numeric(hba1c_series)
    terms["hba1c_dm"] = np.where(hba1c.notna() & (dm == 1), hba1c - 5.3, 0.0)
    terms["hba1c_no_dm"] = np.where(hba1c.notna() & (dm == 0), hba1c - 5.3, 0.0)
    terms["missing_hba1c"] = hba1c.isna().astype("float64")
    terms["constant"] = 1.0
    return terms


def _score_table(
    terms: pd.DataFrame,
    sex: pd.Series,
    table_name: str,
) -> pd.DataFrame:
    coeffs = _load_prevent_coefficients()[table_name]
    term_order = list(coeffs.keys())
    matrix = terms[term_order].to_numpy(dtype=np.float64)
    result = pd.DataFrame(index=terms.index)

    female_mask = (sex == "female").to_numpy()
    male_mask = (sex == "male").to_numpy()

    for outcome in PREVENT_OUTCOMES:
        female_coef = np.array([coeffs[term][f"female_{outcome}"] for term in term_order])
        male_coef = np.array([coeffs[term][f"male_{outcome}"] for term in term_order])

        female_log_odds = matrix @ female_coef
        male_log_odds = matrix @ male_coef
        log_odds = np.full(len(terms), np.nan, dtype=np.float64)
        log_odds[female_mask] = female_log_odds[female_mask]
        log_odds[male_mask] = male_log_odds[male_mask]
        result[outcome] = 1.0 / (1.0 + np.exp(-log_odds))

    return result


@ScoreRegistry.auto
class PREVENTScore(BaseScoreCalculator):
    """PREVENT risk calculator returning 10-year and 30-year event risks."""

    name = "prevent"

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        model: str = "auto",
        cholesterol_unit: str = "mmol/L",
    ):
        super().__init__(column_mapping)
        self.model = _normalize_model(model)
        self.cholesterol_unit = _normalize_chol_unit(cholesterol_unit)

    def required_features(self) -> List[str]:
        return list(PREVENT_BASE_FEATURES)

    def output_columns(self) -> List[str]:
        columns = [
            f"prevent_{time}_{outcome}"
            for time in PREVENT_TIMES
            for outcome in PREVENT_OUTCOMES
        ]
        return columns + ["prevent_model"]

    def _calculate(
        self,
        data: pd.DataFrame,
        column_mapping: Dict[str, str],
    ) -> pd.DataFrame:
        sex = _normalize_sex(data[column_mapping["sex"]])
        terms = _prepare_terms(
            data,
            column_mapping,
            cholesterol_unit=_normalize_chol_unit(
                _resolve_literal_or_default(
                    data,
                    column_mapping,
                    "cholesterol_unit",
                    self.cholesterol_unit,
                )
            ),
        )

        selected_model = _choose_model(column_mapping, self.model)
        results = pd.DataFrame(index=data.index)
        for time in PREVENT_TIMES:
            risks = _score_table(terms, sex, f"{selected_model}_{time}")
            for outcome in PREVENT_OUTCOMES:
                results[f"prevent_{time}_{outcome}"] = risks[outcome]

        base_missing = (
            terms[[
                "age",
                "non_hdl_c",
                "hdl_c",
                "sbp_lt_110",
                "sbp_gte_110",
                "dm",
                "smoking",
                "bmi_lt_30",
                "bmi_gte_30",
                "egfr_lt_60",
                "egfr_gte_60",
                "bp_tx",
                "statin",
            ]].isna().any(axis=1)
            | sex.isna()
        )
        results.loc[base_missing, [c for c in results.columns]] = np.nan
        results["prevent_model"] = selected_model
        results.loc[base_missing, "prevent_model"] = np.nan
        return results