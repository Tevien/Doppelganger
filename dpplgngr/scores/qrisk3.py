"""
QRISK3 10-year cardiovascular disease risk score.

This is a Python translation of the public QRISK3-2017 formula available via
the CRAN QRISK3 package mirror.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dpplgngr.scores.base import BaseScoreCalculator, ScoreRegistry


QRISK3_FEATURES = [
    "sex",
    "age",
    "atrial_fibrillation",
    "atypical_antipsychotic",
    "regular_steroid_tablets",
    "erectile_dysfunction",
    "migraine",
    "rheumatoid_arthritis",
    "chronic_kidney_disease",
    "severe_mental_illness",
    "systemic_lupus_erythematosus",
    "blood_pressure_treatment",
    "diabetes_type1",
    "diabetes_type2",
    "weight",
    "height",
    "ethnicity",
    "heart_attack_relative",
    "cholesterol_hdl_ratio",
    "systolic_blood_pressure",
    "std_systolic_blood_pressure",
    "smoke",
    "townsend",
]

_FEMALE_ETH_RISK = np.array([
    0.0,
    0.28040314332995425,
    0.562989941420754,
    0.29590000851116516,
    0.07278537987798254,
    -0.17072135508857317,
    -0.3937104331487497,
    -0.3263249528353027,
    -0.17127056883241784,
])
_FEMALE_SMOKE = np.array([
    0.0,
    0.13386833786546262,
    0.5620085801243854,
    0.6674959337750255,
    0.8494817764483085,
])
_MALE_ETH_RISK = np.array([
    0.0,
    0.2771924876030828,
    0.4744636071493127,
    0.5296172991968937,
    0.03510015918629902,
    -0.3580789966932792,
    -0.4005648523216514,
    -0.41522792889830173,
    -0.26321348134749967,
])
_MALE_SMOKE = np.array([
    0.0,
    0.19128222863388983,
    0.5524158819264555,
    0.6383505302750607,
    0.7898381988185802,
])


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


def _lookup(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = (values >= 1) & (values <= len(table))
    result[valid] = table[values[valid].astype(int) - 1]
    return result


def _smoke_indicator(smoke: np.ndarray, category: int) -> np.ndarray:
    return (smoke == category).astype(np.float64)


def _female_qrisk3(frame: pd.DataFrame) -> pd.Series:
    age = frame["age"].to_numpy()
    bmi = frame["bmi"].to_numpy()
    rati = frame["cholesterol_hdl_ratio"].to_numpy()
    sbp = frame["systolic_blood_pressure"].to_numpy()
    sbps5 = frame["std_systolic_blood_pressure"].to_numpy()
    town = frame["townsend"].to_numpy()
    smoke = frame["smoke"].to_numpy().astype(int)
    ethnicity = frame["ethnicity"].to_numpy().astype(int)

    age_1 = np.power(age / 10.0, -2) - 0.053274843841791
    age_2 = age / 10.0 - 4.332503318786621
    bmi_1 = np.power(bmi / 10.0, -2) - 0.154946178197861
    bmi_2 = np.power(bmi / 10.0, -2) * np.log(bmi / 10.0) - 0.144462317228317
    rati = rati - 3.47632646560669
    sbp = sbp - 123.13001251220703
    sbps5 = sbps5 - 9.002537727355957
    town = town - 0.392308831214905

    a = np.zeros(len(frame), dtype=np.float64)
    a += _lookup(ethnicity, _FEMALE_ETH_RISK)
    a += _lookup(smoke, _FEMALE_SMOKE)
    a += age_1 * -8.1388109247726188
    a += age_2 * 0.7973337668969910
    a += bmi_1 * 0.2923609227546005
    a += bmi_2 * -4.1513300213837665
    a += rati * 0.15338035820802554
    a += sbp * 0.013131488407103424
    a += sbps5 * 0.0078894541014586095
    a += town * 0.07722379058859011

    for key, coef in [
        ("atrial_fibrillation", 1.5923354969269663),
        ("atypical_antipsychotic", 0.25237642070115557),
        ("regular_steroid_tablets", 0.5952072530460185),
        ("migraine", 0.3012672608703450),
        ("rheumatoid_arthritis", 0.21364803435181942),
        ("chronic_kidney_disease", 0.6519456949384583),
        ("severe_mental_illness", 0.12555308058820178),
        ("systemic_lupus_erythematosus", 0.7588093865426769),
        ("blood_pressure_treatment", 0.50931593683423),
        ("diabetes_type1", 1.7267977510537347),
        ("diabetes_type2", 1.0688773244615468),
        ("heart_attack_relative", 0.45445319020896213),
    ]:
        a += frame[key].to_numpy() * coef

    a += age_1 * _smoke_indicator(smoke, 2) * -4.705716178585189
    a += age_1 * _smoke_indicator(smoke, 3) * -2.7430383403573337
    a += age_1 * _smoke_indicator(smoke, 4) * -0.8660808882939218
    a += age_1 * _smoke_indicator(smoke, 5) * 0.9024156236971065
    a += age_1 * frame["atrial_fibrillation"].to_numpy() * 19.93803488954656
    a += age_1 * frame["regular_steroid_tablets"].to_numpy() * -0.9840804523593628
    a += age_1 * frame["migraine"].to_numpy() * 1.7634979587873
    a += age_1 * frame["chronic_kidney_disease"].to_numpy() * -3.5874047731694114
    a += age_1 * frame["systemic_lupus_erythematosus"].to_numpy() * 19.690303738638292
    a += age_1 * frame["blood_pressure_treatment"].to_numpy() * 11.872809733921812
    a += age_1 * frame["diabetes_type1"].to_numpy() * -1.2444332714320747
    a += age_1 * frame["diabetes_type2"].to_numpy() * 6.86523420000096
    a += age_1 * bmi_1 * 23.802623412141742
    a += age_1 * bmi_2 * -71.18494769208701
    a += age_1 * frame["heart_attack_relative"].to_numpy() * 0.9946780794043513
    a += age_1 * sbp * 0.034131842338615485
    a += age_1 * town * -1.0301180802035639
    a += age_2 * _smoke_indicator(smoke, 2) * -0.07558924464319303
    a += age_2 * _smoke_indicator(smoke, 3) * -0.11951192874867074
    a += age_2 * _smoke_indicator(smoke, 4) * -0.10366306397571923
    a += age_2 * _smoke_indicator(smoke, 5) * -0.1399185359171839
    a += age_2 * frame["atrial_fibrillation"].to_numpy() * -0.0761826510111625
    a += age_2 * frame["regular_steroid_tablets"].to_numpy() * -0.12005364946742472
    a += age_2 * frame["migraine"].to_numpy() * -0.06558691789869986
    a += age_2 * frame["chronic_kidney_disease"].to_numpy() * -0.22688873086442507
    a += age_2 * frame["systemic_lupus_erythematosus"].to_numpy() * 0.07734794967901627
    a += age_2 * frame["blood_pressure_treatment"].to_numpy() * 0.0009685782358817444
    a += age_2 * frame["diabetes_type1"].to_numpy() * -0.2872406462448895
    a += age_2 * frame["diabetes_type2"].to_numpy() * -0.09711225259069549
    a += age_2 * bmi_1 * 0.5236995893366443
    a += age_2 * bmi_2 * 0.04574419012232376
    a += age_2 * frame["heart_attack_relative"].to_numpy() * -0.07688505169842304
    a += age_2 * sbp * -0.0015082501423272358
    a += age_2 * town * -0.03159341467496233

    return pd.Series(100.0 * (1.0 - np.power(0.988876402378082, np.exp(a))), index=frame.index)


def _male_qrisk3(frame: pd.DataFrame) -> pd.Series:
    age = frame["age"].to_numpy()
    bmi = frame["bmi"].to_numpy()
    rati = frame["cholesterol_hdl_ratio"].to_numpy()
    sbp = frame["systolic_blood_pressure"].to_numpy()
    sbps5 = frame["std_systolic_blood_pressure"].to_numpy()
    town = frame["townsend"].to_numpy()
    smoke = frame["smoke"].to_numpy().astype(int)
    ethnicity = frame["ethnicity"].to_numpy().astype(int)

    age_1 = np.power(age / 10.0, -1) - 0.234766781330109
    age_2 = np.power(age / 10.0, 3) - 77.28408050537109
    bmi_1 = np.power(bmi / 10.0, -2) - 0.149176135659218
    bmi_2 = np.power(bmi / 10.0, -2) * np.log(bmi / 10.0) - 0.141913309693336
    rati = rati - 4.300998687744141
    sbp = sbp - 128.5715789794922
    sbps5 = sbps5 - 8.756621360778809
    town = town - 0.52630490064621

    a = np.zeros(len(frame), dtype=np.float64)
    a += _lookup(ethnicity, _MALE_ETH_RISK)
    a += _lookup(smoke, _MALE_SMOKE)
    a += age_1 * -17.839781666005575
    a += age_2 * 0.0022964880605765492
    a += bmi_1 * 2.456277666053636
    a += bmi_2 * -8.301112231471135
    a += rati * 0.17340196856327111
    a += sbp * 0.012910126542553305
    a += sbps5 * 0.010251914291290456
    a += town * 0.033268201277287295

    for key, coef in [
        ("atrial_fibrillation", 0.8820923692805466),
        ("atypical_antipsychotic", 0.13046879855173513),
        ("regular_steroid_tablets", 0.45485399750445543),
        ("erectile_dysfunction", 0.22251859086705383),
        ("migraine", 0.25584178074159913),
        ("rheumatoid_arthritis", 0.20970658013956567),
        ("chronic_kidney_disease", 0.7185326128827438),
        ("severe_mental_illness", 0.12133039882047164),
        ("systemic_lupus_erythematosus", 0.4401572174457522),
        ("blood_pressure_treatment", 0.5165987108269547),
        ("diabetes_type1", 1.2343425521675175),
        ("diabetes_type2", 0.8594207143093222),
        ("heart_attack_relative", 0.5405546900939016),
    ]:
        a += frame[key].to_numpy() * coef

    a += age_1 * _smoke_indicator(smoke, 2) * -0.21011133933516346
    a += age_1 * _smoke_indicator(smoke, 3) * 0.7526867644750319
    a += age_1 * _smoke_indicator(smoke, 4) * 0.9931588755640579
    a += age_1 * _smoke_indicator(smoke, 5) * 2.1331163414389076
    a += age_1 * frame["atrial_fibrillation"].to_numpy() * 3.4896675530623207
    a += age_1 * frame["regular_steroid_tablets"].to_numpy() * 1.1708133653489108
    a += age_1 * frame["erectile_dysfunction"].to_numpy() * -1.506400985745431
    a += age_1 * frame["migraine"].to_numpy() * 2.349115987140244
    a += age_1 * frame["chronic_kidney_disease"].to_numpy() * -0.5065671632722369
    a += age_1 * frame["blood_pressure_treatment"].to_numpy() * 6.511458109853267
    a += age_1 * frame["diabetes_type1"].to_numpy() * 5.337986487800653
    a += age_1 * frame["diabetes_type2"].to_numpy() * 3.646181740622131
    a += age_1 * bmi_1 * 31.004952956033886
    a += age_1 * bmi_2 * -111.29157184391643
    a += age_1 * frame["heart_attack_relative"].to_numpy() * 2.7808628508531887
    a += age_1 * sbp * 0.018858524469865853
    a += age_1 * town * -0.1007554870063731
    a += age_2 * _smoke_indicator(smoke, 2) * -0.0004985487027532612
    a += age_2 * _smoke_indicator(smoke, 3) * -0.0007987563331738541
    a += age_2 * _smoke_indicator(smoke, 4) * -0.000837061842662513
    a += age_2 * _smoke_indicator(smoke, 5) * -0.0007840031915563729
    a += age_2 * frame["atrial_fibrillation"].to_numpy() * -0.0003499560834063605
    a += age_2 * frame["regular_steroid_tablets"].to_numpy() * -0.0002496045095297166
    a += age_2 * frame["erectile_dysfunction"].to_numpy() * -0.0011058218441227373
    a += age_2 * frame["migraine"].to_numpy() * 0.0001989644604147863
    a += age_2 * frame["chronic_kidney_disease"].to_numpy() * -0.0018325930166498813
    a += age_2 * frame["blood_pressure_treatment"].to_numpy() * 0.0006383805310416501
    a += age_2 * frame["diabetes_type1"].to_numpy() * 0.0006409780808752897
    a += age_2 * frame["diabetes_type2"].to_numpy() * -0.00024695695588868315
    a += age_2 * bmi_1 * 0.005038010235632203
    a += age_2 * bmi_2 * -0.013074483002524319
    a += age_2 * frame["heart_attack_relative"].to_numpy() * -0.00024791809907396037
    a += age_2 * sbp * -0.00001271874191588457
    a += age_2 * town * -0.00009329964232327289

    return pd.Series(100.0 * (1.0 - np.power(0.977268040180206, np.exp(a))), index=frame.index)


@ScoreRegistry.auto
class QRISK3Score(BaseScoreCalculator):
    """QRISK3 10-year cardiovascular risk calculator."""

    name = "qrisk3"

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        super().__init__(column_mapping)

    def required_features(self) -> List[str]:
        return list(QRISK3_FEATURES)

    def output_columns(self) -> List[str]:
        return ["qrisk3", "qrisk3_1digit"]

    def _calculate(
        self,
        data: pd.DataFrame,
        column_mapping: Dict[str, str],
    ) -> pd.DataFrame:
        frame = pd.DataFrame(index=data.index)
        frame["sex"] = _normalize_sex(data[column_mapping["sex"]])
        frame["age"] = _coerce_numeric(data[column_mapping["age"]])
        for key in QRISK3_FEATURES[2:]:
            frame[key] = _coerce_numeric(data[column_mapping[key]])

        frame["bmi"] = frame["weight"] / np.square(frame["height"] / 100.0)

        score = pd.Series(np.nan, index=data.index, dtype="float64")
        female_mask = frame["sex"] == "female"
        male_mask = frame["sex"] == "male"

        if female_mask.any():
            score.loc[female_mask] = _female_qrisk3(frame.loc[female_mask])
        if male_mask.any():
            score.loc[male_mask] = _male_qrisk3(frame.loc[male_mask])

        missing = frame.drop(columns=["sex"]).isna().any(axis=1) | frame["sex"].isna()
        score.loc[missing] = np.nan
        rounded = score.round(1)

        return pd.DataFrame({"qrisk3": score, "qrisk3_1digit": rounded}, index=data.index)