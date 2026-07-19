"""
features.py — single source of truth for CartSense feature engineering.

Imported by BOTH Student_Project.ipynb (training) and app.py (serving) so that
training-time and serving-time transformations can never drift apart.
"""
import numpy as np
import pandas as pd

RAW_NUMERIC = ["Administrative", "Administrative_Duration", "Informational",
               "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
               "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]

COUNT_DURATION_PAIRS = [("Administrative", "Administrative_Duration"),
                        ("Informational", "Informational_Duration"),
                        ("ProductRelated", "ProductRelated_Duration")]

LOG_COLS = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
            "TotalDuration", "AvgProductDwell", "PageValues"]


def add_tracking_gap(d: pd.DataFrame) -> pd.DataFrame:
    """Flag sessions whose page count > 0 but recorded dwell time == 0.

    These are physically impossible and represent *implicit* missingness caused by
    analytics-tag failures. We flag rather than delete: the gap itself is informative
    and deleting 1,069 rows (8.8%) would bias the sample.
    """
    d = d.copy()
    gap = np.zeros(len(d), dtype=bool)
    for c, dur in COUNT_DURATION_PAIRS:
        gap |= ((d[c] > 0) & (d[dur] == 0)).values
    d["TrackingGap"] = gap.astype(int)
    return d


def engineer(d: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven derived features + heavy-tail compression."""
    d = d.copy()
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDuration"] = (d["Administrative_Duration"] + d["Informational_Duration"]
                          + d["ProductRelated_Duration"])
    d["AvgProductDwell"] = (d["ProductRelated_Duration"] / d["ProductRelated"].replace(0, np.nan)).fillna(0.0)
    d["PageDepthRatio"] = (d["ProductRelated"] / d["TotalPages"].replace(0, np.nan)).fillna(0.0)
    d["ExitBounceGap"] = d["ExitRates"] - d["BounceRates"]
    d["HasPageValue"] = (d["PageValues"] > 0).astype(int)
    for c in LOG_COLS:
        d["log_" + c] = np.log1p(d[c])
    d["Weekend"] = d["Weekend"].astype(int)
    return d


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """Full raw -> model-ready transformation. Target column dropped if present."""
    d = add_tracking_gap(d)
    d = engineer(d)
    return d.drop(columns=[c for c in ["Revenue"] if c in d.columns])
