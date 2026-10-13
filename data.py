"""CSV reading and column preparation."""
from __future__ import annotations

import io
import re
import warnings

import numpy as np
import pandas as pd


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            out = pd.read_csv(io.BytesIO(data), encoding=enc)
            if out.shape[1] <= 1:
                out = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return out
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file as CSV.")


def label_str(s):
    """Stable string labels: integral numbers become '0','1' (not '0.0','1.0')."""
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.to_numpy(float)
        if len(v) and np.all(np.isfinite(v)) and np.all(np.mod(v, 1) == 0):
            return v.astype(np.int64).astype(str)
    return s.astype(str).to_numpy()


def guess_task(df, target):
    numeric = pd.api.types.is_numeric_dtype(df[target]) and not pd.api.types.is_bool_dtype(df[target])
    return "regression" if (numeric and df[target].nunique() > 10) else "classification"


def detect_time_col(df, target):
    """A column that clearly encodes time order, or None (then row order is used)."""
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            return c
        if not re.search(r"time|date|stamp", str(c), re.I):
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.is_monotonic_increasing:
                return c
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s.head(200), errors="coerce", utc=True)
        if parsed.notna().mean() > 0.95:
            return c
    return None


def build_frame(df, num, cat):
    parts = {}
    for c in num:
        v = np.array(pd.to_numeric(df[c], errors="coerce"), dtype=float)
        v[~np.isfinite(v)] = np.nan
        parts[c] = v
    for c in cat:
        parts[c] = np.where(df[c].notna(), df[c].astype(str), "(missing)")
    return pd.DataFrame(parts, index=df.index)


def pick_features(df, exclude):
    num, cat = [], []
    for c in df.columns:
        if c in exclude:
            continue
        s, n = df[c], df[c].notna().sum()
        if n == 0 or s.nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if not (s.nunique() / n > 0.99 and bool(np.all(np.mod(s.dropna(), 1) == 0))):  # skip id-like
                num.append(c)
        elif s.nunique() <= 50:
            cat.append(c)
    return num, cat
