"""Constants and configuration. The UI only exposes batch size and sensitivity; everything else is automatic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
MIN_FIT = 30                    # minimum usable labelled rows to train a tree
UPDATE, KEEP = "Update model", "Keep current model"
METRICS = {"classification": ["Error rate", "1 - macro F1"], "regression": ["MAE"]}
ROW_WISE = {"Error rate", "MAE"}   # metrics with a per-row loss (paired noise check possible)


@dataclass
class Config:
    train_pct: int = 40             # share of rows used for the initial model
    B: int = 100                    # batch size (rows)
    z: float = 3.0                  # drift threshold multiplier (see SENS)
    delay: int = 0                  # label delay (batches)
    min_rows: int = 400             # new labelled rows needed before asking
    snooze: int = 10                # batches without alerts after "keep"
    hold_frac: float = 0.25         # share of new labelled rows held out for evaluation
    metric: Optional[str] = None    # None = chosen automatically
    seed: int = 42
    grace: int = 200                # Hoeffding Adaptive Tree internals
    delta: float = 1e-7
    tau: float = 0.05
    adwin: float = 0.002
    win_thr: int = 300

    @classmethod
    def simple(cls, batch_size: int = 100, sensitivity: str = "Medium") -> "Config":
        """The two user-facing choices; the rest scales from them."""
        return cls(B=int(batch_size), z=SENS[sensitivity], min_rows=max(400, 4 * int(batch_size)))
