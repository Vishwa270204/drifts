"""Adaptation strategies: data selection (plan) -> training (execute). Add a Strategy to STRATEGIES to add an option."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .settings import KEEP, UPDATE


@dataclass
class Plan:
    kind: str                       # keep | continue
    X: Optional[pd.DataFrame] = None
    y: Optional[np.ndarray] = None


@dataclass
class Strategy:
    name: str
    plan: Callable


def plan_update(sess, Xo, yo, Xn, yn):
    return Plan("continue", Xn, yn)


def plan_keep(sess, Xo, yo, Xn, yn):
    return Plan("keep")


STRATEGIES = [Strategy(UPDATE, plan_update), Strategy(KEEP, plan_keep)]
REG = {s.name: s for s in STRATEGIES}


def build(sess, name, data):
    """Return a model for the strategy. Never mutates the deployed model (updates use a deep copy)."""
    plan = REG[name].plan(sess, *data)
    if plan.kind == "keep":
        return sess.model
    if plan.kind == "continue":
        return copy.deepcopy(sess.model).learn(plan.X, plan.y)
    raise ValueError(f"unknown plan kind {plan.kind}")
