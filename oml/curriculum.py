"""Curriculum designs. Each returns a list of stages (HF Datasets); the trainer runs stages
sequentially, carrying the policy forward. Controls: `none` (single shuffled stage) and `anti`.

Difficulty source:
  * `level`  -- MATH Level 1..5 shipped with the data (no extra compute)
  * `passrate` -- per-problem pass rate of the starting policy, from `oml.difficulty`
                  (a jsonl of {uid, pass_rate}); lower pass rate == harder
"""
from __future__ import annotations

import json

from datasets import Dataset


def _attach_passrate(ds: Dataset, path: str) -> Dataset:
    pr = {}
    for line in open(path):
        r = json.loads(line)
        pr[r["uid"]] = r["pass_rate"]
    ds = ds.filter(lambda e: e["uid"] in pr)
    return ds.add_column("pass_rate", [pr[u] for u in ds["uid"]])


def _split(rows_sorted: list[dict], n_stages: int) -> list[Dataset]:
    k = len(rows_sorted)
    return [Dataset.from_list(rows_sorted[i * k // n_stages : (i + 1) * k // n_stages]) for i in range(n_stages)]


def build_stages(ds: Dataset, mode: str = "none", n_stages: int = 3, difficulty: str = "passrate", passrate_path: str | None = None, seed: int = 0,
                 keep_min: float | None = None, keep_max: float | None = None, subset_size: int | None = None) -> list[Dataset]:
    """mode: none | easy2hard | hard2easy(anti) | filter | filter_easy2hard
            | band      -- single stage, keep problems with keep_min <= pass_rate <= keep_max (pass rate is k/8 of the starting policy)
            | randsub   -- single stage, a seed-dependent random subset of `subset_size` problems (size-matched control for filters)
    """
    if mode == "randsub":
        return [ds.shuffle(seed=seed).select(range(subset_size))]
    if mode == "band":
        ds = _attach_passrate(ds, passrate_path)
        lo = -1.0 if keep_min is None else keep_min - 1e-9
        hi = 2.0 if keep_max is None else keep_max + 1e-9
        ds = ds.filter(lambda e: lo <= e["pass_rate"] <= hi)
        return [ds.shuffle(seed=seed)]
    if mode == "none":
        return [ds.shuffle(seed=seed)]

    if difficulty == "passrate":
        if not passrate_path:
            raise ValueError("difficulty=passrate needs passrate_path (run oml.difficulty first)")
        ds = _attach_passrate(ds, passrate_path)
        key = lambda e: -e["pass_rate"]  # easy first (high pass rate)
    else:
        key = lambda e: e["level"]  # level 1 (easy) first

    if mode.startswith("filter") and difficulty == "passrate":
        # dynamic-sampling-style static filter: drop problems the policy always/never solves (zero advantage)
        ds = ds.filter(lambda e: 0.0 < e["pass_rate"] < 1.0)
        mode = mode.replace("filter", "").lstrip("_") or "none"
        if mode == "none":
            return [ds.shuffle(seed=seed)]

    rows = sorted(ds.shuffle(seed=seed).to_list(), key=key)  # shuffle first => random tie-breaking
    if mode == "easy2hard":
        return _split(rows, n_stages)
    if mode in ("hard2easy", "anti"):
        return _split(rows[::-1], n_stages)
    raise ValueError(f"unknown curriculum mode {mode}")
