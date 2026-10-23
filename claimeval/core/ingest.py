"""Tolerant loader for third-party evaluation outputs.

HAL's trace schema is not documented in its README and the repo is archived, so
this does not hard-code it. Instead: flatten whatever JSON you have, show the
candidate fields (`inspect`), guess a mapping from common aliases, and let an
explicit mapping override the guess. Run `hal-decrypt` first — this reads
decrypted JSON, not the encrypted upload archives.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .schema import Rollout, RolloutSet

ALIASES: dict[str, tuple[str, ...]] = {
    "task": ("task_id", "instance_id", "task", "id", "problem_id", "question_id", "sample_id"),
    "success": ("success", "resolved", "passed", "is_correct", "correct", "score", "reward", "accuracy"),
    "cost": ("cost_usd", "total_cost", "cost", "usd", "price", "total_usd"),
    "config": ("agent_name", "agent", "scaffold", "harness", "model_name", "model", "config", "run_name"),
    "seed": ("seed", "run_id", "rollout_id", "attempt", "trial", "repeat"),
    "benchmark": ("benchmark", "benchmark_name", "dataset", "suite"),
}

_TRUTHY = {"true", "yes", "pass", "passed", "resolved", "success", "1"}
_FALSY = {"false", "no", "fail", "failed", "unresolved", "error", "0"}


@dataclass
class FieldMap:
    """Dotted-path field mapping. `config` joins several fields with '|'."""

    config: list[str] = field(default_factory=list)
    task: str = ""
    success: str = ""
    cost: str = ""
    seed: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "FieldMap":
        cfg = d.get("config", [])
        return cls(
            config=[cfg] if isinstance(cfg, str) else list(cfg),
            task=d.get("task", ""),
            success=d.get("success", ""),
            cost=d.get("cost", ""),
            seed=d.get("seed", ""),
        )


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted paths, keeping scalars only."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    # lists (message histories, trajectories) are deliberately not descended into
    return out


def iter_records(path: str | Path) -> Iterator[dict]:
    """Yield flat record dicts from a file or directory of JSON/JSONL."""
    p = Path(path)
    files = (
        sorted(f for f in p.rglob("*") if f.suffix in {".json", ".jsonl"})
        if p.is_dir()
        else [p]
    )
    for f in files:
        if f.suffix == ".jsonl":
            for line in f.read_text().splitlines():
                if line.strip():
                    yield from _expand(json.loads(line), f)
        else:
            yield from _expand(json.loads(f.read_text()), f)


def _expand(obj: Any, src: Path) -> Iterator[dict]:
    """Normalize the three shapes these files come in."""
    base = {"_file": src.stem}
    if isinstance(obj, list):
        for item in obj:
            yield {**base, **flatten(item)}
    elif isinstance(obj, dict):
        # a dict of {task_id: {...}} is the documented HAL agent-output shape;
        # detect it by the values being dicts rather than scalars
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals) and len(obj) > 1:
            for k, v in obj.items():
                yield {**base, "_key": k, **flatten(v)}
        else:
            yield {**base, **flatten(obj)}


def inspect_path(path: str | Path, limit: int = 500) -> dict:
    """Report which fields exist, how often, and with what values."""
    keys: Counter = Counter()
    samples: dict[str, list] = defaultdict(list)
    n = 0
    for rec in iter_records(path):
        n += 1
        for k, v in rec.items():
            keys[k] += 1
            if len(samples[k]) < 3 and v is not None:
                samples[k].append(v)
        if n >= limit:
            break
    return {"n_records": n, "keys": keys, "samples": dict(samples), "guess": guess_map(keys)}


def guess_map(keys: Counter) -> FieldMap:
    """Best-effort mapping from field aliases. Always verify before trusting."""
    present = list(keys)

    def pick(role: str) -> str:
        for alias in ALIASES[role]:
            for k in present:
                if k.split(".")[-1].lower() == alias:
                    return k
        return ""

    cfg = [k for k in (pick("config"),) if k]
    # a config is model x harness; if both are present, compose them
    model = next((k for k in present if k.split(".")[-1].lower() in ("model", "model_name")), "")
    harness = next(
        (k for k in present if k.split(".")[-1].lower() in ("agent_name", "agent", "scaffold", "harness")),
        "",
    )
    if model and harness:
        cfg = [harness, model]
    task = pick("task") or ("_key" if "_key" in present else "")
    return FieldMap(config=cfg, task=task, success=pick("success"), cost=pick("cost"), seed=pick("seed"))


def _as_success(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUTHY:
            return 1.0
        if s in _FALSY:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return None
    return None


def load(path: str | Path, fmap: FieldMap, strict: bool = False) -> tuple[RolloutSet, dict]:
    """Materialize rollouts. Returns the set plus a dropped-record tally.

    Records that fail to parse are counted by reason rather than silently
    skipped — a loader that quietly drops the expensive crashed runs would
    manufacture exactly the bias this project exists to measure.
    """
    rolls: list[Rollout] = []
    dropped: Counter = Counter()
    seq: Counter = Counter()

    for rec in iter_records(path):
        cfg = "|".join(str(rec.get(k, "?")) for k in fmap.config) if fmap.config else "default"
        task = rec.get(fmap.task) if fmap.task else None
        succ = _as_success(rec.get(fmap.success)) if fmap.success else None
        cost = rec.get(fmap.cost) if fmap.cost else None

        if task is None:
            dropped["no_task"] += 1
            continue
        if succ is None:
            dropped["no_success"] += 1
            continue
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            dropped["no_cost"] += 1
            continue

        if fmap.seed and rec.get(fmap.seed) is not None:
            seed = str(rec[fmap.seed])
        else:
            key = (cfg, str(task))
            seed = str(seq[key])
            seq[key] += 1

        rolls.append(Rollout(cfg, str(task), seed, succ, cost))

    if strict and dropped:
        raise ValueError(f"dropped records: {dict(dropped)}")
    if not rolls:
        raise ValueError(f"no usable records; dropped={dict(dropped)}. Run `inspect` first.")
    return RolloutSet(rolls), dict(dropped)
