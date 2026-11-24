"""Run comparison and the camus10 adoption rule.

  python rag/compare_runs.py --base F [F ...] --cand F [F ...] [--base-fool F --cand-fool F]

Compares judge means and tells between two sets of probe_scores.jsonl runs and
applies the adoption rule: the guard holds unless voice, factuality or composite
falls by more than the noise band, and the candidate is more human-like when the
mean tell rate falls or the fool rate rises. The owner's blind A/B still decides.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tells  # noqa: E402

AXES = ("voice", "factuality", "engagement")
GUARD_AXES = ("voice", "factuality", "composite")
MEM_AXES = ("voice", "memory", "naturalness")
MEM_GUARD_AXES = ("memory", "naturalness", "composite")
FOOL_MIN_RISE = 0.05


def _load_all(paths):
    if isinstance(paths, str):
        paths = [paths]
    rows = []
    for path in paths or []:
        rows.extend(tells._load_rows(path))
    return rows


def _means_for(paths, axes):
    n = 0
    sums = {k: 0.0 for k in axes}
    composite = 0.0
    for row in _load_all(paths):
        scores = row.get("scores") if isinstance(row, dict) else None
        if not isinstance(scores, dict):
            continue
        try:
            vals = {k: float(scores[k]) for k in axes}
        except (KeyError, TypeError, ValueError):
            continue
        n += 1
        for k, v in vals.items():
            sums[k] += v
        composite += sum(vals.values()) / len(axes)
    means = {k: (sums[k] / n if n else 0.0) for k in axes}
    means["composite"] = (composite / n if n else 0.0)
    return means, n


def _means(paths):
    return _means_for(paths, AXES)


def judge_means(paths):
    means, _ = _means(paths)
    return means


def memory_means(paths):
    means, _ = _means_for(paths, MEM_AXES)
    return means


def _tell_mean(report):
    rates = (report or {}).get("rates") or {}
    vals = [float(v) for v in rates.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def verdict(base, cand, noise=0.1):
    bj = base.get("judge", {}) if isinstance(base, dict) else {}
    cj = cand.get("judge", {}) if isinstance(cand, dict) else {}
    reasons = []

    guard_ok = True
    for axis in GUARD_AXES:
        try:
            drop = float(bj.get(axis, 0.0)) - float(cj.get(axis, 0.0))
        except (TypeError, ValueError):
            drop = 0.0
        if round(drop, 9) > noise:
            guard_ok = False
            reasons.append(f"{axis} fell {drop:.3f} (> {noise})")

    bm = base.get("memory") if isinstance(base, dict) else None
    cm = cand.get("memory") if isinstance(cand, dict) else None
    if isinstance(bm, dict) and isinstance(cm, dict) and bm and cm:
        for axis in MEM_GUARD_AXES:
            try:
                drop = float(bm.get(axis, 0.0)) - float(cm.get(axis, 0.0))
            except (TypeError, ValueError):
                drop = 0.0
            if round(drop, 9) > noise:
                guard_ok = False
                reasons.append(f"memory {axis} fell {drop:.3f} (> {noise})")

    base_tells = _tell_mean(base.get("tells") if isinstance(base, dict) else None)
    cand_tells = _tell_mean(cand.get("tells") if isinstance(cand, dict) else None)
    base_fool = base.get("fool") if isinstance(base, dict) else None
    cand_fool = cand.get("fool") if isinstance(cand, dict) else None

    humanlike_better = False
    if cand_tells < base_tells:
        humanlike_better = True
        reasons.append(f"mean tell rate fell {base_tells:.3f} -> {cand_tells:.3f}")
    if base_fool is not None and cand_fool is not None:
        rise = round(float(cand_fool) - float(base_fool), 9)
        if rise >= FOOL_MIN_RISE:
            humanlike_better = True
            reasons.append(f"fool rate rose {float(base_fool):.3f} -> {float(cand_fool):.3f}")

    return {"guard_ok": guard_ok, "humanlike_better": humanlike_better,
            "reasons": reasons}


def _read_fool_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    value = data.get("overall", data.get("fool")) if isinstance(data, dict) else data
    return float(value)


def fool_value(spec):
    if spec is None:
        return None
    if os.path.exists(str(spec)):
        return _read_fool_file(str(spec))
    return float(spec)


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def _table_row(name, base, cand):
    delta = ""
    if base is not None and cand is not None:
        delta = f"{cand - base:+.3f}"
    return f"{name:<12} {_fmt(base):>8} {_fmt(cand):>8} {delta:>8}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare two eval runs and apply the adoption rule.")
    ap.add_argument("--base", nargs="+", required=True, metavar="F",
                    help="probe_scores.jsonl file(s) for the base run")
    ap.add_argument("--cand", nargs="+", required=True, metavar="F",
                    help="probe_scores.jsonl file(s) for the candidate run")
    ap.add_argument("--base-fool", default=None, metavar="F",
                    help="base fool rate (number or fool_rate.json)")
    ap.add_argument("--cand-fool", default=None, metavar="F",
                    help="candidate fool rate (number or fool_rate.json)")
    ap.add_argument("--base-memory", nargs="+", default=None, metavar="F",
                    help="memory_scores.jsonl file(s) for the base run")
    ap.add_argument("--cand-memory", nargs="+", default=None, metavar="F",
                    help="memory_scores.jsonl file(s) for the candidate run")
    args = ap.parse_args(argv)

    base_judge, base_n = _means(args.base)
    cand_judge, cand_n = _means(args.cand)
    base_tells = tells.report(_load_all(args.base))
    cand_tells = tells.report(_load_all(args.cand))
    base_fool = fool_value(args.base_fool)
    cand_fool = fool_value(args.cand_fool)
    base_mem, base_mem_n = _means_for(args.base_memory, MEM_AXES) if args.base_memory else (None, 0)
    cand_mem, cand_mem_n = _means_for(args.cand_memory, MEM_AXES) if args.cand_memory else (None, 0)

    print(f"{'metric':<12} {'base':>8} {'cand':>8} {'delta':>8}")
    for axis in AXES + ("composite",):
        print(_table_row(axis, base_judge[axis], cand_judge[axis]))
    print(_table_row("tell_mean", _tell_mean(base_tells), _tell_mean(cand_tells)))
    print(_table_row("fool", base_fool, cand_fool))
    if base_mem and cand_mem and base_mem_n and cand_mem_n:
        labels = {"voice": "mem_voice", "memory": "mem_recall", "naturalness": "mem_nat", "composite": "mem_comp"}
        for axis in MEM_AXES + ("composite",):
            print(_table_row(labels[axis], base_mem[axis], cand_mem[axis]))

    v = verdict({"judge": base_judge, "tells": base_tells, "fool": base_fool,
                 "memory": base_mem if base_mem_n else None},
                {"judge": cand_judge, "tells": cand_tells, "fool": cand_fool,
                 "memory": cand_mem if cand_mem_n else None})
    print()
    print(f"scored rows: base {base_n}, cand {cand_n}")
    print(f"guard_ok: {v['guard_ok']}")
    print(f"humanlike_better: {v['humanlike_better']}")
    print("reasons:")
    for reason in v["reasons"]:
        print(f"  - {reason}")
    print("owner's blind A/B decides adoption")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
