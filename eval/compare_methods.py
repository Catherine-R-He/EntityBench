"""
compare_methods.py

Paired head-to-head comparison of two methods on the same set of episodes.
Reads per-episode JSONs from each method's eval directory and produces:

  - per_metric_comparison.tsv : every metric, both means, paired delta, Cohen's d, p-value
  - headline_metrics.md       : the metrics ranked by effect size, grouped by pillar
  - per_episode_deltas.tsv    : per-episode paired deltas (for follow-up plots)

Usage:
  python compare_methods.py \\
      --method_a eval_results/method_a \\
      --label_a  method_a \\
      --method_b eval_results/method_b \\
      --label_b  method_b \\
      --out_dir  comparison/

Statistical approach
--------------------
Paired analysis: for each metric we form n paired (a_i, b_i) samples
where i indexes episodes that BOTH methods evaluated. We report:
  - mean_a, std_a, mean_b, std_b  : per-method aggregates
  - delta_mean = mean(a - b)      : paired mean difference (a minus b)
  - cohens_d                      : effect size of the paired difference
  - paired_t_p                    : two-sided paired t-test p-value
  - wilcoxon_p                    : two-sided Wilcoxon signed-rank p-value
  - n_paired                      : how many episodes contributed
  - n_a_higher / n_b_higher       : sign breakdown (qualitative)

With n=10 episodes the t-test is underpowered, so Cohen's d (effect
size, scale-invariant) is the more useful signal. d > 0.8 is "large",
0.5-0.8 "medium", < 0.5 "small" by Cohen's rough rule of thumb.

We do NOT correct for multiple comparisons here — the goal is to pick
headline metrics, not to claim individual statistical significance.
The paper's main result will be a single chosen metric per pillar.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _load_episodes(eval_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all per-episode JSONs from a method's eval directory.

    Returns dict mapping episode_id -> parsed JSON. Skips report/manifest files.
    """
    episodes: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(eval_dir):
        sys.exit(f"Not a directory: {eval_dir}")
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        bn = os.path.basename(path)
        if bn.startswith(("report_", "run_manifest", "_")):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  WARN: failed to parse {bn}: {e}", file=sys.stderr)
            continue
        eid = d.get("episode_id") or os.path.splitext(bn)[0]
        episodes[eid] = d
    return episodes


def _flatten_metrics(ep: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull every numeric metric value out of an episode JSON.

    Combines vbench_means + metrics into a single flat dict. Returns
    None for missing-or-null values so the caller can distinguish
    "metric exists but couldn't be computed" from "metric absent".
    """
    out: Dict[str, Optional[float]] = {}
    for source_key in ("vbench_means", "metrics"):
        d = ep.get(source_key, {}) or {}
        for k, v in d.items():
            if isinstance(v, dict) and "value" in v:
                val = v["value"]
                out[k] = float(val) if isinstance(val, (int, float)) else None
            elif isinstance(v, (int, float)):
                out[k] = float(v)
    return out


def _classify_pillar(metric_name: str) -> str:
    """Group each metric into one of three pillars."""
    if metric_name in {
        "subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality",
        "imaging_quality",
    }:
        return "1_intra_shot_quality"
    if metric_name.startswith("intra_"):
        return "3_intra_shot_alignment"
    # cs_* = DINOv2 cross-shot; llm_* = LLM-judge cross-shot
    # (vlm_* kept for backward-compat with caches before the rename).
    if metric_name.startswith(("cs_", "llm_", "vlm_")):
        return "2_cross_shot_consistency"
    return "other"


def _cohens_d_paired(deltas: List[float]) -> Optional[float]:
    """Cohen's d for paired-sample mean difference: mean(d) / std(d).

    Note: this is the paired-sample d, NOT the unpaired d. With paired
    data, the std should be of the differences, not pooled across samples.
    Returns None for n<2 or zero variance.
    """
    n = len(deltas)
    if n < 2:
        return None
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var)


def _paired_t_p(deltas: List[float]) -> Optional[float]:
    """Two-sided paired t-test p-value for H0: mean(deltas) = 0."""
    n = len(deltas)
    if n < 2:
        return None
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    if var <= 0:
        return 1.0 if mean == 0 else 0.0
    se = math.sqrt(var / n)
    t = mean / se
    # Two-sided p via Student's t CDF; use scipy if available, else
    # a normal approximation (fine for sanity checks at our n).
    try:
        from scipy import stats
        return float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
    except ImportError:
        # Normal approx — overstates significance for small n
        z = abs(t)
        return float(2 * 0.5 * math.erfc(z / math.sqrt(2)))


def _wilcoxon_p(deltas: List[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value. Requires scipy."""
    nz = [d for d in deltas if d != 0]
    if len(nz) < 2:
        return None
    try:
        from scipy import stats
        return float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Paired head-to-head comparison of two evaluation runs."
    )
    parser.add_argument("--method_a", required=True, help="eval_results dir for method A")
    parser.add_argument("--method_b", required=True, help="eval_results dir for method B")
    parser.add_argument("--label_a", required=True, help="display label for A (e.g. 'ours')")
    parser.add_argument("--label_b", required=True, help="display label for B (e.g. 'baseline')")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=8,
                        help="Show top-K differentiating metrics per pillar (default 8)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading episodes...")
    eps_a = _load_episodes(args.method_a)
    eps_b = _load_episodes(args.method_b)
    print(f"  {args.label_a}: {len(eps_a)} episodes from {args.method_a}")
    print(f"  {args.label_b}: {len(eps_b)} episodes from {args.method_b}")

    # Pair on episode_id
    paired_ids = sorted(set(eps_a.keys()) & set(eps_b.keys()))
    only_a = sorted(set(eps_a.keys()) - set(eps_b.keys()))
    only_b = sorted(set(eps_b.keys()) - set(eps_a.keys()))
    print(f"\n  paired:    {len(paired_ids)} episodes")
    if only_a:
        print(f"  only in {args.label_a}: {len(only_a)}  (will be excluded from paired stats)")
        for eid in only_a[:3]:
            print(f"    - {eid}")
    if only_b:
        print(f"  only in {args.label_b}: {len(only_b)}")
        for eid in only_b[:3]:
            print(f"    - {eid}")

    if len(paired_ids) < 2:
        sys.exit("Need at least 2 paired episodes to compare.")

    # Build flat per-episode metric tables
    rows_a = {eid: _flatten_metrics(eps_a[eid]) for eid in paired_ids}
    rows_b = {eid: _flatten_metrics(eps_b[eid]) for eid in paired_ids}

    # Union of all metric names seen anywhere
    all_metrics = set()
    for d in list(rows_a.values()) + list(rows_b.values()):
        all_metrics.update(d.keys())
    all_metrics = sorted(all_metrics)
    print(f"  metrics:   {len(all_metrics)}")

    # ---- Per-metric paired statistics ----
    results = []
    for m in all_metrics:
        # Build paired (a_i, b_i) samples — drop any episode where either
        # side has no value (None / missing) for this metric
        pairs = []
        for eid in paired_ids:
            va = rows_a[eid].get(m)
            vb = rows_b[eid].get(m)
            if va is not None and vb is not None:
                pairs.append((eid, va, vb))
        if not pairs:
            continue

        a_vals = [p[1] for p in pairs]
        b_vals = [p[2] for p in pairs]
        deltas = [a - b for a, b in zip(a_vals, b_vals)]

        n = len(pairs)
        mean_a = sum(a_vals) / n
        mean_b = sum(b_vals) / n
        std_a = math.sqrt(sum((v - mean_a) ** 2 for v in a_vals) / max(1, n - 1)) if n > 1 else 0.0
        std_b = math.sqrt(sum((v - mean_b) ** 2 for v in b_vals) / max(1, n - 1)) if n > 1 else 0.0
        delta_mean = sum(deltas) / n
        d = _cohens_d_paired(deltas)
        t_p = _paired_t_p(deltas)
        w_p = _wilcoxon_p(deltas)
        n_a_higher = sum(1 for x in deltas if x > 0)
        n_b_higher = sum(1 for x in deltas if x < 0)
        n_tied = sum(1 for x in deltas if x == 0)

        results.append({
            "metric": m,
            "pillar": _classify_pillar(m),
            "n_paired": n,
            "mean_a": mean_a, "std_a": std_a,
            "mean_b": mean_b, "std_b": std_b,
            "delta_mean": delta_mean,
            "cohens_d": d,
            "paired_t_p": t_p,
            "wilcoxon_p": w_p,
            "n_a_higher": n_a_higher,
            "n_b_higher": n_b_higher,
            "n_tied": n_tied,
        })

    # ---- Write per-metric TSV ----
    tsv_path = os.path.join(args.out_dir, "per_metric_comparison.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        cols = ["pillar", "metric", "n_paired",
                f"mean_{args.label_a}", f"std_{args.label_a}",
                f"mean_{args.label_b}", f"std_{args.label_b}",
                "delta_mean", "cohens_d",
                f"n_{args.label_a}_higher", f"n_{args.label_b}_higher", "n_tied",
                "paired_t_p", "wilcoxon_p"]
        f.write("\t".join(cols) + "\n")
        for r in sorted(results, key=lambda r: (r["pillar"], r["metric"])):
            def fmt(v, prec=4):
                if v is None:
                    return ""
                if isinstance(v, float):
                    return f"{v:.{prec}f}"
                return str(v)
            f.write("\t".join([
                r["pillar"], r["metric"], str(r["n_paired"]),
                fmt(r["mean_a"]), fmt(r["std_a"]),
                fmt(r["mean_b"]), fmt(r["std_b"]),
                fmt(r["delta_mean"]), fmt(r["cohens_d"], 3),
                str(r["n_a_higher"]), str(r["n_b_higher"]), str(r["n_tied"]),
                fmt(r["paired_t_p"], 4), fmt(r["wilcoxon_p"], 4),
            ]) + "\n")
    print(f"\n  → {tsv_path}")

    # ---- Headline markdown: top-K per pillar by |Cohen's d| ----
    md_path = os.path.join(args.out_dir, "headline_metrics.md")
    pillar_titles = {
        "1_intra_shot_quality":     "Pillar 1 — Intra-shot quality",
        "3_intra_shot_alignment":   "Pillar 3 — Intra-shot prompt-following alignment",
        "2_cross_shot_consistency": "Pillar 2 — Cross-shot consistency  (the headline pillar)",
        "other":                    "Other metrics",
    }
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Head-to-head: `{args.label_a}` vs `{args.label_b}`\n\n")
        f.write(f"Paired comparison over **{len(paired_ids)} episodes**. ")
        f.write(f"Effect size is paired Cohen's d. Positive `delta_mean` = ")
        f.write(f"`{args.label_a}` scored higher.\n\n")
        f.write("With small n the t-test p-value is underpowered — read **Cohen's d** ")
        f.write("as the primary signal of separation between methods.\n\n")
        f.write("| Cohen's d | interpretation |\n|---|---|\n")
        f.write("| > 0.8 | large effect |\n| 0.5 – 0.8 | medium effect |\n")
        f.write("| 0.2 – 0.5 | small effect |\n| < 0.2 | negligible |\n\n")
        f.write("---\n\n")

        for pillar in ("2_cross_shot_consistency", "1_intra_shot_quality",
                       "3_intra_shot_alignment", "other"):
            in_pillar = [r for r in results if r["pillar"] == pillar]
            if not in_pillar:
                continue
            # Rank by |d| (favors metrics that differentiate, in either direction)
            ranked = sorted(in_pillar,
                            key=lambda r: abs(r["cohens_d"]) if r["cohens_d"] is not None else 0,
                            reverse=True)
            f.write(f"## {pillar_titles[pillar]}\n\n")
            f.write(f"| metric | n | {args.label_a} | {args.label_b} | Δ | d | "
                    f"win count | p (t / wilcoxon) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in ranked[:args.top_k]:
                d = r["cohens_d"]
                d_str = f"**{d:+.2f}**" if (d is not None and abs(d) >= 0.8) \
                        else (f"{d:+.2f}" if d is not None else "—")
                tp = r["paired_t_p"]
                wp = r["wilcoxon_p"]
                p_str = f"{tp:.3f} / {wp:.3f}" if tp is not None and wp is not None \
                        else (f"{tp:.3f} / —" if tp is not None else "—")
                wins = f"{r['n_a_higher']}-{r['n_b_higher']}"
                if r['n_tied']:
                    wins += f" ({r['n_tied']}t)"
                ma = r["mean_a"]; mb = r["mean_b"]; dm = r["delta_mean"]
                f.write(f"| `{r['metric']}` | {r['n_paired']} | "
                        f"{ma:.3f}±{r['std_a']:.3f} | "
                        f"{mb:.3f}±{r['std_b']:.3f} | "
                        f"{dm:+.3f} | {d_str} | {wins} | {p_str} |\n")
            f.write("\n")

        # Recommendations: top-1 metric per pillar by |d|
        f.write("---\n\n## Suggested headline metric per pillar\n\n")
        for pillar in ("2_cross_shot_consistency",
                       "1_intra_shot_quality",
                       "3_intra_shot_alignment"):
            in_pillar = [r for r in results
                         if r["pillar"] == pillar and r["cohens_d"] is not None]
            if not in_pillar:
                continue
            best = max(in_pillar, key=lambda r: abs(r["cohens_d"]))
            direction = args.label_a if best["delta_mean"] > 0 else args.label_b
            f.write(f"- **{pillar_titles[pillar]}** → "
                    f"`{best['metric']}` "
                    f"(d={best['cohens_d']:+.2f}, "
                    f"`{direction}` higher by {abs(best['delta_mean']):.3f})\n")
        f.write("\n")
    print(f"  → {md_path}")

    # ---- Per-episode delta TSV (for follow-up plots) ----
    delta_path = os.path.join(args.out_dir, "per_episode_deltas.tsv")
    # Pick a subset of metrics to keep the file readable: top differentiating
    # ones by |d|, plus the recommended headline metrics.
    keep_metrics = set()
    for pillar in ("2_cross_shot_consistency", "1_intra_shot_quality",
                   "3_intra_shot_alignment"):
        in_pillar = sorted(
            [r for r in results if r["pillar"] == pillar and r["cohens_d"] is not None],
            key=lambda r: abs(r["cohens_d"]), reverse=True,
        )[:args.top_k]
        keep_metrics.update(r["metric"] for r in in_pillar)
    keep_metrics = sorted(keep_metrics)

    with open(delta_path, "w", encoding="utf-8") as f:
        f.write("episode_id\t" + "\t".join(keep_metrics) + "\n")
        for eid in paired_ids:
            cells = [eid]
            for m in keep_metrics:
                va = rows_a[eid].get(m)
                vb = rows_b[eid].get(m)
                if va is not None and vb is not None:
                    cells.append(f"{va - vb:.4f}")
                else:
                    cells.append("")
            f.write("\t".join(cells) + "\n")
    print(f"  → {delta_path}  ({len(keep_metrics)} top metrics × {len(paired_ids)} episodes)")

    # ---- Console summary ----
    print(f"\n{'=' * 72}")
    print(f"SUMMARY")
    print(f"{'=' * 72}")
    for pillar in ("2_cross_shot_consistency", "1_intra_shot_quality",
                   "3_intra_shot_alignment"):
        in_pillar = [r for r in results
                     if r["pillar"] == pillar and r["cohens_d"] is not None]
        if not in_pillar:
            continue
        ranked = sorted(in_pillar, key=lambda r: abs(r["cohens_d"]), reverse=True)
        print(f"\n{pillar_titles[pillar]}:")
        for r in ranked[:5]:
            direction = ">" if r["delta_mean"] > 0 else "<"
            print(f"  d={r['cohens_d']:+.2f}  {r['metric']:38s} "
                  f"{args.label_a} {direction} {args.label_b}  "
                  f"(Δ={r['delta_mean']:+.3f}, "
                  f"wins {r['n_a_higher']}-{r['n_b_higher']})")
    print()


if __name__ == "__main__":
    main()
