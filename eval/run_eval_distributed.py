#!/usr/bin/env python3
"""
run_eval_distributed.py

Distribute evaluate_benchmark.py across multiple GPUs.

Strategy:
  - Split episodes into N shards (one per GPU)
  - Each shard runs a separate evaluate_benchmark.py process
  - Episodes are NOT split — each episode stays on one GPU
  - All shards write to the same --out_dir (per-episode JSONs don't conflict)
  - After all shards finish, run aggregation once

Usage:
  # 8 GPUs, all pillars
  python run_eval_distributed.py \
    --n_gpus 8 \
    --results_dir /path/to/generated_outputs \
    --scripts_dir data/scripts \
    --split_json  data/splits/final_split_validated_ids.json \
    --out_dir     eval_results/<method_name> \
    --method_name <method_name> \
    --pillars 1,2,3 \
    --resume

  # Dry run (print commands without executing)
  python run_eval_distributed.py --n_gpus 8 --dry_run ...

  # Run only shard 3 (for manual retry)
  python run_eval_distributed.py --n_gpus 8 --shard_id 3 ...
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import List, Tuple


def load_episodes(split_json: str) -> List[Tuple[str, str]]:
    """Load episode list with tiers, ordered: easy → medium → hard."""
    with open(split_json) as f:
        split_ids = json.load(f)

    episodes = []
    for tier in ["easy", "medium", "hard"]:
        for eid in split_ids.get(tier, []):
            episodes.append((eid, tier))
    return episodes


def shard_episodes(
    episodes: List[Tuple[str, str]],
    n_shards: int,
) -> List[List[Tuple[str, str]]]:
    """
    Split episodes into N shards, balancing shot count across shards.

    Strategy: round-robin assignment sorted by tier (hard first),
    so each GPU gets a mix of easy/medium/hard episodes with
    roughly equal total compute.
    """
    # Sort: hard first (more shots = more compute), then medium, then easy
    tier_order = {"hard": 0, "medium": 1, "easy": 2}
    sorted_eps = sorted(episodes, key=lambda x: tier_order.get(x[1], 3))

    # Round-robin into shards
    shards = [[] for _ in range(n_shards)]
    for i, ep in enumerate(sorted_eps):
        shards[i % n_shards].append(ep)

    return shards


def build_shard_command(
    shard_id: int,
    gpu_id: int,
    shard_episodes: List[Tuple[str, str]],
    args: argparse.Namespace,
    shard_split_path: str,
) -> str:
    """Build the command to run evaluate_benchmark.py for one shard."""
    cmd_parts = [
        sys.executable,
        "evaluate_benchmark.py",
        f"--results_dir {args.results_dir}",
        f"--scripts_dir {args.scripts_dir}",
        f"--split_json {shard_split_path}",
        f"--out_dir {args.out_dir}",
        f"--method_name {args.method_name}",
        f"--pillars {args.pillars}",
        f"--device cuda:{gpu_id}",
        f"--n_frames {args.n_frames}",
        f"--llm_concurrency {args.llm_concurrency}",
    ]

    if args.resume:
        cmd_parts.append("--resume")
    if args.vbench_cache_dir:
        cmd_parts.append(f"--vbench_cache_dir {args.vbench_cache_dir}")
    if args.vlm_model:
        cmd_parts.append(f"--vlm_model {args.vlm_model}")
    if args.save_action_grids:
        cmd_parts.append("--save_action_grids")

    return " ".join(cmd_parts)


def run_aggregation(args: argparse.Namespace):
    """
    After all shards finish, run aggregation on the collected
    per-episode JSONs to produce the final report.
    """
    print(f"\n{'=' * 60}")
    print("Running final aggregation...")
    print(f"{'=' * 60}")

    # Import aggregation functions
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate_benchmark import (
        EpisodeResult,
        ALL_METRICS,
        VBENCH_DIMENSIONS,
        aggregate_results,
        write_report,
        _episode_result_from_json,  # canonical deserializer with MetricValue support
    )

    # Load split
    with open(args.split_json) as f:
        split_ids = json.load(f)

    # Load all per-episode results using the canonical deserializer
    # so that legacy and current field names both map correctly.
    all_results = []
    missing = []

    episodes = load_episodes(args.split_json)
    for eid, tier in episodes:
        result_path = os.path.join(args.out_dir, f"{eid}.json")
        if os.path.isfile(result_path):
            try:
                with open(result_path) as f:
                    cached = json.load(f)
                result = _episode_result_from_json(eid, tier, cached)
                all_results.append(result)
            except Exception as e:
                missing.append(eid)
                print(f"  WARNING: Failed to load {eid}: {e}")
        else:
            missing.append(eid)

    print(f"  Loaded: {len(all_results)} / {len(episodes)} episodes")
    if missing:
        print(f"  Missing: {len(missing)} episodes")
        for eid in missing[:10]:
            print(f"    {eid}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    # Aggregate
    report = aggregate_results(all_results, split_ids)

    # Save
    report_json = os.path.join(args.out_dir, f"report_{args.method_name}.json")
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report_md = os.path.join(args.out_dir, f"report_{args.method_name}.md")
    write_report(report, args.method_name, report_md)

    # Print summary using canonical metric names + coverage tracking
    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE: {args.method_name}")
    print(f"{'=' * 60}")

    for tier in ["all", "easy", "medium", "hard"]:
        r = report.get(tier, {})
        if not r:
            continue

        print(f"\n  {tier.upper()} ({r['num_episodes']} eps, {r['num_shots']} shots)")
        for metric in ["cs_face", "cs_object", "cs_transition_boundary",
                        "llm_face_accuracy", "llm_face_mean_score",
                        "llm_scene_accuracy", "llm_scene_mean_score",
                        "llm_object_accuracy", "llm_object_mean_score"]:
            s = r.get(metric, {})
            if s.get("value", "missing") is None:
                print(f"    {metric:25s}: --- "
                      f"(0/{s.get('n_episodes_total', '?')} episodes)")
            elif "mean" in s:
                cov = (f"{s.get('n_episodes_evaluated','?')}/"
                       f"{s.get('n_episodes_total','?')}")
                print(f"    {metric:25s}: {s['mean']:.4f} ± "
                      f"{s.get('std', 0):.4f}  ({cov})")

    print(f"\n  {report_json}")
    print(f"  {report_md}")


def main():
    parser = argparse.ArgumentParser(
        description="Distribute evaluation across multiple GPUs"
    )
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--scripts_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--pillars", default="1,2,3")
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--vlm_model", default="gemini-2.5-pro")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vbench_cache_dir", default="")
    parser.add_argument("--llm_concurrency", type=int, default=5,
                        help="Concurrent LLM calls per shard "
                             "(default 5, matching n_keys)")
    parser.add_argument("--save_action_grids", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--shard_id", type=int, default=-1,
                        help="Run only this shard (-1 = all shards)")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip eval, just aggregate existing results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Aggregate-only mode ----
    if args.aggregate_only:
        run_aggregation(args)
        return

    # ---- Load and shard episodes ----
    episodes = load_episodes(args.split_json)
    shards = shard_episodes(episodes, args.n_gpus)

    print(f"Distributed evaluation: {args.method_name}")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  GPUs: {args.n_gpus}")
    print(f"  Pillars: {args.pillars}")
    print()

    for i, shard in enumerate(shards):
        tier_counts = {}
        for _, tier in shard:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        tier_str = ", ".join(f"{t}={c}" for t, c in sorted(tier_counts.items()))
        print(f"  GPU {i}: {len(shard):3d} episodes ({tier_str})")
    print()

    # ---- Write per-shard split files ----
    shard_dir = os.path.join(args.out_dir, ".shards")
    os.makedirs(shard_dir, exist_ok=True)

    shard_splits = []
    for i, shard in enumerate(shards):
        # Build a mini split_ids.json for this shard
        shard_split = {"easy": [], "medium": [], "hard": [], "all": []}
        for eid, tier in shard:
            shard_split[tier].append(eid)
            shard_split["all"].append(eid)

        shard_path = os.path.join(shard_dir, f"shard_{i}.json")
        with open(shard_path, "w") as f:
            json.dump(shard_split, f, indent=2)
        shard_splits.append(shard_path)

    # ---- Build and run commands ----
    commands = []
    for i, (shard, shard_path) in enumerate(zip(shards, shard_splits)):
        if not shard:
            continue
        gpu_id = i  # GPU 0..N-1
        cmd = build_shard_command(i, gpu_id, shard, args, shard_path)
        commands.append((i, gpu_id, cmd, len(shard)))

    # Filter to single shard if requested
    if args.shard_id >= 0:
        commands = [(i, g, c, n) for i, g, c, n in commands if i == args.shard_id]
        if not commands:
            print(f"ERROR: shard_id {args.shard_id} not found")
            return

    if args.dry_run:
        print("DRY RUN — commands that would be executed:\n")
        for i, gpu_id, cmd, n_eps in commands:
            print(f"# Shard {i} on GPU {gpu_id} ({n_eps} episodes)")
            print(cmd)
            print()
        print("# After all shards complete:")
        print(f"python {__file__} --aggregate_only "
              f"--split_json {args.split_json} "
              f"--out_dir {args.out_dir} "
              f"--method_name {args.method_name} "
              f"--n_gpus {args.n_gpus} "
              f"--results_dir {args.results_dir} "
              f"--scripts_dir {args.scripts_dir}")
        return

    # Launch all shards in parallel
    print(f"Launching {len(commands)} processes...\n")
    log_dir = os.path.join(args.out_dir, ".logs")
    os.makedirs(log_dir, exist_ok=True)

    processes = []
    for i, gpu_id, cmd, n_eps in commands:
        log_path = os.path.join(log_dir, f"shard_{i}.log")
        print(f"  Shard {i} → GPU {gpu_id} ({n_eps} eps) → {log_path}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Rewrite cuda:N → cuda:0 since CUDA_VISIBLE_DEVICES remaps
        cmd_local = cmd.replace(f"--device cuda:{gpu_id}", "--device cuda:0")

        with open(log_path, "w") as log_f:
            log_f.write(f"# Shard {i}, GPU {gpu_id}, {n_eps} episodes\n")
            log_f.write(f"# {cmd_local}\n\n")
            log_f.flush()

            proc = subprocess.Popen(
                cmd_local,
                shell=True,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            processes.append((i, proc, log_path))

    # Wait for all to finish
    print(f"\n  Waiting for {len(processes)} processes...")
    start_time = time.time()

    results = []
    for i, proc, log_path in processes:
        proc.wait()
        elapsed = time.time() - start_time
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  Shard {i}: {status} ({elapsed/60:.1f} min elapsed)")
        results.append((i, proc.returncode))

    # Check for failures
    failed = [i for i, rc in results if rc != 0]
    if failed:
        print(f"\nWARNING: {len(failed)} shards failed: {failed}")
        print("Check logs:")
        for i in failed:
            print(f"  cat {os.path.join(log_dir, f'shard_{i}.log')}")
        print("\nTo retry a failed shard:")
        print(f"  python {__file__} --shard_id <N> [same args...]")
    else:
        print(f"\nAll {len(processes)} shards completed successfully!")

    # Run aggregation
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} min")
    run_aggregation(args)


if __name__ == "__main__":
    main()