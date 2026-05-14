#!/usr/bin/env python3
"""
evaluate_benchmark.py

Evaluation system for the Cross-Shot Consistency Benchmark.

STRICT MODE (default for benchmark runs):
  - All configuration is locked to BENCHMARK_CONFIG below. CLI overrides
    are rejected unless --allow_config_override is passed.
  - Missing infrastructure (VBench not installed, models unavailable,
    videos unreadable) raises RuntimeError. No silent fallbacks.
  - Per-item failures (one entity not grounded, one VLM call rejected)
    are recorded as null in per-episode JSON, NEVER counted as 0 in
    the aggregate. n_evaluated and n_failed are tracked separately
    from n_total so reproducibility can be audited.
  - A run_manifest.json is written alongside results recording every
    config knob, model checkpoint hash, and library version. Two runs
    are comparable IFF their manifests match.

Three evaluation pillars:
  Pillar 1: Intra-shot quality (VBench 7 custom-video dimensions)
    - subject_consistency, background_consistency, temporal_flickering,
      motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality
    - Requires: pip install vbench. NO fallback in strict mode.

  Pillar 2: Inter-shot consistency (OUR CONTRIBUTION)
    Unified pipeline: GroundingDINO → CLIP match → DINOv2 compare + Gemini judge
    Metrics:
      CS-Face / CS-Object: DINOv2 crop similarity (first-vs-each)
      VLM-Face / VLM-Scene / VLM-Object: Gemini judge (first-vs-each)
      CS-Style: DINOv2 Gram matrix similarity (all pairs)
      VLM-Style: Gemini style judge (sampled pairs)
      CS-Transition: DINOv2 boundary frame similarity (MI2V pairs only)
    Gap decay: similarity vs recurrence gap for paper figure.

  Aggregation runs automatically after evaluation:
    Per-episode JSON, per-tier stats, markdown report, run manifest.

Usage:
  # Canonical benchmark run (strict mode, locked config)
  python evaluate_benchmark.py \
    --results_dir /path/to/generated_outputs \
    --scripts_dir data/scripts \
    --split_json  data/splits/final_split_validated_ids.json \
    --out_dir     eval_results/<method_name> \
    --method_name <method_name> \
    --pillars 1,2,3 --resume

  # Iteration mode (allows config tweaks, no manifest enforcement)
  python evaluate_benchmark.py ... --allow_config_override --n_frames 3

Dependencies (ALL required in strict mode, no fallbacks):
  pip install vbench                 # Pillar 1
  pip install groundingdino-py       # Pillar 2 grounding
  # transformers (CLIP + DINOv2)    # Pillar 2 matching + comparison
  # openai (Gemini VLM-Judge)       # Pillar 2 VLM judging
"""

import argparse
import glob
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from PIL import Image

# tqdm is optional. If not installed, fall back to a no-op so the
# evaluator never depends on it. Progress bars are nice-to-have, not
# required for correctness.
try:
    from tqdm import tqdm as _tqdm
    def tqdm(it, **kw):
        return _tqdm(it, **kw)
except ImportError:
    def tqdm(it, **kw):
        return it

logger = logging.getLogger(__name__)


# ============================================================
# Strict-mode infrastructure
# ============================================================

class StrictModeError(RuntimeError):
    """Raised in strict mode when something would have silently degraded.

    Catching this is fine for diagnostic tools, but the main eval loop
    should let it propagate so the user sees a hard failure rather than
    a quietly-broken benchmark number.
    """


# Locked configuration. Any deviation from these values produces results
# that are NOT comparable to canonical benchmark numbers. In strict mode
# (--strict, default ON), CLI cannot override these. The values are
# embedded in the run manifest so two runs can be byte-checked for
# comparability.
BENCHMARK_CONFIG = {
    # Pillar 2 grounding
    "gdino_box_threshold": 0.25,
    "gdino_text_threshold": 0.20,
    "gdino_query_max_words": 12,
    # Pillar 2 matching
    "clip_match_threshold": 0.20,
    # Pillar 2 frame sampling
    "n_frames_per_shot": 5,
    "n_boundary_frames": 3,
    "n_scene_frames_per_shot": 2,   # for VLM scene judge
    "n_style_pairs_per_episode": 5,  # for VLM style judge
    # Pillar 2 entity crop padding
    "crop_pad_ratio": 0.10,
    # VLM judge — all use gemini-2.5-pro for the canonical benchmark
    # (pro is more reliable than flash; the benchmark must use one model
    # consistently across entity / scene / style / action).
    "vlm_model_entity": "gemini-2.5-pro",
    "vlm_model_scene": "gemini-2.5-pro",
    "vlm_model_style": "gemini-2.5-pro",
    "vlm_model_action": "gemini-2.5-pro",
    "vlm_model_intra_fidelity": "gemini-2.5-pro",
    # Pillar 2 cross-shot VLM-Judge — N-at-once scoring.
    # Gemini-2.5-pro reliably handles up to ~16 images per call. For
    # entities that recur in more than 16 shots (rare; mostly the hard
    # tier), we down-sample evenly across the recurrence to preserve
    # temporal spread. Locked at 16 in BENCHMARK_CONFIG so the metric
    # is comparable across runs regardless of cluster image-input limits.
    "vlm_n_max_images_per_call": 16,
    "vlm_max_tokens": 50000,
    # Pillar 2 cross-shot fidelity gate.
    # Only count a (shot, entity) appearance toward cross-shot consistency
    # if the intra-shot fidelity Gemini score for that appearance was at
    # least this threshold. The motivation: a model that renders the WRONG
    # location/object/character in a shot (or renders nothing recognizable)
    # should not be rewarded with high "consistency" if those wrong
    # renderings happen to look similar across shots — a known failure mode
    # is static shots that all look identical because the model froze.
    # Set to 0 to disable gating (counts every grounded appearance, the
    # pre-fidelity-gate behavior). Requires Pillar 3 to have run; if Pillar 3
    # didn't run, no gating is applied (gate is silently disabled).
    "cross_shot_fidelity_gate_threshold": 0.5,
    # Models (fixed for the benchmark)
    "dino_model": "facebook/dinov2-base",
    "clip_model": "openai/clip-vit-base-patch32",
}


def build_run_manifest(method_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a manifest recording everything that affects results.

    Two runs are comparable IFF their manifests are identical (modulo
    method_name and timestamp). Includes library versions, model
    checkpoint paths + sizes, and full config.
    """
    import hashlib
    import platform
    import sys

    manifest = {
        "method_name": method_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": dict(config),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {},
        "model_checkpoints": {},
    }

    # Library versions
    for libname in ["torch", "transformers", "numpy", "PIL", "vbench",
                    "groundingdino", "openai", "imageio"]:
        try:
            mod = __import__(libname)
            manifest["libraries"][libname] = getattr(mod, "__version__", "unknown")
        except Exception:
            manifest["libraries"][libname] = "NOT_INSTALLED"

    # Model checkpoint fingerprints (path + size + first-1MB sha256)
    checkpoint_env_vars = [
        "GDINO_CONFIG", "GDINO_WEIGHTS",
        "DINO_MODEL_PATH", "CLIP_MODEL_PATH",
        "VBENCH_CACHE_DIR",
    ]
    for env in checkpoint_env_vars:
        path = os.environ.get(env, "")
        if not path:
            manifest["model_checkpoints"][env] = "UNSET"
            continue
        if not os.path.exists(path):
            manifest["model_checkpoints"][env] = f"MISSING ({path})"
            continue
        if os.path.isfile(path):
            size = os.path.getsize(path)
            try:
                with open(path, "rb") as f:
                    head = f.read(1024 * 1024)
                fingerprint = hashlib.sha256(head).hexdigest()[:16]
            except Exception:
                fingerprint = "unreadable"
            manifest["model_checkpoints"][env] = {
                "path": path, "size": size, "head_sha256": fingerprint,
            }
        else:
            manifest["model_checkpoints"][env] = {"path": path, "type": "directory"}

    return manifest


def manifest_comparable(m1: Dict[str, Any], m2: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check whether two manifests describe comparable runs.

    Returns (is_comparable, list_of_differences).

    Excluded from check (these are runtime metadata, not result-determining):
      - method_name, timestamp_utc, platform
      - n_vlm_keys / n_llm_keys (number of API keys affects throughput, not values)
      - vlm_call_stats / llm_call_stats (per-run statistics, not config)

    Library version drift: we compare library versions, but minor version
    drift within the same major version is NOT a hard failure. Only major
    version changes break comparability. The rationale: between e.g.
    `transformers==4.33.2` and `transformers==4.48.3`, the inference
    paths for CLIP and DINOv2 have not meaningfully changed; the score
    drift is well below noise. A jump from `transformers==4.x` to `5.x`
    would be flagged.
    """
    differences = []
    ignore = {
        "method_name", "timestamp_utc", "platform",
        "n_vlm_keys", "n_llm_keys",
        "vlm_call_stats", "llm_call_stats",
    }
    keys = set(m1.keys()) | set(m2.keys())
    for k in sorted(keys - ignore):
        v1, v2 = m1.get(k), m2.get(k)
        if v1 == v2:
            continue
        # Special handling for the libraries dict: compare library-by-library,
        # tolerate minor version drift.
        if k == "libraries" and isinstance(v1, dict) and isinstance(v2, dict):
            lib_diffs = _compare_library_versions(v1, v2)
            differences.extend(lib_diffs)
            continue
        differences.append(f"{k}: {v1!r} vs {v2!r}")
    return (len(differences) == 0), differences


def _compare_library_versions(libs1: Dict[str, str], libs2: Dict[str, str]) -> List[str]:
    """Compare two library version dicts, only flagging major-version changes.

    Returns list of diff strings (empty if all libraries are major-compatible).
    Unknown / NOT_INSTALLED versions are treated as identical to themselves
    but flagged if one side has a real version and the other doesn't.
    """
    diffs = []
    libs = set(libs1) | set(libs2)
    for lib in sorted(libs):
        v1 = libs1.get(lib, "NOT_INSTALLED")
        v2 = libs2.get(lib, "NOT_INSTALLED")
        if v1 == v2:
            continue
        # Try to extract major version; if either is non-semver, fall back
        # to exact match (and complain).
        major1 = _major_version(v1)
        major2 = _major_version(v2)
        if major1 is None or major2 is None:
            diffs.append(f"libraries.{lib}: {v1!r} vs {v2!r} (unparseable, treating as different)")
        elif major1 != major2:
            diffs.append(f"libraries.{lib}: {v1!r} vs {v2!r} (major version change)")
        # else: minor or patch drift — tolerate
    return diffs


def _major_version(v: str) -> Optional[int]:
    """Parse the major version number from a version string. Returns None if unparseable.

    Handles: '4.33.2', '4.48.3+cu124', '2.5.1', etc.
    Treats 'unknown', 'NOT_INSTALLED', '' as not a real version (returns None).
    """
    if not v or v in ("unknown", "NOT_INSTALLED"):
        return None
    try:
        # Strip everything after first '+' or '-' (e.g. '+cu124', '-rc1')
        clean = v.split("+")[0].split("-")[0]
        first = clean.split(".")[0]
        return int(first)
    except (ValueError, IndexError):
        return None


# ============================================================
# Data structures
# ============================================================

@dataclass
class MetricValue:
    """A metric with explicit success/failure accounting.

    `value` is None if the metric could not be computed (e.g., no entities
    of this type, no shots in episode, all VLM calls failed). It is NEVER
    set to 0.0 as a fallback. n_evaluated counts pairs/items that
    contributed; n_failed counts pairs/items that were attempted but
    failed (e.g., VLM API rejection, frame index out of range).
    n_total = n_evaluated + n_failed + n_skipped, where n_skipped covers
    items that legitimately had no comparison to make (e.g., entity
    appears in only one shot).
    """
    value: Optional[float] = None
    n_evaluated: int = 0
    n_failed: int = 0
    n_skipped: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "n_evaluated": self.n_evaluated,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
        }

    @classmethod
    def from_json(cls, d: Any) -> "MetricValue":
        if d is None:
            return cls()
        if isinstance(d, (int, float)):
            # Legacy format — single number with no provenance.
            return cls(value=float(d))
        return cls(
            value=d.get("value"),
            n_evaluated=int(d.get("n_evaluated", 0)),
            n_failed=int(d.get("n_failed", 0)),
            n_skipped=int(d.get("n_skipped", 0)),
        )


@dataclass
class EntityCrop:
    """A grounded entity crop from a specific frame."""
    entity_name: str
    entity_type: str  # "character", "location", "object"
    entity_description: str
    shot_key: str  # "scene:shot" e.g. "1:2"
    frame_idx: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    crop_path: str
    clip_text_sim: float = 0.0
    dino_embedding: Optional[np.ndarray] = None


@dataclass
class ShotResult:
    """Evaluation results for one shot."""
    episode_id: str
    shot_key: str
    video_path: str
    vbench_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    entity_crops: List[EntityCrop] = field(default_factory=list)
    transition_score: Optional[float] = None


@dataclass
class EpisodeResult:
    """Aggregated evaluation for one episode.

    Each metric is a MetricValue with explicit None-vs-0 distinction.
    Legacy float access (e.g., `r.cs_face`) returns the .value via
    __getattr__ shim — but new code should use `r.metrics["cs_face"].value`.
    """
    episode_id: str
    tier: str
    num_shots: int
    shot_results: List[ShotResult] = field(default_factory=list)
    # Pillar 1 — per-VBench-dim, with provenance
    vbench_means: Dict[str, MetricValue] = field(default_factory=dict)
    # Pillar 2 — DINOv2 metrics
    metrics: Dict[str, MetricValue] = field(default_factory=dict)
    # Detail
    per_entity_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    gap_decay_data: List[Dict[str, float]] = field(default_factory=list)

    def __getattr__(self, name):
        # Backward-compat shim: r.cs_face → r.metrics["cs_face"].value or None
        if name in _LEGACY_METRIC_NAMES:
            mv = self.metrics.get(name)
            return mv.value if mv else None
        raise AttributeError(name)


# Names that the old API exposed as flat float attributes on EpisodeResult.
# Kept for read-only backward compat via __getattr__; writes must go
# through r.metrics[name] = MetricValue(...). Updated to include all
# current metric names so any old-style getattr probes return None
# rather than raising.
_LEGACY_METRIC_NAMES = {
    "cs_face", "cs_object", "cs_transition_boundary",
    "llm_face_accuracy", "llm_face_mean_score",
    "llm_scene_accuracy", "llm_scene_mean_score",
    "llm_object_accuracy", "llm_object_mean_score",
}


# ============================================================
# Frame extraction
# ============================================================

def extract_frames(video_path: str, n_frames: int = 5, strict: bool = True) -> List[np.ndarray]:
    """Extract n_frames uniformly from a video.

    In strict mode (default), raises StrictModeError if:
      - the video cannot be opened
      - the video has zero readable frames
      - the video has fewer frames than n_frames (would silently shrink the sample)

    In non-strict mode, returns whatever frames could be read (possibly empty).
    """
    try:
        import imageio.v2 as iio
    except ImportError:
        import imageio as iio

    if not os.path.isfile(video_path):
        if strict:
            raise StrictModeError(f"Video file not found: {video_path}")
        return []

    try:
        reader = iio.get_reader(video_path)
    except Exception as e:
        if strict:
            raise StrictModeError(f"Cannot open video {video_path}: {e}")
        logger.warning(f"Cannot read video: {video_path}")
        return []

    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    except Exception as e:
        if strict:
            raise StrictModeError(f"Failed reading frames from {video_path}: {e}")
    finally:
        try:
            reader.close()
        except Exception:
            pass

    if not frames:
        if strict:
            raise StrictModeError(f"Video has zero readable frames: {video_path}")
        return []

    n = len(frames)
    if n < n_frames:
        if strict:
            raise StrictModeError(
                f"Video {video_path} has {n} frames, expected >= {n_frames}. "
                f"Sampling fewer frames would make this episode incomparable to others."
            )
        return frames

    # Uniform sampling, skipping the very first/last frame (often duplicate/black)
    start = 1
    end = n - 2
    step = max(1, (end - start) / max(1, n_frames - 1))
    indices = [min(int(start + i * step), n - 1) for i in range(n_frames)]
    return [frames[i] for i in indices]


def extract_boundary_frames(
    video_path: str, n: int = 3, which: str = "last", strict: bool = True,
) -> List[np.ndarray]:
    """Extract the first or last n frames from a video.

    In strict mode, raises if the video can't be read or has fewer than n frames.
    """
    try:
        import imageio.v2 as iio
    except ImportError:
        import imageio as iio

    if not os.path.isfile(video_path):
        if strict:
            raise StrictModeError(f"Video file not found: {video_path}")
        return []

    frames = []
    try:
        reader = iio.get_reader(video_path)
        for frame in reader:
            frames.append(frame)
        reader.close()
    except Exception as e:
        if strict:
            raise StrictModeError(f"Cannot read boundary frames from {video_path}: {e}")
        return []

    if not frames:
        if strict:
            raise StrictModeError(f"Video has zero frames: {video_path}")
        return []

    if len(frames) < n:
        if strict:
            raise StrictModeError(
                f"Video {video_path} has {len(frames)} frames, "
                f"need {n} boundary frames."
            )
        return frames if which == "last" else frames

    if which == "last":
        return frames[-n:]
    return frames[:n]


# ============================================================
# Model loaders (lazy, singleton)
# ============================================================

_MODELS = {}


def get_dino_model(device: str = "cuda"):
    """Load DINOv2 for visual similarity.

    Loads from the local path in $DINO_MODEL_PATH if set; otherwise
    falls back to downloading "facebook/dinov2-base" from HuggingFace.
    Local path is preferred on the cluster — avoids re-downloading on
    every fresh worker and keeps the model checkpoint pinned in the
    manifest.
    """
    if "dino" not in _MODELS:
        from transformers import AutoModel, AutoImageProcessor
        local = os.environ.get("DINO_MODEL_PATH", "").strip()
        if local and os.path.isdir(local):
            model_src = local
            logger.info(f"[Eval] Loading DINOv2 from local path: {local}")
        else:
            if local:
                logger.warning(
                    f"[Eval] DINO_MODEL_PATH={local!r} is set but not a "
                    f"directory; falling back to HuggingFace download"
                )
            model_src = "facebook/dinov2-base"
            logger.info(f"[Eval] Loading DINOv2 from HuggingFace: {model_src}")
        processor = AutoImageProcessor.from_pretrained(model_src)
        model = AutoModel.from_pretrained(model_src).to(device).eval()
        _MODELS["dino"] = (model, processor, device)
        logger.info(f"[Eval] DINOv2 ready on {device}")
    return _MODELS["dino"]


def get_clip_model(device: str = "cuda"):
    """Load CLIP for text-image matching.

    Loads from the local path in $CLIP_MODEL_PATH if set; otherwise
    falls back to downloading "openai/clip-vit-base-patch32" from
    HuggingFace. See get_dino_model for the rationale.
    """
    if "clip" not in _MODELS:
        from transformers import CLIPModel, CLIPProcessor
        local = os.environ.get("CLIP_MODEL_PATH", "").strip()
        if local and os.path.isdir(local):
            model_src = local
            logger.info(f"[Eval] Loading CLIP from local path: {local}")
        else:
            if local:
                logger.warning(
                    f"[Eval] CLIP_MODEL_PATH={local!r} is set but not a "
                    f"directory; falling back to HuggingFace download"
                )
            model_src = "openai/clip-vit-base-patch32"
            logger.info(f"[Eval] Loading CLIP from HuggingFace: {model_src}")
        processor = CLIPProcessor.from_pretrained(model_src)
        model = CLIPModel.from_pretrained(model_src).to(device).eval()
        _MODELS["clip"] = (model, processor, device)
        logger.info(f"[Eval] CLIP ready on {device}")
    return _MODELS["clip"]


_DEFAULT_CACHE = Path(os.path.expanduser("~/.cache/entitybench"))

_GDINO_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
_GDINO_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
def _download(url: str, dest: Path) -> Path:
    """Download `url` to `dest` if it doesn't exist. Returns dest."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    logger.info(f"[Eval] Downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def get_grounding_dino(device: str = "cuda", strict: bool = True):
    """Load GroundingDINO. Uses $GDINO_CONFIG / $GDINO_WEIGHTS if set,
    otherwise downloads the standard SwinT_OGC weights+config to
    ~/.cache/entitybench/groundingdino/ on first use."""
    if "gdino" not in _MODELS:
        try:
            from groundingdino.util.inference import load_model
            cfg = os.environ.get("GDINO_CONFIG", "").strip()
            wts = os.environ.get("GDINO_WEIGHTS", "").strip()
            if not (cfg and os.path.exists(cfg)):
                cfg = str(_download(
                    _GDINO_CONFIG_URL,
                    _DEFAULT_CACHE / "groundingdino" / "GroundingDINO_SwinT_OGC.py",
                ))
            if not (wts and os.path.exists(wts)):
                wts = str(_download(
                    _GDINO_WEIGHTS_URL,
                    _DEFAULT_CACHE / "groundingdino" / "groundingdino_swint_ogc.pth",
                ))
            model = load_model(cfg, wts, device=device)
            _MODELS["gdino"] = (model, device)
            logger.info("[Eval] Loaded GroundingDINO")
        except Exception as e:
            if strict:
                raise StrictModeError(
                    f"GroundingDINO unavailable: {e}. With network access "
                    f"this should auto-download; otherwise point "
                    f"GDINO_CONFIG and GDINO_WEIGHTS at local paths."
                )
            logger.warning(f"[Eval] GroundingDINO not available: {e}")
            _MODELS["gdino"] = None
    return _MODELS["gdino"]


# ============================================================
# Pillar 1: VBench intra-shot evaluation (7 custom-video dims)
# ============================================================

VBENCH_DIMENSIONS = [
    # NOTE: `background_consistency` was dropped from the suite — it
    # measures within-shot CLIP cosine on consecutive frames, which is
    # confounded by camera motion (a pan or zoom of a stable background
    # gets penalized as inconsistency). For our long-range multi-shot
    # benchmark where camera motion is intentional, the metric was
    # systematically misleading.
    "subject_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

# Per-dim score scale. Used in the report writer to label each row
# correctly. We do NOT normalize anything here — VBench's canonical
# scales are what every other VBench paper reports, so dividing
# imaging_quality by 100 (or any post-processing) would silently break
# comparability with the broader literature. Keep the raw scales,
# document them, move on.
VBENCH_DIM_SCALE = {
    "subject_consistency":    "[0, 1]",
    "temporal_flickering":    "[0, 1]",
    "motion_smoothness":      "[0, 1]",
    "dynamic_degree":         "{0, 1}",   # binary per-shot fraction
    "aesthetic_quality":      "[0, 1]",
    "imaging_quality":        "[0, 100]",  # MUSIQ raw scale — NOT normalized
}

_VBENCH_DIM_DEPS = {
    "subject_consistency":    "DINO",
    "temporal_flickering":    "pixel diff (no model)",
    "motion_smoothness":      "RAFT",
    "dynamic_degree":         "RAFT",
    "aesthetic_quality":      "LAION aesthetic (CLIP MLP)",
    "imaging_quality":        "MUSIQ (IQA-PyTorch)",
}


def _check_vbench_available() -> Tuple[bool, Optional[str]]:
    """Check if VBench is importable."""
    try:
        import vbench
        ver = getattr(vbench, "__version__", "unknown")
        return True, ver
    except ImportError:
        return False, None


class VBenchEvaluator:
    """
    Wrapper around VBench for the 7 custom-video dimensions.

    STRICT MODE (default): every dimension MUST succeed for every shot.
    A failure on any dim aborts the episode (or the whole eval, depending
    on --strict / --allow_partial). NO fallback to DINOv2/CLIP — those
    metrics are not equivalent and mixing them in the aggregate corrupts
    cross-method comparisons.

    Dimension failures that occur on episode N are NOT silently propagated
    to episodes N+1..end. Each episode's VBench evaluation is independent
    and audited.
    """

    def __init__(self, device: str = "cuda", cache_dir: str = "",
                 output_dir: str = "", strict: bool = True):
        self.device = device
        self._cache_dir = cache_dir or os.path.expanduser("~/.cache/vbench")
        self._output_dir = output_dir
        self._vbench = None
        self._available = None
        self.strict = strict
        # Per-episode tracking — does NOT persist across episodes in strict mode.
        # In non-strict mode, kept across episodes for backward compat.
        self._failed_dims_this_episode: Set[str] = set()

    def _ensure_loaded(self, output_dir: str = "") -> bool:
        if self._available is not None:
            return self._available

        ok, ver = _check_vbench_available()
        if not ok:
            if self.strict:
                raise StrictModeError(
                    "VBench is not installed. Run: pip install vbench. "
                    "Strict mode does NOT permit fallback to DINOv2/CLIP "
                    "approximations because the resulting numbers would not "
                    "be comparable to canonical benchmark scores."
                )
            logger.warning("[VBench] Not installed.")
            self._available = False
            return False

        try:
            from vbench import VBench
            out = output_dir or self._output_dir or os.path.join(self._cache_dir, "eval_output")
            os.makedirs(out, exist_ok=True)
            logger.info(f"[VBench] Initializing: cache_dir={self._cache_dir}, output_dir={out}")
            self._vbench = VBench(self.device, self._cache_dir, out)
            self._available = True
            logger.info(f"[VBench] Loaded v{ver} on {self.device}")
        except Exception as e:
            if self.strict:
                raise StrictModeError(f"VBench init failed: {e}")
            logger.warning(f"[VBench] Init failed: {e}")
            self._available = False

        return self._available

    def evaluate_episode_shots(
        self,
        shot_videos: Dict[str, str],
        episode_id: str,
        work_dir: str,
    ) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
        """Evaluate all shots across the 7 VBench dims.

        Returns:
            (results, failed_dims):
                results: {shot_key: {dim: score_or_None}} — None means the
                    dim genuinely could not be computed for this shot.
                failed_dims: list of dimensions that failed at the
                    evaluator level (apply to all shots in this episode).

        In strict mode, raises StrictModeError if any dim fails.
        """
        # Reset per-episode failure set so episodes don't poison each other.
        self._failed_dims_this_episode = set()

        if not self._ensure_loaded(output_dir=work_dir):
            if self.strict:
                # _ensure_loaded would have raised; defensive
                raise StrictModeError("VBench not loaded")
            # Non-strict: nothing computable
            return ({sk: {d: None for d in VBENCH_DIMENSIONS} for sk in shot_videos},
                    list(VBENCH_DIMENSIONS))

        results: Dict[str, Dict[str, Optional[float]]] = {
            sk: {d: None for d in VBENCH_DIMENSIONS} for sk in shot_videos
        }

        results = self._run_vbench_batch(
            shot_videos, list(VBENCH_DIMENSIONS),
            episode_id, work_dir, results,
        )

        if self._failed_dims_this_episode:
            msg = (f"VBench dims failed on episode {episode_id}: "
                   f"{sorted(self._failed_dims_this_episode)}")
            if self.strict:
                raise StrictModeError(
                    msg + ". Strict mode requires all 7 dims to succeed. "
                    "Run --diagnose_vbench to check checkpoints, or pass "
                    "--allow_partial to record these as null."
                )
            logger.warning(msg)

        # Validate every shot has every dim (in strict mode)
        if self.strict:
            for sk, dims in results.items():
                for d in VBENCH_DIMENSIONS:
                    if dims.get(d) is None:
                        raise StrictModeError(
                            f"Episode {episode_id}, shot {sk}, dim {d}: "
                            f"no score returned by VBench."
                        )

        return results, sorted(self._failed_dims_this_episode)

    def _run_vbench_batch(self, shot_videos, dimensions, episode_id, work_dir, results):
        """Run VBench batch evaluation via symlinked video dir."""
        import shutil

        batch_dir = os.path.join(work_dir, f"vbench_batch_{episode_id}")
        os.makedirs(batch_dir, exist_ok=True)

        shot_key_to_filename = {}
        for shot_key, vid_path in shot_videos.items():
            # Use zero-padded canonical naming (matches the generation
            # pipeline's "{scene:03d}_{shot:03d}.mp4" output and lets the
            # `_parse_output` matcher map VBench's video_path strings back
            # to our shot keys directly.
            try:
                sn_str, shot_str = shot_key.split(":")
                safe_name = f"{int(sn_str):03d}_{int(shot_str):03d}.mp4"
            except (ValueError, AttributeError):
                # Fallback for any non-standard shot keys
                safe_name = shot_key.replace(":", "_") + ".mp4"
            link_path = os.path.join(batch_dir, safe_name)
            if not os.path.exists(link_path):
                try:
                    os.symlink(os.path.abspath(vid_path), link_path)
                except OSError:
                    shutil.copy2(vid_path, link_path)
            shot_key_to_filename[shot_key] = safe_name

        output_dir = os.path.join(work_dir, f"vbench_out_{episode_id}")
        os.makedirs(output_dir, exist_ok=True)

        # VBench's per-call `output_path=` argument is honored inconsistently
        # across versions — older versions write to the `output_path`
        # passed to VBench(__init__), ignoring per-call overrides. So we
        # search BOTH candidate locations for the result file.
        candidate_dirs = [output_dir]
        if self._vbench is not None:
            init_out = getattr(self._vbench, "output_path", None) \
                or self._output_dir or self._cache_dir
            if init_out and init_out not in candidate_dirs:
                candidate_dirs.append(init_out)

        for dim in dimensions:
            run_name = f"{episode_id}_{dim}"
            try:
                self._vbench.evaluate(
                    videos_path=batch_dir,
                    name=run_name,
                    dimension_list=[dim],
                    mode="custom_input",
                    output_path=output_dir,
                )

                # VBench writes two files per dimension:
                #   {run_name}_eval_results.json   ← the scores we want
                #   {run_name}_full_info.json      ← per-frame metadata
                # Match BOTH the run_name prefix AND the dim string to
                # avoid picking up stale files from previous runs (the
                # work dir persists across episodes when --resume is on).
                vb_results = None
                matched_path = None
                for d in candidate_dirs:
                    if not os.path.isdir(d):
                        continue
                    for fname in os.listdir(d):
                        if (fname.startswith(run_name)
                                and "eval_results" in fname
                                and fname.endswith(".json")):
                            matched_path = os.path.join(d, fname)
                            try:
                                with open(matched_path, encoding="utf-8") as f:
                                    vb_results = json.load(f)
                            except Exception as e:
                                logger.warning(
                                    f"[VBench] {dim}: failed to read {matched_path}: {e}"
                                )
                                vb_results = None
                            break
                    if vb_results is not None:
                        break

                if vb_results is None:
                    self._failed_dims_this_episode.add(dim)
                    logger.warning(
                        f"[VBench] {dim} ran but no eval_results json found "
                        f"in {candidate_dirs} for {episode_id}"
                    )
                    continue

                # Try to parse and count how many shot scores landed
                before_n = sum(1 for sk in results
                               if results[sk].get(dim) is not None)
                self._parse_output(vb_results, dim, shot_key_to_filename, results)
                after_n = sum(1 for sk in results
                              if results[sk].get(dim) is not None)
                n_extracted = after_n - before_n

                if n_extracted == 0:
                    self._failed_dims_this_episode.add(dim)
                    # Surface a sample of the file to make debugging fast
                    sample = json.dumps(vb_results, default=str)[:300]
                    logger.warning(
                        f"[VBench] {dim} parsed file {matched_path} but "
                        f"extracted 0 shot scores. Sample: {sample}..."
                    )
                else:
                    logger.info(
                        f"[VBench] {dim}: extracted {n_extracted}/{len(shot_videos)} "
                        f"shot scores from {os.path.basename(matched_path)}"
                    )
            except Exception as e:
                dep = _VBENCH_DIM_DEPS.get(dim, "unknown")
                logger.warning(f"[VBench] {dim} failed (needs {dep}): {e}")
                self._failed_dims_this_episode.add(dim)

        shutil.rmtree(batch_dir, ignore_errors=True)
        return results

    def _parse_output(self, vb_results, dim, shot_key_to_filename, results):
        """Parse VBench output and map per-video scores back to shot keys.

        VBench's `..._eval_results.json` shape varies by version; the
        most common observed shape is:
          {dim_name: [episode_mean, [{"video_path": ..., "video_results": ...}, ...]]}
        but older / forked versions also produce flat lists or
        plain {video_name: score} maps. This parser unwraps the dim
        key, peels off a [mean, list] tuple if present, then walks
        the list extracting (video_path, video_results) per record.

        Side effect: writes scores into `results[shot_key][dim]`.
        Returns nothing — the caller diffs `results` to count how many
        scores were extracted (so an empty file produces 0, not a fake
        "parsed" status).
        """
        fn_to_sk = {v: k for k, v in shot_key_to_filename.items()}

        def _match(vname):
            if not vname:
                return None
            bn = os.path.basename(str(vname))
            if bn in fn_to_sk:
                return fn_to_sk[bn]
            stem = os.path.splitext(bn)[0]
            for fn, sk in fn_to_sk.items():
                if os.path.splitext(fn)[0] == stem:
                    return sk
            return None

        # Step 1: unwrap the {dim_name: ...} envelope if present
        payload = vb_results
        if isinstance(payload, dict) and dim in payload:
            payload = payload[dim]

        # Step 2: VBench commonly wraps results as
        #   [<episode_mean_float>, [<per_video_record>, ...]]
        # Peel off the mean to get the per-video list.
        if (isinstance(payload, list) and len(payload) == 2
                and isinstance(payload[0], (int, float))
                and isinstance(payload[1], list)):
            payload = payload[1]

        # Step 3: walk the per-video records
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    # Common keys: video_path / video_name for identifier,
                    # video_results / score / <dim> for value.
                    vn = (item.get("video_path")
                          or item.get("video_name")
                          or item.get("path", ""))
                    sc = item.get("video_results")
                    if sc is None:
                        sc = item.get("score")
                    if sc is None:
                        sc = item.get(dim)
                    sk = _match(vn)
                    if sk and sk in results and isinstance(sc, (int, float)):
                        results[sk][dim] = float(sc)
                elif isinstance(item, list) and len(item) >= 2:
                    # [path_or_name, score] tuple form
                    sk = _match(str(item[0]))
                    if sk and sk in results and isinstance(item[1], (int, float)):
                        results[sk][dim] = float(item[1])

        # Step 4: dict-of-videos form (legacy) — handled if step 3 didn't fire
        elif isinstance(payload, dict):
            for vn, sd in payload.items():
                sk = _match(vn)
                if not sk or sk not in results:
                    continue
                if isinstance(sd, (int, float)):
                    results[sk][dim] = float(sd)
                elif isinstance(sd, list) and sd and isinstance(sd[0], (int, float)):
                    results[sk][dim] = float(sd[0])
                elif isinstance(sd, dict):
                    s = sd.get("video_results", sd.get("score", sd.get(dim)))
                    if isinstance(s, (int, float)):
                        results[sk][dim] = float(s)

        # Step 5: scalar global score (extremely rare) — broadcast to all shots
        elif isinstance(payload, (int, float)):
            for sk in results:
                results[sk][dim] = float(payload)


# NOTE: _fallback_quality_metrics has been REMOVED from strict mode.
# Mixing DINOv2/CLIP approximations into VBench numbers makes results
# silently incomparable across runs. If VBench is not available in your
# environment, fix the environment — do not paper over it with proxies.


# ============================================================
# Pillar 2: Inter-shot consistency
# ============================================================

# ---- Step 1: Entity grounding ----

def _gdino_transform_frame(frame: np.ndarray):
    """Transform a numpy frame into the tensor format GroundingDINO expects."""
    import groundingdino.datasets.transforms as T
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_pil = Image.fromarray(frame).convert("RGB")
    img_transformed, _ = transform(img_pil, None)
    return img_transformed


def ground_one_entity_in_frame(
    frame: np.ndarray,
    entity_name: str,
    description: str,
    entity_type: str,
    device: str = "cuda",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    return_all: bool = False,
) -> List[Dict[str, Any]]:
    """Run GroundingDINO for ONE entity (description as the query).

    Unlike `ground_entities_in_frame` which expects entity_descriptions
    + entity_schedule + shot_key (and resolves the entity list from the
    schedule), this is a direct single-entity call: caller provides the
    entity name, description, and type.

    Returns a list of detection dicts, sorted by confidence descending.
    Empty list = GroundingDINO found nothing matching the query (this is
    a strong "absent" signal, not an error).

    Args:
        return_all: if True, return ALL boxes that cleared the thresholds
                    (for action annotation across frames). If False
                    (default), return at most the highest-confidence box.
    """
    gdino_result = get_grounding_dino(device, strict=False)
    if gdino_result is None:
        return []
    model, dev = gdino_result
    from groundingdino.util.inference import predict

    img_tensor = _gdino_transform_frame(frame)
    query = " ".join(description.split()[:BENCHMARK_CONFIG["gdino_query_max_words"]])

    try:
        boxes, logits, phrases = predict(
            model=model, image=img_tensor, caption=query,
            box_threshold=box_threshold, text_threshold=text_threshold,
            device=dev,
        )
    except Exception as e:
        logger.warning(f"[Ground] one-entity predict failed for {entity_name!r}: {e}")
        return []

    if len(boxes) == 0:
        return []

    h, w = frame.shape[:2]
    results = []
    # Convert each cxcywh-normalized box to xyxy-pixel
    for i in range(len(boxes)):
        box = boxes[i].tolist()
        x1 = max(0, int((box[0] - box[2]/2) * w))
        y1 = max(0, int((box[1] - box[3]/2) * h))
        x2 = min(w, int((box[0] + box[2]/2) * w))
        y2 = min(h, int((box[1] + box[3]/2) * h))
        results.append({
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "bbox": (x1, y1, x2, y2),
            "confidence": float(logits[i]),
        })
    # Sort by confidence, return as requested
    results.sort(key=lambda r: r["confidence"], reverse=True)
    if not return_all:
        return results[:1]
    return results


# ============================================================
# Crop quality scoring — used by the unified shot grounder to pick
# canonical crops that are not just semantically right (high CLIP)
# but also visually usable (sharp + large enough to evaluate).
# ============================================================

def _crop_sharpness(crop: np.ndarray) -> float:
    """Laplacian-variance-style sharpness score for a crop.

    Returns the variance of the second-order luminance derivatives,
    which is the standard no-reference sharpness proxy. Higher = sharper.
    Scale: typically 0–500 for natural images, 0–50 for blurry ones.
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = np.mean(crop, axis=2).astype(np.float64) if crop.ndim == 3 else crop.astype(np.float64)
    # Approximation of Laplacian: sum of horizontal + vertical second differences
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    dx2 = np.diff(gray, n=2, axis=1)
    dy2 = np.diff(gray, n=2, axis=0)
    return float(dx2.var() + dy2.var())


def _sigmoid(x: float, midpoint: float, scale: float) -> float:
    """Standard sigmoid centered at midpoint with given soft-scale.

    Returns ~0.0 for x << midpoint, ~1.0 for x >> midpoint, 0.5 at midpoint.
    """
    try:
        return float(1.0 / (1.0 + np.exp(-(x - midpoint) / max(1e-6, scale))))
    except (OverflowError, FloatingPointError):
        return 0.0 if x < midpoint else 1.0


def compute_selection_score(
    clip_score: float,
    crop: np.ndarray,
    frame_shape: Tuple[int, ...],
) -> Dict[str, float]:
    """Combined per-detection selection score: CLIP × sharpness × area.

    Each component:
      clip          — raw CLIP text-image similarity ∈ [0, ~0.5]
      sharpness     — sigmoid((laplacian_var - 100) / 200) ∈ [0, 1]
      area          — sigmoid((crop_area_pct - 2) / 5)     ∈ [0, 1]

    Sharpness sigmoid is centered at lap_var=100 because typical sharp
    natural-image crops score 200–400 while motion-blurred or low-detail
    crops score 30–80. Area sigmoid is centered at 2% to keep tiny crops
    (which are still counted as "present" for presence purposes) from
    winning the selection contest when a larger usable crop exists.

    The product means a crop must clear ALL three to score high:
      - high CLIP but blurry → loses on sharpness
      - sharp + large but wrong entity → loses on CLIP
      - right entity + sharp but tiny → loses on area
    This is what we want.
    """
    sharpness = _crop_sharpness(crop)
    area_pct = 100.0 * _bbox_area_ratio((0, 0, crop.shape[1], crop.shape[0]),
                                        frame_shape)
    sharp_norm = _sigmoid(sharpness, midpoint=100.0, scale=200.0)
    area_norm = _sigmoid(area_pct, midpoint=2.0, scale=5.0)
    selection = float(clip_score) * sharp_norm * area_norm
    return {
        "clip": float(clip_score),
        "sharpness_raw": sharpness,
        "sharpness_norm": sharp_norm,
        "area_pct": area_pct,
        "area_norm": area_norm,
        "selection": selection,
    }


def ground_entities_in_frame(
    frame: np.ndarray, entity_descriptions: Dict[str, Any],
    entity_schedule: Dict[str, Any], shot_key: str,
    device: str = "cuda", box_threshold: float = 0.25, text_threshold: float = 0.20,
) -> List[Dict[str, Any]]:
    """Ground entities using GroundingDINO with registry descriptions."""
    gdino_result = get_grounding_dino(device)
    if gdino_result is None:
        return []

    model, dev = gdino_result
    from groundingdino.util.inference import predict

    schedule = entity_schedule.get(shot_key, {})
    all_entities = (
        schedule.get("characters", []) + schedule.get("places", []) + schedule.get("objects", [])
    )

    if not all_entities:
        return []

    # Transform frame to tensor (GroundingDINO expects transformed tensor, NOT PIL)
    img_tensor = _gdino_transform_frame(frame)

    results = []
    for entity_name in all_entities:
        desc, etype = "", "object"
        for label, info in entity_descriptions.items():
            name = info.get("name", "") if isinstance(info, dict) else label
            if name == entity_name:
                desc = info.get("description", "") if isinstance(info, dict) else str(info)
                if label.startswith("Character"):
                    etype = "character"
                elif label.startswith("Location"):
                    etype = "location"
                break
        if not desc:
            logger.debug(f"[Ground] No description found for '{entity_name}'")
            continue

        query = " ".join(desc.split()[:12])
        try:
            boxes, logits, phrases = predict(
                model=model, image=img_tensor, caption=query,
                box_threshold=box_threshold, text_threshold=text_threshold, device=dev,
            )
            if len(boxes) > 0:
                best = logits.argmax().item()
                box = boxes[best].tolist()
                h, w = frame.shape[:2]
                x1 = max(0, int((box[0] - box[2]/2) * w))
                y1 = max(0, int((box[1] - box[3]/2) * h))
                x2 = min(w, int((box[0] + box[2]/2) * w))
                y2 = min(h, int((box[1] + box[3]/2) * h))
                results.append({
                    "entity_name": entity_name, "entity_type": etype,
                    "description": desc, "bbox": (x1, y1, x2, y2),
                    "confidence": float(logits[best]),
                })
            else:
                logger.debug(f"[Ground] No detection for '{entity_name}' (query: '{query}')")
        except Exception as e:
            logger.warning(f"[Ground] GroundingDINO failed for '{entity_name}': {e}")

    return results


def crop_entity(frame: np.ndarray, bbox: Tuple[int,int,int,int], pad_ratio: float = 0.1) -> np.ndarray:
    """Crop entity from frame using bbox with padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2-x1, y2-y1
    x1, y1 = max(0, x1 - int(bw*pad_ratio)), max(0, y1 - int(bh*pad_ratio))
    x2, y2 = min(w, x2 + int(bw*pad_ratio)), min(h, y2 + int(bh*pad_ratio))
    return frame[y1:y2, x1:x2].copy()


# ---- Step 2: CLIP text-image matching ----

def compute_clip_text_sim(crop: np.ndarray, text: str, device: str = "cuda") -> float:
    model, processor, dev = get_clip_model(device)
    import torch
    inputs = processor(text=[text], images=[Image.fromarray(crop)], return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        return float(model(**inputs).logits_per_image.item() / 100.0)


# ---- Step 3: DINOv2 embedding + comparison ----

def compute_dino_embedding(crop: np.ndarray, device: str = "cuda") -> np.ndarray:
    model, processor, dev = get_dino_model(device)
    import torch
    inputs = processor(images=Image.fromarray(crop), return_tensors="pt").to(dev)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0]
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().flatten()


def compute_dino_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))


# ---- CS-Face / CS-Object (DINOv2, anchor-vs-each) ----
#
# ---- CS-Face / CS-Object (DINOv2, centroid-based) ----
#
# Each entity's cross-shot consistency is measured against the centroid
# of all its appearances, NOT against a single anchor crop. Centroid
# similarity is the principled answer to "how clustered are this
# entity's renderings?" — no single crop has privileged status, so
# an outlier crop can't drag the score down just because it happened
# to be picked as anchor.
#
# Per-entity diagnostics:
#   mean_sim_to_centroid      ← typical render quality (headline)
#   min_sim_to_centroid       ← worst-case deviation
#   pairwise_median_sim       ← robustness cross-check
#   worst_shot                ← which shot has the worst render (auditable)
#   representative_shot       ← which shot is closest to centroid
#   representative_crop_path  ← saved path for the most-typical crop
#
# We also save the most-representative crop (closest to centroid) for
# audit purposes — it serves the same role the anchor used to play
# for downstream visual inspection, but without the metric depending
# on its choice.

def compute_cross_shot_consistency(
    entity_crops: Dict[str, List[EntityCrop]],
    entity_type_filter: str,
    shot_order: Optional[List[str]] = None,
    representative_save_dir: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Dict[str, Any]], int]:
    """Centroid-based cross-shot consistency for a given entity type.

    For each entity with N >= 2 appearances:
      1. Stack DINOv2 embeddings from all per-shot canonical crops
      2. Compute the centroid (re-normalized to unit length)
      3. Score each crop's cosine similarity to the centroid
      4. Report mean as the entity's consistency score; min as the
         worst-case deviation; identify which shot is the outlier

    The episode-level score is the mean of per-crop centroid similarities
    pooled across all entities of this type. (Equivalent to weighting
    each entity by its number of appearances — entities seen more often
    contribute more, which matches what we want: a character who appears
    in 8 shots matters more for episode-level consistency than one in 2.)

    Returns
    -------
    (episode_mean_or_None, per_entity_breakdown, n_crops_evaluated)

    Returns None for the mean when no entity has >=2 appearances, so the
    aggregator can distinguish "no data" from "all bad".
    """
    if representative_save_dir:
        os.makedirs(representative_save_dir, exist_ok=True)
    shot_pos = {sk: i for i, sk in enumerate(shot_order)} if shot_order else {}

    per_entity: Dict[str, Dict[str, Any]] = {}
    all_sims_to_centroid: List[float] = []

    for name, crops in entity_crops.items():
        typed = [c for c in crops if c.entity_type == entity_type_filter
                 and c.dino_embedding is not None]
        if len(typed) < 2:
            continue

        typed_sorted = sorted(typed, key=lambda c: shot_pos.get(c.shot_key, 0))

        # ---- Centroid similarity ----
        embs = np.stack([c.dino_embedding for c in typed_sorted])
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-9:
            # Degenerate: all embeddings cancel out (essentially never happens
            # with normalized DINO embeddings, but guard anyway). Skip entity.
            continue
        centroid = centroid / norm
        sims_to_centroid = embs @ centroid  # (N,) cosine similarities

        # ---- Pairwise median (robustness cross-check) ----
        # Compute all upper-triangle pairs once; the median gives us
        # a metric robust to a single outlier crop.
        n = len(typed_sorted)
        pair_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_sims.append(float(np.dot(embs[i], embs[j])))
        pair_median = float(np.median(pair_sims)) if pair_sims else None

        # ---- Locate the most-representative and worst-deviation shots ----
        rep_idx = int(np.argmax(sims_to_centroid))
        worst_idx = int(np.argmin(sims_to_centroid))
        rep_crop = typed_sorted[rep_idx]
        worst_crop = typed_sorted[worst_idx]

        # ---- Save the representative crop for audit ----
        rep_path = None
        if representative_save_dir and os.path.isfile(rep_crop.crop_path):
            try:
                safe = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
                shot_safe = rep_crop.shot_key.replace(":", "_")
                rep_path = os.path.join(
                    representative_save_dir,
                    f"{safe}__shot_{shot_safe}__sim{float(sims_to_centroid[rep_idx]):.2f}.png",
                )
                Image.open(rep_crop.crop_path).save(rep_path)
            except Exception as e:
                logger.warning(
                    f"[CS-{entity_type_filter}] {name}: failed to save "
                    f"representative crop: {e}"
                )

        per_entity[name] = {
            # Headline score: how tightly clustered is this entity?
            "mean_sim_to_centroid": float(np.mean(sims_to_centroid)),
            "min_sim_to_centroid": float(np.min(sims_to_centroid)),
            "max_sim_to_centroid": float(np.max(sims_to_centroid)),
            "pairwise_median_sim": pair_median,
            "n_appearances": n,
            "n_pairs": len(pair_sims),
            # Audit metadata
            "representative_shot": rep_crop.shot_key,
            "representative_clip_sim": float(rep_crop.clip_text_sim),
            "representative_crop_path": rep_path,
            "worst_shot": worst_crop.shot_key,
            "worst_clip_sim": float(worst_crop.clip_text_sim),
            "worst_sim_to_centroid": float(sims_to_centroid[worst_idx]),
            # Per-shot breakdown so reviewers can drill down
            "per_shot_sim_to_centroid": [
                {"shot": c.shot_key,
                 "sim_to_centroid": float(s),
                 "clip_text_sim": float(c.clip_text_sim)}
                for c, s in zip(typed_sorted, sims_to_centroid)
            ],
        }
        # Pool all per-crop sims for the episode-level mean
        all_sims_to_centroid.extend(float(s) for s in sims_to_centroid)

    return (
        (float(np.mean(all_sims_to_centroid)) if all_sims_to_centroid else None),
        per_entity,
        len(all_sims_to_centroid),
    )


# ---- CS-Style (DINOv2 Gram matrix) ----

def compute_style_consistency(frames: List[np.ndarray], device: str = "cuda") -> float:
    """DINOv2 Gram matrix similarity averaged over all frame pairs.

    REQUIRES len(frames) >= 2. Caller is responsible for the precondition;
    earlier code returned 1.0 (a fake perfect score) on len < 2 which
    silently inflated cs_style for episodes with single-shot tiers.
    """
    if len(frames) < 2:
        raise ValueError(
            f"compute_style_consistency requires >=2 frames, got {len(frames)}. "
            "Caller should record this as MetricValue(value=None, n_skipped=1) "
            "rather than calling with insufficient frames."
        )
    model, processor, dev = get_dino_model(device)
    import torch
    grams = []
    for frame in frames:
        inputs = processor(images=Image.fromarray(frame), return_tensors="pt").to(dev)
        with torch.no_grad():
            feat = model(**inputs).last_hidden_state[:, 1:].squeeze(0)
            gram = torch.mm(feat.T, feat) / feat.shape[0]
            grams.append((gram / gram.norm()).cpu().numpy().flatten())
    sims = [float(np.dot(grams[i], grams[j]))
            for i in range(len(grams)) for j in range(i+1, len(grams))]
    return float(np.mean(sims))


# ---- CS-Transition (DINOv2 boundary frames) ----

def compute_transition_score(
    video_path_prev: str, video_path_next: str,
    n_boundary: int = 3, device: str = "cuda",
) -> float:
    """DINOv2 cosine similarity averaged over (last-i, first-i) frame pairs.

    Raises StrictModeError on read failure (extract_boundary_frames is strict).
    Caller catches the error and counts the pair as n_failed rather than
    silently averaging a 0.0 into the transition mean.
    """
    last_f = extract_boundary_frames(video_path_prev, n=n_boundary, which="last")
    first_f = extract_boundary_frames(video_path_next, n=n_boundary, which="first")
    if not last_f or not first_f:
        raise StrictModeError(
            f"Boundary frames missing: prev={video_path_prev!r}, next={video_path_next!r}"
        )
    model, processor, dev = get_dino_model(device)
    import torch

    def _e(frame):
        inputs = processor(images=Image.fromarray(frame), return_tensors="pt").to(dev)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 0]
            return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().flatten()

    le, fe = [_e(f) for f in last_f], [_e(f) for f in first_f]
    sims = [float(np.dot(le[-(i+1)], fe[i])) for i in range(min(len(le), len(fe)))]
    return float(np.mean(sims))


# ---- VLM-Judge (Gemini) ----

def _vlm_api_call(images_b64: List[str], prompt: str, model: str) -> Optional[dict]:
    """Shared Gemini API call with key rotation.

    Uses the module-level _RotatingVLMClient (lazy-initialized from the
    GEMINI_API_KEYS env var on first call). Falls back to single-key
    behavior using AZURE_OPENAI_API_KEY when GEMINI_API_KEYS is unset.

    On 429 / quota errors the client cycles through all available keys
    with a short pause between attempts and a longer pause at the end
    of each full round. After 3 full rounds without success it gives up
    and this function returns None — which the strict-mode caller
    correctly records as `n_failed` (not as a fake 0.0 score).
    """
    client = _get_rotating_vlm_client()

    content = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    messages = [{"role": "user", "content": content}]

    raw = client.call(model=model, messages=messages,
                      max_tokens=BENCHMARK_CONFIG["vlm_max_tokens"])
    if not raw:
        return None

    # Extract JSON from response — handle markdown code blocks,
    # thinking text before JSON, etc.
    try:
        # Strategy: find the first { ... } block in the response.
        # The regex handles one level of nested braces, which is
        # sufficient for our flat JSON schemas.
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw)
        if json_match:
            return json.loads(json_match.group())
        # Fallback: strip markdown fences and parse directly
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"[VLM] JSON parse failed: {e}\n  Raw (first 200 chars): {raw[:200]}")
        return None


# ============================================================
# Rotating VLM client — single module-level instance, thread-safe
# key rotation. One OpenAI-compatible client per API key, dispatched
# via a queue with per-key cooling on 429 / rate-limit responses.
# ============================================================

class _RotatingLLMClient:
    """Pool of one OpenAI client per API key, distributed via a queue.

    Why one client per key instead of a single shared client whose api_key
    field gets swapped before each call:

    The shared-client approach is racy under concurrent use. Two threads
    can both `swap api_key + fire request` interleaved, so the second
    swap clobbers the first thread's key BEFORE its request goes out.
    Result: requests get sent with the wrong key, leading to confused
    rate-limiting and stat tracking.

    Per-key clients sidestep this entirely: each thread checks out a
    client (which is bound to one key), uses it, returns it. The queue
    naturally limits in-flight calls to `n_keys` requests at a time —
    which IS the rate-limit budget — and per-key throttling is enforced
    by the underlying API's own per-key counter.

    Concurrency model: callers should call `client.call(...)` from
    multiple threads. The queue blocks if all clients are busy; this
    is the desired backpressure.

    Cooling/quarantine: when a key returns 429, we mark it cooling for
    30 seconds. The queue still hands out the client, but the call()
    method short-circuits and returns the client to the queue if the
    bound key is still in its cooling window — letting another thread
    grab it. With max_rounds full retry rounds, calls eventually fail
    rather than hang forever.
    """

    def __init__(self, api_keys: List[str], max_rounds: int = 3):
        if not api_keys:
            raise ValueError("RotatingLLMClient requires at least one API key")
        import openai
        import queue as _queue
        import threading as _threading

        self._keys = list(api_keys)
        self._n = len(self._keys)
        self._max_rounds = max_rounds

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2023-07-01-preview"
        ).strip()
        if not endpoint:
            raise StrictModeError(
                "AZURE_OPENAI_ENDPOINT is not set. Set it to the Azure "
                "OpenAI–compatible endpoint that serves your VLM "
                "(e.g. `https://<your-resource>.openai.azure.com/`)."
            )

        # One client per key — safe to use from any single thread at a time
        self._clients: List[Any] = [
            openai.AzureOpenAI(
                azure_endpoint=endpoint,
                api_version=api_version,
                api_key=k,
            ) for k in self._keys
        ]
        # Pool: index of an available client. Threads block on get(),
        # use the client, then put() the index back.
        self._pool: "_queue.Queue[int]" = _queue.Queue()
        for i in range(self._n):
            self._pool.put(i)

        # Per-key cooling timestamps (seconds since epoch)
        self._cooling_until: List[float] = [0.0] * self._n
        # Per-key request-firing lock (the openai client itself isn't
        # documented as thread-safe; a per-client lock keeps us tidy
        # even though we already serialize per-key via the queue)
        self._client_locks = [_threading.Lock() for _ in range(self._n)]

        # Stats — aggregate across all keys/threads. Updates protected by
        # _stats_lock since several threads += into it.
        self._stats_lock = _threading.Lock()
        self.stats: Dict[str, int] = {
            "total_calls": 0,
            "successes": 0,
            "rate_limit_hits": 0,
            "exhausted_after_max_rounds": 0,
            "non_rate_limit_errors": 0,
        }

    @staticmethod
    def _is_rate_limit(err_str: str) -> bool:
        s = err_str.lower()
        return any(t in s for t in [
            "429", "too many", "rate limit", "rate_limit", "quota",
            "resource_exhausted",
        ])

    def _bump(self, stat_name: str) -> None:
        with self._stats_lock:
            self.stats[stat_name] = self.stats.get(stat_name, 0) + 1

    def call(self, model: str, messages: List[Dict[str, Any]],
             max_tokens: int = 4096) -> str:
        """Run one chat completion. Thread-safe.

        Implementation: rotate through up to `n_keys × max_rounds` real
        attempts from the pool. A "real attempt" means a request actually
        fires — cooling-skips don't count against the budget, otherwise
        threads would burn through their budget waiting for keys to
        recover and never get a successful call. We cap the total wall
        time at `max_total_wait_sec` as a backstop against pathological
        loops.
        """
        import time as _time
        self._bump("total_calls")
        max_real_attempts = self._n * self._max_rounds
        real_attempts = 0
        deadline = _time.time() + 180.0  # absolute backstop: 3 minutes

        while real_attempts < max_real_attempts and _time.time() < deadline:
            idx = self._pool.get()
            fired = False
            try:
                # If this key is still cooling, wait for the soonest key
                # (across all keys) to expire, then retry without burning
                # an attempt. This is the only safe way to handle the case
                # where ALL keys cool at once: we have to actually wait.
                cool_until = self._cooling_until[idx]
                if cool_until > _time.time():
                    soonest = min(self._cooling_until)
                    wait = max(0.0, soonest - _time.time())
                    if wait > 0:
                        _time.sleep(min(wait + 0.05, 5.0))
                    continue  # don't increment real_attempts

                with self._client_locks[idx]:
                    fired = True
                    real_attempts += 1
                    try:
                        resp = self._clients[idx].chat.completions.create(
                            model=model, messages=messages, max_tokens=max_tokens,
                        )
                        content = resp.choices[0].message.content or ""
                        self._bump("successes")
                        return content.strip()
                    except Exception as e:
                        err = str(e)
                        if self._is_rate_limit(err):
                            self._bump("rate_limit_hits")
                            self._cooling_until[idx] = _time.time() + 30.0
                            logger.warning(
                                f"[LLM] 429 on key#{idx} "
                                f"(real attempt {real_attempts}/{max_real_attempts})"
                            )
                            _time.sleep(0.3)
                        else:
                            self._bump("non_rate_limit_errors")
                            logger.warning(
                                f"[LLM] non-429 error on key#{idx}: {err[:200]}"
                            )
                            _time.sleep(0.3)
            finally:
                self._pool.put(idx)

        self._bump("exhausted_after_max_rounds")
        logger.error(
            f"[LLM] Exhausted ({real_attempts} real attempts of "
            f"{max_real_attempts}, n_keys={self._n}). Returning empty; "
            f"caller will record as n_failed."
        )
        return ""


# Backward-compat alias: code elsewhere still imports this name.
_RotatingVLMClient = _RotatingLLMClient


_VLM_CLIENT: Optional["_RotatingVLMClient"] = None
_VLM_CLIENT_LOCK = None


def _get_rotating_vlm_client() -> "_RotatingVLMClient":
    """Lazily initialize the module-level rotating VLM client.

    Reads GEMINI_API_KEYS (comma-separated) from the environment. Falls
    back to AZURE_OPENAI_API_KEY (single key) for backward compatibility.

    Strict reproducibility note: the NUMBER of keys does not affect
    results — every call ultimately reaches Gemini through the same
    gateway. The keys exist purely to spread rate-limit budget. The
    manifest records `n_vlm_keys` so reviewers can see whether a run
    used 1 key or 5; results should match either way (modulo 429-driven
    n_failed counts in the audit).
    """
    global _VLM_CLIENT, _VLM_CLIENT_LOCK
    if _VLM_CLIENT is not None:
        return _VLM_CLIENT
    if _VLM_CLIENT_LOCK is None:
        import threading
        _VLM_CLIENT_LOCK = threading.Lock()
    with _VLM_CLIENT_LOCK:
        if _VLM_CLIENT is not None:
            return _VLM_CLIENT
        keys_env = os.environ.get("GEMINI_API_KEYS", "").strip()
        keys: List[str] = []
        if keys_env:
            keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        if not keys:
            single = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
            if single:
                keys = [single]
        if not keys:
            raise StrictModeError(
                "No VLM API keys configured. Set GEMINI_API_KEYS "
                "(comma-separated) or AZURE_OPENAI_API_KEY in the environment."
            )
        _VLM_CLIENT = _RotatingVLMClient(keys)
        logger.info(
            f"[VLM] Initialized rotating client with {len(keys)} key(s)"
        )
        return _VLM_CLIENT


def vlm_call_stats() -> Dict[str, int]:
    """Return cumulative call stats from the rotating client.

    Used by main() to embed counts into the run manifest and report.
    Safe to call before the client has been initialized — returns zeros.
    """
    if _VLM_CLIENT is None:
        return {"total_calls": 0, "successes": 0, "rate_limit_hits": 0,
                "exhausted_after_max_rounds": 0, "non_rate_limit_errors": 0,
                "n_keys": 0}
    s = dict(_VLM_CLIENT.stats)
    s["n_keys"] = _VLM_CLIENT._n
    return s


def _encode_crop(crop: np.ndarray) -> str:
    import base64
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(crop).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# Per-entity-type criteria for VLM-Judge. Single source of truth: shared
# by the cross-shot set judge (vlm_judge_entity_set), the intra-shot
# fidelity judge (Pillar 3), and the report writer.
_CROSS_SHOT_CRITERIA = {
    "character": {
        "face": "Face shape, facial features, and expression style",
        "hair": "Hair color, hairstyle, and length",
        "clothing": "Clothing style, color, and accessories",
        "build": "Body build, proportions, and posture",
    },
    "location": {
        "layout": "Architectural style, spatial layout, and structure",
        "color_mood": "Color palette, lighting mood, and atmosphere",
        "landmarks": "Key visual landmarks, distinctive features, or objects in the scene",
        "perspective": "Viewing angle, depth, and spatial composition",
    },
    "object": {
        "shape": "Shape, form, and silhouette",
        "color_texture": "Color and texture/material",
        "proportions": "Size and proportions",
        "details": "Distinctive visual features and fine details",
    },
}


def _evenly_sampled_indices(n: int, k: int) -> List[int]:
    """Return k indices evenly spread across [0, n-1] (inclusive endpoints).

    Used to down-sample crop sets that exceed the VLM's image-input
    capacity. Even spacing preserves the temporal spread of the
    appearances rather than biasing toward any region of the episode.

    For n <= k, returns all indices in order. For n > k, the result
    always includes 0 and n-1 (so first and last appearances are scored).
    """
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def vlm_judge_entity_set(
    crops: List[np.ndarray],
    shot_keys: List[str],
    description: str,
    entity_name: str,
    entity_type: str,
    model: str = "gemini-2.5-pro",
    max_images: int = 16,
) -> Dict[str, Any]:
    """Ask Gemini to rate consistency across N appearances of an entity.

    Sends all N (or up to max_images) crops in a single call. Asks for:
      * Overall consistency score 1-10
      * Per-criteria consistency scores 1-10 (4 criteria per entity type)
      * A binary all_consistent verdict
      * The list of outlier image indices (those that don't match the others)

    Returns dict with:
      consistency: float           — overall consistency 1-10, normalized 0-1
      all_consistent: bool         — Gemini's verdict that all images depict
                                      the same entity (analog of pair-based
                                      "is_same" but at the set level)
      outlier_shots: List[str]     — shot keys whose images Gemini flagged
                                      as inconsistent with the rest
      criteria: Dict[str, float]   — per-criteria consistency scores 0-1
      reason: str
      vlm_failed: bool             — True if the API call or parse failed
      n_images_sent: int           — actual count sent to Gemini (after capping)
      sampled_shot_keys: List[str] — which shots were sampled (after capping)

    On failure: returns vlm_failed=True with all numeric fields None.
    Caller should record n_failed in the corresponding MetricValue.
    """
    n_total = len(crops)
    if n_total < 2:
        return {
            "consistency": None, "all_consistent": None,
            "outlier_shots": [], "criteria": {}, "reason": "n<2",
            "vlm_failed": False, "n_images_sent": 0,
            "sampled_shot_keys": [],
        }

    # Down-sample evenly if N exceeds image-input capacity
    keep_indices = _evenly_sampled_indices(n_total, max_images)
    sampled_crops = [crops[i] for i in keep_indices]
    sampled_shots = [shot_keys[i] for i in keep_indices]
    n_sent = len(sampled_crops)

    criteria = _CROSS_SHOT_CRITERIA.get(entity_type, _CROSS_SHOT_CRITERIA["object"])
    criteria_list = "\n".join(f"  - {k}: {v}" for k, v in criteria.items())
    criteria_json = ", ".join(f'"{k}": N' for k in criteria.keys())

    image_listing = "\n".join(
        f"  Image {i + 1}: from shot {sk}" for i, sk in enumerate(sampled_shots)
    )

    prompt = (
        f"You are evaluating cross-shot consistency of an entity that "
        f"appears in {n_total} shots of a generated video episode"
        + (f" (showing {n_sent} sampled appearances)" if n_sent < n_total else "")
        + ".\n\n"
        f"Entity type: {entity_type}\n"
        f"Entity name: {entity_name}\n"
        f"Description: {description}\n\n"
        f"Images provided (in order):\n{image_listing}\n\n"
        f"Rate how CONSISTENT these appearances are with each other "
        f"(NOT how well they match the description — that's a different "
        f"metric). All images should depict the same entity with the "
        f"same visible attributes.\n\n"
        f"Score each criterion from 1 to 10 "
        f"(1=appearances are completely inconsistent, "
        f"10=appearances are identical across all images):\n{criteria_list}\n\n"
        f"Then give an overall consistency score (1-10), a binary verdict "
        f"on whether ALL images depict the same entity consistently, "
        f"and a list of any image indices (1-based) that visibly differ "
        f"from the rest.\n\n"
        f"Respond ONLY with JSON:\n"
        f'{{{criteria_json}, "consistency": N, "all_consistent": true/false, '
        f'"outlier_indices": [array of 1-based indices, or empty if none], '
        f'"reason": "brief explanation"}}'
    )

    images_b64 = [_encode_crop(c) for c in sampled_crops]
    result = _vlm_api_call(images_b64, prompt, model)

    if not result:
        return {
            "consistency": None, "all_consistent": None,
            "outlier_shots": [], "criteria": {k: None for k in criteria},
            "reason": "VLM call failed", "vlm_failed": True,
            "n_images_sent": n_sent, "sampled_shot_keys": sampled_shots,
        }

    parsed: Dict[str, Any] = {
        "consistency": None,
        "all_consistent": bool(result.get("all_consistent", False)),
        "outlier_shots": [],
        "criteria": {k: None for k in criteria},
        "reason": result.get("reason", ""),
        "vlm_failed": False,
        "n_images_sent": n_sent,
        "sampled_shot_keys": sampled_shots,
    }
    if "consistency" in result:
        try:
            parsed["consistency"] = float(result["consistency"]) / 10.0
        except (TypeError, ValueError):
            pass
    for k in criteria.keys():
        if k in result and result[k] is not None:
            try:
                parsed["criteria"][k] = float(result[k]) / 10.0
            except (TypeError, ValueError):
                pass

    # Map 1-based outlier indices back to shot keys for auditability.
    raw_outliers = result.get("outlier_indices", []) or []
    if isinstance(raw_outliers, list):
        for oi in raw_outliers:
            try:
                idx = int(oi) - 1  # 1-based to 0-based
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(sampled_shots):
                parsed["outlier_shots"].append(sampled_shots[idx])

    return parsed


def vlm_judge_entity(
    crop_first: np.ndarray, crop_later: np.ndarray,
    description: str, entity_name: str, entity_type: str,
    model: str = "gemini-2.5-pro",
) -> Dict[str, Any]:
    """DEPRECATED pairwise judge — kept for backward compat with intra-shot
    fidelity (which sends just one crop) and any callers that still use
    the old API. Cross-shot judging now uses `vlm_judge_entity_set`
    instead, which sends all N appearances in a single call.
    """
    criteria = _CROSS_SHOT_CRITERIA.get(entity_type, _CROSS_SHOT_CRITERIA["object"])
    criteria_list = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    criteria_json = ", ".join(f'"{k}": N' for k in criteria.keys())

    prompt = f"""You are evaluating cross-shot consistency in generated videos.

Entity type: {entity_type}
Entity name: {entity_name}
Description: {description}

Image 1 is from the entity's first appearance.
Image 2 is from a later appearance in a different shot.

Score each criterion from 1 to 10 (1=completely different, 10=identical):
{criteria_list}

Then give an overall similarity score (1-10) and a binary judgment.

Respond ONLY with JSON:
{{{criteria_json}, "similarity": N, "is_same": true/false, "reason": "brief explanation"}}"""

    result = _vlm_api_call([_encode_crop(crop_first), _encode_crop(crop_later)], prompt, model)

    if result:
        parsed = {
            "is_same": bool(result.get("is_same", False)),
            "similarity": float(result.get("similarity", 0)) / 10.0,
            "reason": result.get("reason", ""),
            "criteria": {},
        }
        for k in criteria.keys():
            if k in result:
                parsed["criteria"][k] = float(result[k]) / 10.0
        return parsed

    return {"is_same": False, "similarity": 0.0, "reason": "VLM call failed", "criteria": {}}


def vlm_judge_scene_set(
    shot_frames: List[List[np.ndarray]],
    shot_keys: List[str],
    description: str,
    location_name: str,
    model: str = "gemini-2.5-pro",
    max_shots: int = 8,
    max_frames_per_shot: int = 2,
) -> Dict[str, Any]:
    """N-at-once location consistency judge using full frames.

    Sends `shot_frames[i]` (1-2 quality frames per shot) for each of N
    shots where the location appears, all in one Gemini call. Caps at
    max_shots × max_frames_per_shot total images so we stay under the
    image-input limit; the default (8 × 2 = 16) matches the entity set
    judge's cap.

    The shot index for each image is included in the prompt so Gemini
    can flag which specific shot is the outlier (mapped back to shot
    keys in the response).

    Returns dict matching vlm_judge_entity_set's contract:
      consistency, all_consistent, outlier_shots, criteria, reason,
      vlm_failed, n_images_sent, sampled_shot_keys.
    """
    n_shots = len(shot_frames)
    if n_shots < 2:
        return {
            "consistency": None, "all_consistent": None,
            "outlier_shots": [], "criteria": {}, "reason": "n<2",
            "vlm_failed": False, "n_images_sent": 0,
            "sampled_shot_keys": [],
        }

    keep_indices = _evenly_sampled_indices(n_shots, max_shots)
    sampled_shot_keys = [shot_keys[i] for i in keep_indices]

    images_b64: List[str] = []
    image_to_shot_idx: List[int] = []  # which sampled-shot each image came from
    image_listing_lines: List[str] = []
    img_counter = 0
    for sampled_pos, shot_orig_idx in enumerate(keep_indices):
        frames_for_shot = shot_frames[shot_orig_idx][:max_frames_per_shot]
        if not frames_for_shot:
            continue
        for f in frames_for_shot:
            images_b64.append(_encode_crop(f))
            image_to_shot_idx.append(sampled_pos)
            img_counter += 1
            image_listing_lines.append(
                f"  Image {img_counter}: from shot {sampled_shot_keys[sampled_pos]}"
            )
    if not images_b64:
        return {
            "consistency": None, "all_consistent": None,
            "outlier_shots": [], "criteria": {}, "reason": "no frames",
            "vlm_failed": False, "n_images_sent": 0,
            "sampled_shot_keys": sampled_shot_keys,
        }

    n_sent = len(images_b64)
    image_listing = "\n".join(image_listing_lines)

    prompt = (
        f"You are evaluating cross-shot LOCATION consistency in a "
        f"generated video episode. The same location appears in "
        f"{n_shots} shots"
        + (f" (showing {n_sent} sampled frames across "
           f"{len(sampled_shot_keys)} of those shots)" if n_shots > max_shots else "")
        + ".\n\n"
        f"Location name: {location_name}\n"
        f"Location description: {description}\n\n"
        f"Images provided (in order):\n{image_listing}\n\n"
        f"IGNORE the characters/people in the foreground. Focus ONLY on "
        f"the background environment.\n\n"
        f"Score each criterion from 1 to 10 "
        f"(1=completely inconsistent across shots, 10=identical location):\n"
        f"  - layout: Architectural style, spatial layout, and room/space structure\n"
        f"  - color_mood: Color palette, lighting mood, and atmosphere\n"
        f"  - landmarks: Key visual landmarks, furniture, distinctive features\n"
        f"  - perspective: Camera angle/depth (different angles of the SAME "
        f"place should still score high on layout/landmarks)\n\n"
        f"Then give an overall consistency score (1-10), a binary verdict on "
        f"whether ALL shots depict the same location, and any outlier "
        f"image indices (1-based).\n\n"
        f"Respond ONLY with JSON:\n"
        f'{{"layout": N, "color_mood": N, "landmarks": N, "perspective": N, '
        f'"consistency": N, "all_consistent": true/false, '
        f'"outlier_indices": [array of 1-based indices, or empty], '
        f'"reason": "brief explanation"}}'
    )

    result = _vlm_api_call(images_b64, prompt, model)

    criteria_keys = ["layout", "color_mood", "landmarks", "perspective"]
    if not result:
        return {
            "consistency": None, "all_consistent": None,
            "outlier_shots": [], "criteria": {k: None for k in criteria_keys},
            "reason": "VLM call failed", "vlm_failed": True,
            "n_images_sent": n_sent, "sampled_shot_keys": sampled_shot_keys,
        }

    parsed: Dict[str, Any] = {
        "consistency": None,
        "all_consistent": bool(result.get("all_consistent", False)),
        "outlier_shots": [],
        "criteria": {k: None for k in criteria_keys},
        "reason": result.get("reason", ""),
        "vlm_failed": False,
        "n_images_sent": n_sent,
        "sampled_shot_keys": sampled_shot_keys,
    }
    if "consistency" in result:
        try:
            parsed["consistency"] = float(result["consistency"]) / 10.0
        except (TypeError, ValueError):
            pass
    for k in criteria_keys:
        if k in result and result[k] is not None:
            try:
                parsed["criteria"][k] = float(result[k]) / 10.0
            except (TypeError, ValueError):
                pass

    # Map 1-based image outlier indices → unique shot keys
    raw_outliers = result.get("outlier_indices", []) or []
    if isinstance(raw_outliers, list):
        outlier_shots: Set[str] = set()
        for oi in raw_outliers:
            try:
                idx = int(oi) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(image_to_shot_idx):
                outlier_shots.add(sampled_shot_keys[image_to_shot_idx[idx]])
        parsed["outlier_shots"] = sorted(outlier_shots)

    return parsed


def vlm_judge_scene(
    frames_first: List[np.ndarray], frames_later: List[np.ndarray],
    description: str, location_name: str,
    model: str = "gemini-2.5-pro",
) -> Dict[str, Any]:
    """
    Ask Gemini to evaluate cross-shot LOCATION consistency using full frames.

    Locations can't be reliably cropped — they ARE the background. So we send
    2-3 full frames per shot and ask Gemini to compare the backgrounds while
    ignoring foreground characters.

    Args:
        frames_first: 2-3 quality frames from the location's first appearance
        frames_later: 2-3 quality frames from a later shot at the same location
    """
    # Send up to 2 frames per shot (4 images total max)
    f1 = frames_first[:2]
    f2 = frames_later[:2]
    all_b64 = [_encode_crop(f) for f in f1] + [_encode_crop(f) for f in f2]

    n1, n2 = len(f1), len(f2)
    lbl1 = f"Images 1-{n1}" if n1 > 1 else "Image 1"
    lbl2 = f"Images {n1+1}-{n1+n2}" if n2 > 1 else f"Image {n1+1}"

    prompt = f"""You are evaluating cross-shot LOCATION consistency in generated videos.

Location name: {location_name}
Location description: {description}

{lbl1}: frames from this location's first appearance.
{lbl2}: frames from a later shot at what should be the same location.

IGNORE the characters/people in the foreground. Focus ONLY on the background environment.

Score each criterion from 1 to 10 (1=completely different, 10=identical):
- layout: Architectural style, spatial layout, and room/space structure
- color_mood: Color palette, lighting mood, and atmosphere of the environment
- landmarks: Key visual landmarks, furniture, distinctive features in the background
- perspective: Camera angle and spatial depth (note: different angles of the SAME place should still score high on layout/landmarks)

Then give an overall similarity score (1-10) and a binary judgment.

Respond ONLY with JSON:
{{"layout": N, "color_mood": N, "landmarks": N, "perspective": N, "similarity": N, "is_same": true/false, "reason": "brief explanation"}}"""

    result = _vlm_api_call(all_b64, prompt, model)
    if result:
        parsed = {
            "is_same": bool(result.get("is_same", False)),
            "similarity": float(result.get("similarity", 0)) / 10.0,
            "reason": result.get("reason", ""),
            "criteria": {},
        }
        for k in ["layout", "color_mood", "landmarks", "perspective"]:
            if k in result:
                parsed["criteria"][k] = float(result[k]) / 10.0
        return parsed
    return {"is_same": False, "similarity": 0.0, "reason": "VLM call failed", "criteria": {}}


def vlm_judge_style(
    frame1: np.ndarray, frame2: np.ndarray, model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Ask Gemini to evaluate visual style consistency with per-criteria scores.

    Returns dict with:
      criteria: {art_style, color_grading, rendering, lighting, line_work} (each 0-1)
      similarity: float (1-10, normalized to 0-1)
      is_consistent: bool
      reason: str
    """
    prompt = """You are evaluating visual style consistency between two frames from a generated video episode.

Score each criterion from 1 to 10 (1=completely different, 10=identical):
- art_style: Art style (anime, photorealistic, cartoon, painterly, etc.)
- color_grading: Color grading and overall palette
- rendering: Rendering quality and detail level
- lighting: Lighting style (warm/cold, harsh/soft, direction)
- line_work: Line work, edge treatment, and texture style

Then give an overall similarity score (1-10) and a binary judgment.

Respond ONLY with JSON:
{"art_style": N, "color_grading": N, "rendering": N, "lighting": N, "line_work": N, "similarity": N, "is_consistent": true/false, "reason": "brief explanation"}"""

    result = _vlm_api_call([_encode_crop(frame1), _encode_crop(frame2)], prompt, model)

    if result:
        parsed = {
            "is_consistent": bool(result.get("is_consistent", False)),
            "similarity": float(result.get("similarity", 5)) / 10.0,
            "reason": result.get("reason", ""),
            "criteria": {},
        }
        for k in ["art_style", "color_grading", "rendering", "lighting", "line_work"]:
            if k in result:
                parsed["criteria"][k] = float(result[k]) / 10.0
        return parsed

    return {"is_consistent": False, "similarity": 0.0, "reason": "VLM call failed", "criteria": {}}


# ---- Gap decay ----

def compute_gap_decay(
    entity_crops: Dict[str, List[EntityCrop]],
    shot_order: List[str],
    entity_types: Tuple[str, ...] = ("character", "object"),
) -> List[Dict[str, Any]]:
    """Compute (gap, similarity) records for ALL pairs of appearances.

    Unlike the earlier anchor-based version, this enumerates the full
    upper-triangle of (i, j) pairs so the decay curve isn't biased by
    which crop was chosen as the anchor. Each pair contributes once
    (i < j by shot position), tagged with entity_type for per-type plots.
    """
    shot_to_pos = {sk: i for i, sk in enumerate(shot_order)}
    points: List[Dict[str, Any]] = []
    for name, crops in entity_crops.items():
        for etype in entity_types:
            typed = [c for c in crops if c.entity_type == etype
                     and c.dino_embedding is not None]
            if len(typed) < 2:
                continue
            typed_sorted = sorted(typed, key=lambda c: shot_to_pos.get(c.shot_key, 0))
            n = len(typed_sorted)
            for i in range(n):
                for j in range(i + 1, n):
                    gap = abs(shot_to_pos.get(typed_sorted[j].shot_key, 0)
                              - shot_to_pos.get(typed_sorted[i].shot_key, 0))
                    points.append({
                        "entity_name": name,
                        "entity_type": etype,
                        "gap": gap,
                        "similarity": compute_dino_similarity(
                            typed_sorted[i].dino_embedding,
                            typed_sorted[j].dino_embedding,
                        ),
                    })
    return points


# ============================================================
# Pillar 1B: Intra-shot prompt-following alignment
#   - intra_entity_presence  : did each scheduled entity actually appear?
#   - intra_entity_fidelity  : per-criteria match against registry description
#   - intra_action_fidelity  : annotated-grid VLM judge on action description
#
# These metrics measure WHETHER THE GENERATED VIDEO FOLLOWS THE PROMPT,
# which VBench's 7 quality dims do not test. A model could produce a
# beautiful, smooth, perfectly-self-consistent shot of the WRONG character
# doing the WRONG thing and Pillar 1 (VBench) would score it highly.
# These three metrics close that gap.
# ============================================================

# Per-entity-type criteria for the intra-shot fidelity judge. Same rubric
# as cross-shot vlm_judge_entity (deliberate consistency — reviewers can
# compare cross-shot and intra-shot scores apples-to-apples).
_INTRA_FIDELITY_CRITERIA = {
    "character": {
        "face": "Face shape, facial features, and expression style",
        "hair": "Hair color, hairstyle, and length",
        "clothing": "Clothing style, color, and accessories",
        "build": "Body build, proportions, and posture",
    },
    "object": {
        "shape": "Shape, form, and silhouette",
        "color_texture": "Color and texture/material",
        "proportions": "Size and proportions",
        "details": "Distinctive visual features and fine details",
    },
    "location": {
        "layout": "Architectural style, spatial layout, and structure",
        "color_mood": "Color palette, lighting mood, and atmosphere",
        "landmarks": "Key visual landmarks, distinctive features, or objects",
        "perspective": "Viewing angle, depth, and spatial composition",
    },
}

# Action fidelity criteria. "object_interaction" is None'd out at the
# scoring stage when the action description doesn't reference any scheduled
# objects — we don't penalize a shot for failing a criterion that didn't
# apply to it.
_INTRA_ACTION_CRITERIA = {
    "subject_identity": "Are the labeled subjects the correct people from the description? "
                        "(Score 10 if the labeled boxes match who the description names.)",
    "subject_action": "Does the named subject perform the action/verb described? "
                      "(Score 10 if the action is clearly depicted across the frame sequence.)",
    "object_interaction": "If the action involves an object, is the object present and "
                          "interacted with as described? Use null if no object is named.",
    "motion_quality": "Is the motion natural, smooth, and consistent across the frames? "
                      "(Score 10 if frames clearly progress through the described action.)",
}

# Tiny boxes are flagged in the audit but still counted as present:
# entity detection is upstream of fidelity, and a small but correct
# detection shouldn't be discarded as "absent."
_TINY_BOX_THRESHOLD = 0.01  # bbox area / frame area

# Distinct visually-separable colors for entity annotation overlays.
_ENTITY_COLORS = [
    (255, 64, 64),    # red
    (64, 192, 255),   # cyan-blue
    (64, 255, 96),    # green
    (255, 192, 0),    # amber
    (224, 64, 224),   # magenta
    (255, 128, 0),    # orange
    (128, 255, 255),  # pale cyan
    (255, 96, 160),   # pink
]


def _bbox_area_ratio(bbox: Tuple[float, float, float, float],
                     frame_shape: Tuple[int, int]) -> float:
    """Return bbox area as fraction of frame area."""
    h, w = frame_shape[:2]
    if h == 0 or w == 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1)) / float(h * w)


def _draw_labeled_bbox(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> np.ndarray:
    """Draw a labeled colored bounding box on a copy of frame."""
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Box
    for t in range(thickness):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
    # Label background and text
    try:
        font = ImageFont.load_default()
        # Approximate text size; PIL versions differ on the API
        try:
            tw = draw.textlength(label, font=font)
            th = 12
        except AttributeError:
            tw, th = font.getsize(label)
    except Exception:
        font = None
        tw, th = len(label) * 7, 12
    pad = 2
    label_y_top = max(0, y1 - th - 2 * pad)
    draw.rectangle(
        [x1, label_y_top, x1 + tw + 2 * pad, label_y_top + th + 2 * pad],
        fill=color,
    )
    if font:
        draw.text((x1 + pad, label_y_top + pad), label, fill=(0, 0, 0), font=font)
    else:
        draw.text((x1 + pad, label_y_top + pad), label, fill=(0, 0, 0))
    return np.array(img)


def _build_action_grid(
    annotated_frames: List[np.ndarray],
    cols: int = 3,
    cell_w: int = 360,
    cell_h: int = 240,
    gap: int = 8,
) -> np.ndarray:
    """Tile annotated frames into a labeled grid (default 2x3)."""
    n = len(annotated_frames)
    rows = (n + cols - 1) // cols
    canvas_h = rows * cell_h + (rows + 1) * gap
    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)  # dark grey
    for i, fr in enumerate(annotated_frames):
        r, c = divmod(i, cols)
        # Resize keeping aspect, then pad
        pil = Image.fromarray(fr).convert("RGB")
        pil.thumbnail((cell_w, cell_h), Image.LANCZOS)
        arr = np.array(pil)
        oy = gap + r * (cell_h + gap) + (cell_h - arr.shape[0]) // 2
        ox = gap + c * (cell_w + gap) + (cell_w - arr.shape[1]) // 2
        canvas[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
    return canvas


def _name_in_action(action: str, name: str) -> bool:
    """Case-insensitive whole-word presence of an entity name in action text."""
    if not name:
        return False
    pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    return bool(pat.search(action or ""))


# ============================================================
# Unified shot-level grounding pass
#
# Runs ONCE per shot, produces everything Pillar 2 and Pillar 3 need:
#   - canonical crop per (shot, entity)  → used for cross-shot DINOv2,
#                                          intra-shot fidelity, anchor save
#   - all per-frame detections per entity → used for action grid annotation
#   - presence status per entity          → "present" / "absent" / "weak"
#
# Why unified: previously these three needs ran THREE separate grounding
# passes on the same frames, sometimes with different selection criteria,
# producing different crops per use. That meant the "anchor crop saved
# for audit" was not actually the same pixels Gemini judged for fidelity.
# After this refactor, the anchor IS the canonical IS the fidelity crop —
# one pass, one crop on disk, full traceability.
# ============================================================

def ground_shot_unified(
    shot_key: str,
    frames: List[np.ndarray],
    scheduled_entities: Dict[str, List[str]],
    entity_descriptions: Dict[str, dict],
    artifact_dir: str,
    device: str,
    grounding_thresholds: Dict[str, float],
    clip_match_threshold: float,
) -> Dict[str, Any]:
    """Single grounding pass per shot, returns everything downstream needs.

    Returns
    -------
    {
      "per_entity": {
        entity_name: {
          "etype": "character" | "object" | "location",
          "description": str,
          "scheduled": True,
          "canonical": EntityCrop or None,    # best-quality detection
          "canonical_score_components": {     # for audit
            "clip", "sharpness_norm", "area_norm", "selection",
          },
          "all_detections": [                 # ALL per-frame detections,
            {                                 #   used by action annotation
              "frame_idx": int, "bbox": (...), "confidence": float,
              "clip": float, "selection": float,
            },
            ...
          ],
          "status": "present" | "absent" | "weak",
            # present : canonical crop exists AND CLIP >= threshold
            # weak    : canonical exists, CLIP between (0, threshold)
            #           — flagged in audit but still scored
            # absent  : no detections OR all CLIP scores were 0
          "is_low_confidence": bool,
            # True iff status=="present" AND clip < threshold + 0.05
            # — barely cleared the bar; flagged in fidelity audit
          "is_small": bool,
            # True iff canonical bbox area < 1% of frame area
          "canonical_path": str or None,      # path to saved PNG
        }
      },
      "n_grounding_calls": int,               # GroundingDINO calls made
    }

    Saves canonical crops to {artifact_dir}/canonical_{etype}_{name}_*.png
    """
    os.makedirs(artifact_dir, exist_ok=True)

    # name → (description, etype) lookup
    name_to_desc: Dict[str, Tuple[str, str]] = {}
    for label, info in entity_descriptions.items():
        if isinstance(info, dict):
            nm = info.get("name", label)
            desc = info.get("description", "")
        else:
            nm = label
            desc = str(info) if isinstance(info, str) else ""
        if label.startswith("Character"):
            etype = "character"
        elif label.startswith("Location"):
            etype = "location"
        else:
            etype = "object"
        name_to_desc[nm] = (desc, etype)

    # All scheduled entities (with their declared types) for this shot
    targets: List[Tuple[str, str, str]] = []  # (name, desc, etype)
    for sched_key, default_etype in [("characters", "character"),
                                      ("places", "location"),
                                      ("objects", "object")]:
        for name in scheduled_entities.get(sched_key, []) or []:
            desc, registered_etype = name_to_desc.get(name, (name, default_etype))
            # Trust the schedule's declared type when available
            targets.append((name, desc or name, default_etype))

    out: Dict[str, Any] = {"per_entity": {}, "n_grounding_calls": 0}

    for ent_name, ent_desc, etype in targets:
        det_records: List[Dict[str, Any]] = []
        best = {"selection": -1.0, "clip": 0.0, "frame_idx": -1,
                "bbox": None, "crop": None, "frame": None, "components": None}

        for fi, frame in enumerate(frames):
            # GroundingDINO call: ONE per (frame, entity). This is the
            # only place we call the grounder for this shot.
            detections = ground_one_entity_in_frame(
                frame=frame,
                entity_name=ent_name,
                description=ent_desc,
                entity_type=etype,
                device=device,
                box_threshold=grounding_thresholds["box_threshold"],
                text_threshold=grounding_thresholds["text_threshold"],
                return_all=True,  # action annotation may want multiple boxes
            )
            out["n_grounding_calls"] += 1

            for det in detections:
                bbox = det["bbox"]
                crop = crop_entity(frame, bbox)
                try:
                    clip_score = compute_clip_text_sim(crop, ent_desc, device)
                except Exception:
                    clip_score = 0.0
                comp = compute_selection_score(clip_score, crop, frame.shape)

                det_records.append({
                    "frame_idx": fi,
                    "bbox": bbox,
                    "confidence": det["confidence"],
                    "clip": clip_score,
                    "selection": comp["selection"],
                    "components": comp,
                })

                if comp["selection"] > best["selection"]:
                    best.update({
                        "selection": comp["selection"],
                        "clip": clip_score,
                        "frame_idx": fi,
                        "bbox": bbox,
                        "crop": crop,
                        "frame": frame,
                        "components": comp,
                    })

        # Build the EntityCrop (or None) and save the canonical
        canonical: Optional[EntityCrop] = None
        canonical_path: Optional[str] = None
        is_small = False
        is_low_confidence = False

        if best["bbox"] is not None:
            area = _bbox_area_ratio(best["bbox"], best["frame"].shape)
            is_small = area < _TINY_BOX_THRESHOLD
            is_low_confidence = best["clip"] < clip_match_threshold + 0.05

            safe = re.sub(r"[^a-zA-Z0-9]+", "_", ent_name).strip("_") or "entity"
            canonical_path = os.path.join(
                artifact_dir,
                f"canonical_{etype}_{safe}_f{best['frame_idx']}_clip{best['clip']:.2f}.png",
            )
            try:
                Image.fromarray(best["crop"]).save(canonical_path)
            except Exception as e:
                logger.warning(f"[Ground] {shot_key}: save canonical failed for {ent_name}: {e}")
                canonical_path = None

            # Compute DINOv2 embedding once, attach to EntityCrop
            try:
                dino_emb = compute_dino_embedding(best["crop"], device)
            except Exception as e:
                logger.warning(f"[Ground] {shot_key}: dino emb failed for {ent_name}: {e}")
                dino_emb = None

            canonical = EntityCrop(
                entity_name=ent_name,
                entity_type=etype,
                entity_description=ent_desc,
                shot_key=shot_key,
                frame_idx=best["frame_idx"],
                bbox=best["bbox"],
                crop_path=canonical_path or "",
                clip_text_sim=best["clip"],
                dino_embedding=dino_emb,
            )

        # Status
        if canonical is None:
            status = "absent"
        elif best["clip"] < clip_match_threshold:
            status = "weak"
        else:
            status = "present"

        out["per_entity"][ent_name] = {
            "etype": etype,
            "description": ent_desc,
            "scheduled": True,
            "canonical": canonical,
            "canonical_score_components": best["components"],
            "all_detections": det_records,
            "status": status,
            "is_low_confidence": is_low_confidence,
            "is_small": is_small,
            "canonical_path": canonical_path,
        }

    return out



def evaluate_intra_shot_alignment(
    shot_key: str,
    ground_result: Dict[str, Any],
    frames: List[np.ndarray],
    action_description: str,
    scheduled_entities: Dict[str, List[str]],
    artifact_dir: str,
    vlm_model: str,
    save_action_grid: bool = False,
) -> Dict[str, Any]:
    """Compute Pillar 3 metrics for ONE shot from a pre-grounded result.

    Takes the output of `ground_shot_unified` (already-found canonical
    crops + per-frame detections) so we don't re-run grounding. Pillar 2
    and Pillar 3 share the same grounding pass and the same canonical
    crops, so the audit chain (canonical crop → DINOv2 → VLM judge) is
    consistent across both pillars.

    Returns
    -------
    {
      "presence": {etype: {n_total, n_present, n_weak, n_small, n_absent, items}},
      "fidelity": {etype: {entity_name: {overall, criteria, vlm_failed,
                                          is_low_confidence, ...}}},
      "action":   {criteria: {...}, overall, action_depicted, n_unannotated,
                   skipped_reason, vlm_failed, grid_path or None},
      "errors":   [...],
    }

    Parameters
    ----------
    save_action_grid : if True, persist the annotated 2x3 grid PNG to
        artifact_dir. Default False (the grid is built and sent to Gemini
        in-memory but not saved, since 140 episodes x ~18 shots = ~2500
        grid PNGs is a lot of disk for a metric that's already audit-able
        through the per-shot _eval.json + canonical crops).
    """
    out: Dict[str, Any] = {
        "presence": {},
        "fidelity": {},
        "action": None,
        "errors": [],
    }

    per_entity = ground_result["per_entity"]

    # ---- PRESENCE: tri-state count per entity type ----
    type_to_sched_key = {"character": "characters",
                         "object": "objects",
                         "location": "places"}
    for etype, sched_key in type_to_sched_key.items():
        items = scheduled_entities.get(sched_key, []) or []
        type_results = {
            "n_total": len(items),
            "n_present": 0,        # status == "present" (CLIP cleared threshold)
            "n_weak": 0,           # status == "weak"    (CLIP > 0 but < threshold)
            "n_small": 0,          # bbox area < 1% of frame (subset of n_present)
            "n_absent": 0,         # GroundingDINO returned nothing
            "n_low_confidence": 0, # CLIP just barely cleared threshold
            "items": [],
        }
        for ent_name in items:
            info = per_entity.get(ent_name)
            if info is None:
                # Should not happen — every scheduled entity should have an
                # entry. Treat as absent for safety.
                type_results["n_absent"] += 1
                type_results["items"].append({
                    "name": ent_name, "status": "absent",
                    "note": "missing from grounding result",
                })
                continue
            status = info["status"]
            if status == "present":
                type_results["n_present"] += 1
                if info["is_small"]:
                    type_results["n_small"] += 1
                if info["is_low_confidence"]:
                    type_results["n_low_confidence"] += 1
            elif status == "weak":
                type_results["n_weak"] += 1
            else:
                type_results["n_absent"] += 1
            canon = info["canonical"]
            type_results["items"].append({
                "name": ent_name,
                "status": status,
                "is_small": info["is_small"],
                "is_low_confidence": info["is_low_confidence"],
                "clip_sim": canon.clip_text_sim if canon else None,
                "frame_idx": canon.frame_idx if canon else None,
                "canonical_path": info["canonical_path"],
            })
        out["presence"][etype] = type_results

    # ---- FIDELITY: send canonical crop to Gemini-2.5-pro ----
    # Per-criteria scores are surfaced as separate metrics by the caller.
    # We send for status in ("present", "weak") — weak crops still get
    # judged but are flagged with is_low_confidence in the audit.
    for ent_name, info in per_entity.items():
        if info["status"] == "absent":
            continue
        canonical = info["canonical"]
        if canonical is None or not info["canonical_path"]:
            continue
        etype = info["etype"]
        criteria = _INTRA_FIDELITY_CRITERIA.get(etype)
        if not criteria:
            continue
        criteria_list = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        criteria_json = ", ".join(f'"{k}": N' for k in criteria.keys())

        prompt = (
            f"You are evaluating whether a generated frame depicts an entity "
            f"matching its description.\n\n"
            f"Entity type: {etype}\n"
            f"Entity name: {ent_name}\n"
            f"Description: {info['description']}\n\n"
            f"Score each criterion from 1 to 10 (1=completely mismatched, "
            f"10=perfect match):\n{criteria_list}\n\n"
            f"Then give an overall fidelity score (1-10).\n\n"
            f"Respond ONLY with JSON:\n"
            f'{{{criteria_json}, "fidelity": N, "reason": "brief explanation"}}'
        )

        try:
            crop = np.array(Image.open(info["canonical_path"]))
            result = _vlm_api_call([_encode_crop(crop)], prompt, vlm_model)
        except Exception as e:
            result = None
            out["errors"].append(f"fidelity {ent_name}: {e}")

        ent_fid: Dict[str, Any] = {
            "overall": None, "reason": "", "vlm_failed": True,
            "criteria": {k: None for k in criteria.keys()},
            "is_low_confidence": info["is_low_confidence"],
            "is_small": info["is_small"],
            "status": info["status"],
            "anchor_clip": canonical.clip_text_sim,
        }
        if result:
            ent_fid["vlm_failed"] = False
            ent_fid["reason"] = result.get("reason", "")
            if "fidelity" in result:
                try:
                    ent_fid["overall"] = float(result["fidelity"]) / 10.0
                except (TypeError, ValueError):
                    pass
            for k in criteria.keys():
                if k in result and result[k] is not None:
                    try:
                        ent_fid["criteria"][k] = float(result[k]) / 10.0
                    except (TypeError, ValueError):
                        pass

        out["fidelity"].setdefault(etype, {})[ent_name] = ent_fid

    # ---- ACTION FIDELITY: annotated grid + VLM judge ----
    # Annotation uses ALL detections from the unified grounding pass —
    # we want to show motion across frames, so a single best frame won't
    # do. Each scheduled character/object gets a unique color.
    annotation_targets = []  # (name, etype, color)
    color_idx = 0
    for sched_key, etype in [("characters", "character"), ("objects", "object")]:
        for ent_name in scheduled_entities.get(sched_key, []) or []:
            color = _ENTITY_COLORS[color_idx % len(_ENTITY_COLORS)]
            color_idx += 1
            annotation_targets.append((ent_name, etype, color))

    # Index per-frame detections per entity (re-using ground_shot_unified output)
    frame_to_dets: Dict[int, List[Tuple[str, Tuple[int, ...], Tuple[int, int, int]]]] = \
        defaultdict(list)
    for ent_name, etype, color in annotation_targets:
        info = per_entity.get(ent_name)
        if info is None:
            continue
        for det in info["all_detections"]:
            frame_to_dets[det["frame_idx"]].append(
                (ent_name, det["bbox"], color)
            )

    annotated_frames: List[np.ndarray] = []
    n_unannotated = 0
    for fi, fr in enumerate(frames):
        annotated = fr.copy()
        dets = frame_to_dets.get(fi, [])
        if not dets:
            n_unannotated += 1
        else:
            for label, bbox, color in dets:
                annotated = _draw_labeled_bbox(annotated, bbox, label, color)
        annotated_frames.append(annotated)

    grid_path: Optional[str] = None
    action_result: Dict[str, Any] = {
        "criteria": {k: None for k in _INTRA_ACTION_CRITERIA},
        "action_depicted": None,
        "overall": None,
        "n_unannotated": n_unannotated,
        "n_frames": len(annotated_frames),
        "vlm_failed": False,
        "skipped_reason": None,
        "grid_path": None,
    }

    if not action_description:
        action_result["skipped_reason"] = "no_action_description"
    elif not annotated_frames:
        action_result["skipped_reason"] = "no_frames"
    else:
        grid = _build_action_grid(annotated_frames)
        if save_action_grid:
            os.makedirs(artifact_dir, exist_ok=True)
            grid_path = os.path.join(artifact_dir, "action_grid.png")
            try:
                Image.fromarray(grid).save(grid_path)
                action_result["grid_path"] = grid_path
            except Exception as e:
                out["errors"].append(f"save_grid: {e}")
                grid_path = None

        scheduled_objects = scheduled_entities.get("objects", []) or []
        object_in_action = any(_name_in_action(action_description, o)
                               for o in scheduled_objects)
        criteria_lines = []
        criteria_keys = []
        for k, desc in _INTRA_ACTION_CRITERIA.items():
            if k == "object_interaction" and not object_in_action:
                continue
            criteria_lines.append(f"- {k}: {desc}")
            criteria_keys.append(k)
        criteria_json = ", ".join(f'"{k}": N' for k in criteria_keys)

        labels_used = ", ".join(t[0] for t in annotation_targets) or "(no entities labeled)"
        prompt = (
            f"You are evaluating whether a generated video shot depicts an action "
            f"as described. The image below is a temporal grid of "
            f"{len(annotated_frames)} frames sampled across the shot, ordered "
            f"left-to-right, top-to-bottom. Detected entities are highlighted "
            f"with colored labeled boxes when grounding succeeded.\n\n"
            f"Action description: {action_description}\n\n"
            f"Labeled entities in the grid: {labels_used}\n\n"
            f"Score each criterion from 1 to 10 (1=completely fails, "
            f"10=perfect):\n" + "\n".join(criteria_lines) + "\n\n"
            f"Then give an overall action-fidelity score (1-10) and a binary "
            f"verdict on whether the action is clearly depicted.\n\n"
            f"Respond ONLY with JSON:\n"
            f'{{{criteria_json}, "overall": N, "action_depicted": true/false, '
            f'"reason": "brief explanation"}}'
        )

        try:
            result = _vlm_api_call([_encode_crop(grid)], prompt, vlm_model)
        except Exception as e:
            result = None
            out["errors"].append(f"action_vlm: {e}")

        if result:
            for k in criteria_keys:
                if k in result and result[k] is not None:
                    try:
                        action_result["criteria"][k] = float(result[k]) / 10.0
                    except (TypeError, ValueError):
                        pass
            if "overall" in result:
                try:
                    action_result["overall"] = float(result["overall"]) / 10.0
                except (TypeError, ValueError):
                    pass
            action_result["action_depicted"] = bool(result.get("action_depicted", False))
            action_result["reason"] = result.get("reason", "")
        else:
            action_result["vlm_failed"] = True

    out["action"] = action_result
    return out


# ============================================================
# Episode-level evaluation orchestrator
# ============================================================

def evaluate_episode(
    episode_id: str, results_dir: str, script: dict, tier: str,
    pillars: Set[int], device: str = "cuda", vlm_model: str = "gemini-2.5-pro",
    n_frames_per_shot: int = 5, n_boundary_frames: int = 3,
    clip_match_threshold: float = 0.2,
    vbench_evaluator: Optional["VBenchEvaluator"] = None, work_dir: str = "",
    save_action_grids: bool = False,
    llm_concurrency: int = 5,
) -> EpisodeResult:

    episode_dir = os.path.join(results_dir, episode_id)
    if not os.path.isdir(episode_dir):
        logger.warning(f"[Eval] Episode dir not found: {episode_dir}")
        return EpisodeResult(episode_id=episode_id, tier=tier, num_shots=0)

    scenes = script.get("scenes", [])
    entity_descriptions = script.get("entity_descriptions", {})
    entity_schedule = script.get("entity_schedule", {})

    shot_order, shot_videos, shot_prompts = [], {}, {}
    # Build list of MI2V transition pairs from the "cut" field.
    # cut[i] = True means shot i is the START of a new cut (T2V first frame).
    # cut[i] = False means shot i continues from the previous shot (MI2V).
    # So transition from shot i-1 to shot i is MI2V when cut[i] == False.
    mi2v_pairs: List[Tuple[str, str]] = []  # [(prev_shot_key, next_shot_key), ...]

    for scene in scenes:
        sn = scene.get("scene_num", 0)
        cuts = scene.get("cut", [])
        prompts = scene.get("video_prompts", [])
        for i, prompt in enumerate(prompts):
            sk = f"{sn}:{i+1}"
            shot_order.append(sk)
            shot_prompts[sk] = prompt
            vids = sorted(glob.glob(os.path.join(episode_dir, f"{sn:03d}_{i+1:03d}*.mp4")))
            if vids:
                shot_videos[sk] = vids[0]

            # Track MI2V transitions: if cut[i] is False, this shot
            # continues from the previous shot via MI2V
            if i > 0 and i < len(cuts) and not cuts[i]:
                prev_sk = f"{sn}:{i}"
                mi2v_pairs.append((prev_sk, sk))

    logger.info(
        f"[Eval] {episode_id}: {len(shot_order)} shots, "
        f"{len(mi2v_pairs)} MI2V transitions"
    )

    result = EpisodeResult(episode_id=episode_id, tier=tier, num_shots=len(shot_order))

    # ================================================================
    # Pillar 1: VBench (7 dims) — STRICT
    # ================================================================
    if 1 in pillars:
        logger.info(f"[Eval] {episode_id}: Pillar 1 — VBench ({len(shot_order)} shots)")
        if vbench_evaluator is None:
            raise StrictModeError(
                f"Pillar 1 requested but no VBench evaluator was provided "
                f"(episode {episode_id})."
            )
        ew = work_dir or os.path.join(results_dir, ".eval_work", episode_id)
        os.makedirs(ew, exist_ok=True)
        pss, failed_dims = vbench_evaluator.evaluate_episode_shots(
            shot_videos, episode_id, ew
        )

        vba: Dict[str, List[float]] = defaultdict(list)
        n_total_shots = len(shot_order)
        for sk in shot_order:
            sc = pss.get(sk, {d: None for d in VBENCH_DIMENSIONS})
            result.shot_results.append(ShotResult(
                episode_id=episode_id, shot_key=sk,
                video_path=shot_videos.get(sk, ""), vbench_scores=sc,
            ))
            for dim, v in sc.items():
                # Only treat real numbers as scores. None means
                # genuinely-not-computed and is excluded from the mean
                # but counted in n_total. dynamic_degree=0 IS a valid
                # score (static video) and is included.
                if isinstance(v, (int, float)):
                    vba[dim].append(float(v))

        # Build MetricValue per VBench dim, recording n_evaluated/n_failed
        result.vbench_means = {}
        for dim in VBENCH_DIMENSIONS:
            scores = vba.get(dim, [])
            n_eval = len(scores)
            n_fail = n_total_shots - n_eval if dim in failed_dims else 0
            n_skip = n_total_shots - n_eval - n_fail
            result.vbench_means[dim] = MetricValue(
                value=float(np.mean(scores)) if scores else None,
                n_evaluated=n_eval, n_failed=n_fail, n_skipped=n_skip,
            )

    # ================================================================
    # Shared grounding pass (Pillar 2 + Pillar 3 both consume this).
    #
    # Single GroundingDINO pass per (shot, entity, frame). The output
    # gives us:
    #   - canonical EntityCrop per (shot, entity)  → used for cs_face,
    #                                                  cs_object,
    #                                                  intra_*_fidelity,
    #                                                  cross-shot anchor
    #   - all per-frame detections per entity      → used for action grid
    #                                                  annotation
    #   - presence status per entity               → used for intra_*_presence
    # ================================================================
    need_grounding = (2 in pillars) or (3 in pillars)
    ground_per_shot: Dict[str, Dict[str, Any]] = {}
    style_frames: List[np.ndarray] = []
    shot_frames_top: Dict[str, List[np.ndarray]] = {}  # top-3 by sharpness, for scene VLM
    shot_frame_cache: Dict[str, List[np.ndarray]] = {}

    if need_grounding:
        # Preflight: validate the grounding stack BEFORE iterating shots.
        # If either model fails to load, the per-shot grounder would just
        # silently return [] for every entity, producing the (very confusing)
        # symptom of vbench/style/transition working fine while presence,
        # cs_face, cs_object, vlm_face, vlm_object, vlm_scene all come back
        # null/zero. Fail loud here so the cause is obvious.
        try:
            gd = get_grounding_dino(device, strict=True)
            if gd is None:
                raise StrictModeError(
                    "GroundingDINO returned None even in strict mode "
                    "(should never happen — check the loader)."
                )
        except StrictModeError:
            raise
        except Exception as e:
            raise StrictModeError(
                f"Pillar 2/3 require GroundingDINO but it failed to load: {e}\n"
                f"  Check: GDINO_CONFIG (currently {os.environ.get('GDINO_CONFIG', '<unset>')!r}) "
                f"and GDINO_WEIGHTS (currently {os.environ.get('GDINO_WEIGHTS', '<unset>')!r}).\n"
                f"  Both must point to existing files. If your cluster has "
                f"network limits, ensure both files are downloaded to disk."
            )
        try:
            _ = get_clip_model(device)
        except Exception as e:
            raise StrictModeError(
                f"Pillar 2/3 require CLIP but it failed to load: {e}\n"
                f"  Set CLIP_MODEL_PATH (currently "
                f"{os.environ.get('CLIP_MODEL_PATH', '<unset>')!r}) to a "
                f"local clip-vit-base-patch32 directory if HuggingFace "
                f"downloads are blocked."
            )
        try:
            _ = get_dino_model(device)
        except Exception as e:
            raise StrictModeError(
                f"Pillar 2 requires DINOv2 but it failed to load: {e}\n"
                f"  Set DINO_MODEL_PATH (currently "
                f"{os.environ.get('DINO_MODEL_PATH', '<unset>')!r}) to a "
                f"local dinov2-base directory if HuggingFace "
                f"downloads are blocked."
            )

        gthr = {
            "box_threshold": BENCHMARK_CONFIG["gdino_box_threshold"],
            "text_threshold": BENCHMARK_CONFIG["gdino_text_threshold"],
        }
        canonical_root = os.path.join(
            work_dir or results_dir, "eval_artifacts", episode_id, "canonical_crops",
        )
        os.makedirs(canonical_root, exist_ok=True)

        logger.info(
            f"[Eval] {episode_id}: Shared grounding pass "
            f"({len(shot_order)} shots, ~{n_frames_per_shot} frames each)"
        )

        for sk in tqdm(shot_order, desc=f"Ground {episode_id[:18]}",
                       leave=False, unit="shot"):
            vid = shot_videos.get(sk)
            if not vid:
                continue
            try:
                frames = extract_frames(vid, n_frames=n_frames_per_shot, strict=True)
            except StrictModeError as e:
                logger.warning(f"[Eval] {episode_id} {sk}: skip frames: {e}")
                continue
            shot_frame_cache[sk] = frames
            style_frames.append(frames[len(frames) // 2])

            # Rank frames by global sharpness for the scene LLM (location
            # judge). TODO: replace with background-aware sharpness once
            # masks are available (Laplacian only on non-foreground regions).
            sharps = []
            for f in frames:
                gray = np.mean(f, axis=2).astype(np.float64)
                sharps.append(float(np.diff(gray, axis=1).var()
                                    + np.diff(gray, axis=0).var()))
            ranked = sorted(range(len(frames)), key=lambda i: sharps[i], reverse=True)
            shot_frames_top[sk] = [frames[i] for i in ranked[:3]]

            shot_canon_dir = os.path.join(canonical_root, sk.replace(":", "_"))
            sched = entity_schedule.get(sk, {})
            ground_per_shot[sk] = ground_shot_unified(
                shot_key=sk,
                frames=frames,
                scheduled_entities=sched,
                entity_descriptions=entity_descriptions,
                artifact_dir=shot_canon_dir,
                device=device,
                grounding_thresholds=gthr,
                clip_match_threshold=clip_match_threshold,
            )

        n_total_calls = sum(g["n_grounding_calls"] for g in ground_per_shot.values())
        n_present = sum(
            sum(1 for v in g["per_entity"].values() if v["status"] == "present")
            for g in ground_per_shot.values()
        )
        n_total_targets = sum(
            len(g["per_entity"]) for g in ground_per_shot.values()
        )
        logger.info(
            f"[Eval] {episode_id}: grounded {n_total_calls} calls, "
            f"{n_present}/{n_total_targets} entity-shot slots confidently present"
        )

        # Defense in depth: if grounding ran but found nothing for any
        # scheduled entity in any shot, something is wrong (e.g. CLIP is
        # rejecting everything because the model loaded from a corrupted
        # path). Flag this loudly — strict mode would raise, partial mode
        # warns.
        if n_total_calls > 0 and n_present == 0 and n_total_targets > 0:
            msg = (
                f"[Eval] {episode_id}: grounding ran ({n_total_calls} calls "
                f"across {n_total_targets} entity-shot slots) but "
                f"ZERO entities passed the CLIP threshold "
                f"({BENCHMARK_CONFIG['clip_match_threshold']:.2f}). "
                f"Likely causes: (a) CLIP model loaded from wrong checkpoint, "
                f"(b) GroundingDINO weights mismatched to its config, "
                f"(c) all scheduled entity descriptions are too generic to "
                f"match in this episode. Check artifact dir for canonical "
                f"crops to diagnose."
            )
            logger.error(msg)

    # Per-shot per-entity fidelity scores — populated by Pillar 3 (if it
    # runs) and consumed by Pillar 2's cross-shot fidelity gate. Empty
    # dict means "no gating applied" (Pillar 2 admits everything).
    # Layout: {(shot_key, entity_name): fidelity_overall in [0,1] or None}
    per_shot_fidelity: Dict[Tuple[str, str], Optional[float]] = {}

    # ================================================================
    # Pillar 3: Intra-shot prompt-following alignment
    #   - Did each scheduled entity actually appear?  (presence)
    #   - When present, does it match its registry?    (fidelity)
    #   - Does the shot depict the action described?   (action)
    # Consumes the shared grounding output (no new GroundingDINO calls).
    # ================================================================
    if 3 in pillars:
        logger.info(
            f"[Eval] {episode_id}: Pillar 3 — Intra-shot alignment "
            f"({len(shot_order)} shots)"
        )

        # Aggregator buckets — one entry per shot per metric, then averaged.
        agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"vals": [], "n_failed": 0, "n_skipped": 0}
        )
        artifact_root = os.path.join(work_dir or results_dir, "eval_artifacts",
                                     episode_id, "intra_shot")
        os.makedirs(artifact_root, exist_ok=True)

        scene_actions: Dict[int, List[Optional[str]]] = {}
        for scene in scenes:
            sn = scene.get("scene_num", 0)
            scene_actions[sn] = scene.get(
                "action_descriptions",
                [None] * len(scene.get("video_prompts", [])),
            )

        # Parallel shot evaluation: each shot's `evaluate_intra_shot_alignment`
        # runs independently and makes its own LLM calls. The rotating
        # client's per-key queue serializes concurrent calls naturally,
        # so we get a clean ~N_keys speedup without rate-limit issues.
        from concurrent.futures import ThreadPoolExecutor

        def _eval_shot(sk: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            ground_result = ground_per_shot.get(sk)
            frames = shot_frame_cache.get(sk)
            if not ground_result or not frames:
                return (sk, None)
            sn_str, shot_str = sk.split(":")
            sn, shot_i = int(sn_str), int(shot_str) - 1
            actions = scene_actions.get(sn, [])
            action_desc = actions[shot_i] if shot_i < len(actions) else None
            sched = entity_schedule.get(sk, {})
            shot_artifacts = os.path.join(artifact_root, sk.replace(":", "_"))
            try:
                ev = evaluate_intra_shot_alignment(
                    shot_key=sk,
                    ground_result=ground_result,
                    frames=frames,
                    action_description=action_desc or "",
                    scheduled_entities=sched,
                    artifact_dir=shot_artifacts,
                    vlm_model=BENCHMARK_CONFIG["vlm_model_intra_fidelity"],
                    save_action_grid=save_action_grids,
                )
                return (sk, ev)
            except Exception as e:
                logger.warning(f"[Pillar3] {sk}: failed: {e}")
                return (sk, None)

        # Fan shots out to threads. We process results in shot_order so
        # the per-shot _eval.json saves and metric aggregations are
        # deterministic.
        shot_evals: Dict[str, Optional[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=llm_concurrency) as ex:
            for sk_done, ev in tqdm(
                ex.map(_eval_shot, shot_order),
                total=len(shot_order),
                desc=f"Pillar 3 {episode_id[:18]}",
                leave=False, unit="shot",
            ):
                shot_evals[sk_done] = ev

        # Sequential aggregation in deterministic shot order
        for sk in shot_order:
            ev = shot_evals.get(sk)
            if ev is None:
                continue
            shot_artifacts = os.path.join(artifact_root, sk.replace(":", "_"))

            # Persist rich per-shot audit JSON
            try:
                os.makedirs(shot_artifacts, exist_ok=True)
                with open(os.path.join(shot_artifacts, "_eval.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(ev, f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                logger.warning(f"[Pillar3] {sk}: audit json save failed: {e}")

            # ---- Presence: per-type fraction ----
            for etype, presence_metric in [
                ("character", "intra_character_presence"),
                ("object",    "intra_object_presence"),
                ("location",  "intra_location_presence"),
            ]:
                p = ev["presence"].get(etype, {})
                n_total = p.get("n_total", 0)
                if n_total > 0:
                    frac = p["n_present"] / n_total
                    agg[presence_metric]["vals"].append(frac)
                # else: shot has no entities of this type — skipped, NOT 0.

            # ---- Per-entity fidelity ----
            for etype, name_in_metric, criteria in [
                ("character", "face",     _INTRA_FIDELITY_CRITERIA["character"]),
                ("object",    "object",   _INTRA_FIDELITY_CRITERIA["object"]),
                ("location",  "location", _INTRA_FIDELITY_CRITERIA["location"]),
            ]:
                fid_per_ent = ev["fidelity"].get(etype, {})
                if not fid_per_ent:
                    continue
                # Record per-shot per-entity fidelity for Pillar 2's
                # cross-shot fidelity gate. None values are passed through
                # as-is (Pillar 2 treats None as "unknown" → admit by default,
                # don't penalize for missing VLM judgments).
                for ent_name, fid_info in fid_per_ent.items():
                    per_shot_fidelity[(sk, ent_name)] = fid_info.get("overall")

                overalls = [f["overall"] for f in fid_per_ent.values()
                            if f["overall"] is not None]
                if overalls:
                    agg[f"intra_{name_in_metric}_fidelity"]["vals"].append(
                        float(np.mean(overalls))
                    )
                for crit_key in criteria.keys():
                    crit_scores = [f["criteria"].get(crit_key)
                                   for f in fid_per_ent.values()]
                    crit_scores = [s for s in crit_scores if s is not None]
                    if crit_scores:
                        suffix = ("face"
                                  if (name_in_metric == "face" and crit_key == "face")
                                  else crit_key)
                        agg[f"intra_{name_in_metric}_{suffix}"]["vals"].append(
                            float(np.mean(crit_scores))
                        )

            # ---- Action fidelity ----
            act = ev["action"]
            if act and act.get("skipped_reason") is None and not act.get("vlm_failed"):
                if act["overall"] is not None:
                    agg["intra_action_overall"]["vals"].append(act["overall"])
                if act["action_depicted"] is not None:
                    agg["intra_action_depicted"]["vals"].append(
                        1.0 if act["action_depicted"] else 0.0
                    )
                for ck in ["subject_identity", "subject_action",
                           "object_interaction", "motion_quality"]:
                    v = act["criteria"].get(ck)
                    if v is not None:
                        agg[f"intra_action_{ck}"]["vals"].append(v)
            elif act and act.get("vlm_failed"):
                agg["intra_action_overall"]["n_failed"] += 1
            elif act and act.get("skipped_reason"):
                agg["intra_action_overall"]["n_skipped"] += 1

        # Materialize into MetricValues
        for metric_name, bucket in agg.items():
            vals = bucket["vals"]
            result.metrics[metric_name] = MetricValue(
                value=float(np.mean(vals)) if vals else None,
                n_evaluated=len(vals),
                n_failed=bucket["n_failed"],
                n_skipped=bucket["n_skipped"],
            )

    # ================================================================
    # Pillar 2: Cross-shot consistency
    # Consumes the shared grounding output. The canonical crop saved
    # by ground_shot_unified IS the same crop used here for cs_face /
    # cs_object DINOv2 comparison and for VLM-Judge cross-shot.
    # ================================================================
    if 2 in pillars:
        logger.info(f"[Eval] {episode_id}: Pillar 2 — Inter-shot consistency")

        # Build the deduped dict from canonical crops. Each (shot, entity)
        # contributes at most one EntityCrop — the canonical one with
        # selection score = clip × sharpness × area.
        # Only entities whose status == "present" contribute. Weak/absent
        # entities are excluded from cross-shot comparison entirely (they
        # had no usable detection in this shot).
        #
        # Cross-shot fidelity gate: if Pillar 3 ran and produced a per-shot
        # fidelity score for this entity, AND that score is below
        # `cross_shot_fidelity_gate_threshold`, the appearance is rejected
        # from the cross-shot pool. This prevents a degenerate failure mode
        # where a model that renders the wrong entity in many shots gets
        # rewarded with high "consistency" (because the wrong renderings
        # happen to look similar to each other, e.g. all-static frames).
        # The audit JSON records how many appearances got gated so reviewers
        # can see the threshold's effect.
        gate_thr = BENCHMARK_CONFIG["cross_shot_fidelity_gate_threshold"]
        deduped: Dict[str, List[EntityCrop]] = defaultdict(list)
        n_appearances_total = 0
        n_appearances_gated_out = 0
        n_appearances_gate_unknown = 0
        for sk in shot_order:
            g = ground_per_shot.get(sk)
            if not g:
                continue
            for ent_name, info in g["per_entity"].items():
                if info["status"] != "present" or info["canonical"] is None:
                    continue
                n_appearances_total += 1
                fid = per_shot_fidelity.get((sk, ent_name))
                if gate_thr > 0 and fid is not None:
                    if fid < gate_thr:
                        n_appearances_gated_out += 1
                        continue
                elif gate_thr > 0 and fid is None and per_shot_fidelity:
                    # Pillar 3 ran but produced no fidelity for this pair
                    # (e.g. VLM call failed). Admit by default but flag so
                    # the audit shows how often we couldn't gate.
                    n_appearances_gate_unknown += 1
                deduped[ent_name].append(info["canonical"])

        # Re-key to dict (defaultdict not strictly needed downstream)
        deduped = {k: v for k, v in deduped.items()}

        if gate_thr > 0 and per_shot_fidelity:
            logger.info(
                f"[Eval] {episode_id}: cross-shot fidelity gate "
                f"(thr={gate_thr:.2f}): "
                f"admitted {n_appearances_total - n_appearances_gated_out}/"
                f"{n_appearances_total} appearances "
                f"({n_appearances_gated_out} gated out, "
                f"{n_appearances_gate_unknown} admitted with unknown fidelity)"
            )
            # Surface in the result for auditability
            result.metrics["_meta_cross_shot_gate"] = MetricValue(
                value=float(n_appearances_total - n_appearances_gated_out),
                n_evaluated=n_appearances_total,
                n_failed=n_appearances_gate_unknown,
                n_skipped=n_appearances_gated_out,
            )

        # ---- DINOv2 metrics (centroid-based) ----
        # No cs_scene: location DINOv2 crops include foreground characters,
        # which makes the metric unfair. Locations are only judged by VLM.
        # Save the most-representative crop per entity (closest to the
        # centroid) for audit. The metric does NOT depend on this choice
        # — it's just the most useful single image to show a reviewer.
        rep_dir = os.path.join(work_dir or results_dir, "eval_artifacts",
                               episode_id, "cross_shot_representatives")
        face_score, face_per_ent, face_n = compute_cross_shot_consistency(
            deduped, "character",
            shot_order=shot_order,
            representative_save_dir=rep_dir,
        )
        obj_score, obj_per_ent, obj_n = compute_cross_shot_consistency(
            deduped, "object",
            shot_order=shot_order,
            representative_save_dir=rep_dir,
        )
        result.metrics["cs_face"] = MetricValue(
            value=face_score, n_evaluated=face_n, n_failed=0, n_skipped=0,
        )
        result.metrics["cs_object"] = MetricValue(
            value=obj_score, n_evaluated=obj_n, n_failed=0, n_skipped=0,
        )
        result.per_entity_scores.update({f"face_{k}": v for k, v in face_per_ent.items()})
        result.per_entity_scores.update({f"object_{k}": v for k, v in obj_per_ent.items()})

        # ---- CS-Transition (boundary only) ----
        # MI2V pairs only — scene cuts are intentional discontinuity, so
        # `mi2v_pairs` is already filtered to (prev, next) where
        # cut[next] == False. The "boundary" version compares the immediate
        # last-frame-of-prev with first-frame-of-next, which is the most
        # diagnostic single number for MI2V continuity.
        # We dropped `cs_transition` (the 3-pair-average variant) and
        # `cs_style` (DINOv2 Gram-matrix style) for the final benchmark —
        # both were highly redundant with other signals and noisy.
        tscs_boundary, n_trans_failed = [], 0
        for prev_sk, next_sk in mi2v_pairs:
            vp, vn = shot_videos.get(prev_sk), shot_videos.get(next_sk)
            if not vp or not vn:
                logger.warning(
                    f"[Eval] {episode_id}: MI2V pair ({prev_sk}, {next_sk}) "
                    f"missing video; vp={bool(vp)}, vn={bool(vn)}"
                )
                n_trans_failed += 1
                continue
            try:
                # n_boundary=1 ⇒ single pair (last frame of prev, first of next)
                tscs_boundary.append(compute_transition_score(vp, vn, 1, device))
            except StrictModeError as e:
                logger.warning(f"[Eval] {episode_id}: transition pair failed: {e}")
                n_trans_failed += 1
        result.metrics["cs_transition_boundary"] = MetricValue(
            value=float(np.mean(tscs_boundary)) if tscs_boundary else None,
            n_evaluated=len(tscs_boundary),
            n_failed=n_trans_failed,
        )
        logger.info(
            f"[Eval] {episode_id}: transition computed for "
            f"{len(tscs_boundary)}/{len(mi2v_pairs)} MI2V pairs "
            f"({n_trans_failed} failed)"
        )

        # ---- LLM judge: characters and objects (pairwise vs centroid-representative) ----
        # We use anchor-vs-each pairwise judging because Gemini's
        # set-counting (outlier_indices) is unreliable for N>3 — it
        # sometimes returns out-of-range indices or miscounts. Pairwise
        # calls have clean binary semantics ("are these two the same?")
        # that Gemini handles consistently.
        #
        # CRITICAL: anchor is NOT the first appearance — it's the
        # centroid-representative crop computed in compute_cross_shot_consistency
        # (the crop closest to the DINOv2 centroid). This avoids two biases
        # that would have plagued first-shot anchoring:
        #   (1) ordering dependency: same crops in different shot order →
        #       different anchor → different metric value
        #   (2) T2V-vs-MI2V bias: first shot is generated by T2V while
        #       continuation shots use MI2V; pinning anchor to first shot
        #       systematically compares MI2V outputs against a T2V output
        #
        # The representative crop is already saved to disk during DINOv2
        # cross-shot, so we read it from `representative_shot`/`representative_crop_path`
        # in per_entity_scores. Audit chain: same image used for centroid
        # selection AND VLM anchor, so reviewers see the same crop in both
        # places.
        shot_pos_map = {sk: i for i, sk in enumerate(shot_order)}
        for etype, prefix, criteria_keys in [
            ("character", "llm_face",   ["face", "hair", "clothing", "build"]),
            ("object",    "llm_object", ["shape", "color_texture", "proportions", "details"]),
        ]:
            n_eval, n_fail, n_skip = 0, 0, 0
            correct, sims = 0, []
            crit_sims: Dict[str, List[float]] = {k: [] for k in criteria_keys}

            for name, crops in deduped.items():
                typed = [c for c in crops if c.entity_type == etype]
                if len(typed) < 2:
                    n_skip += 1
                    continue

                # Find the representative crop chosen by the centroid step.
                # `result.per_entity_scores` uses the prefix "face" for
                # characters (legacy convention from cs_face) and "object"
                # for objects, NOT the entity type. We mirror that here so
                # the lookup actually matches.
                key_prefix = "face" if etype == "character" else etype
                ent_key = f"{key_prefix}_{name}"
                ent_info = result.per_entity_scores.get(ent_key, {})
                rep_shot = ent_info.get("representative_shot")
                anchor = None
                if rep_shot:
                    for c in typed:
                        if c.shot_key == rep_shot:
                            anchor = c
                            break
                if anchor is None:
                    typed_sorted_for_fallback = sorted(
                        typed, key=lambda c: shot_pos_map.get(c.shot_key, 0),
                    )
                    anchor = typed_sorted_for_fallback[0]
                    logger.warning(
                        f"[VLM-pairwise] {name}: no representative_shot in "
                        f"per_entity_scores; falling back to first-shot anchor"
                    )

                # Compare anchor vs each OTHER crop (skip the anchor itself).
                # Order comparisons by shot position for deterministic per-pair logging.
                comparisons = sorted(
                    [c for c in typed if c is not anchor],
                    key=lambda c: shot_pos_map.get(c.shot_key, 0),
                )
                if not comparisons:
                    n_skip += 1
                    continue

                # Load anchor crop once
                try:
                    anchor_img = np.array(Image.open(anchor.crop_path))
                except Exception as e:
                    logger.warning(
                        f"[LLM-pairwise] {name}: failed to load anchor crop "
                        f"{anchor.crop_path!r}: {e}"
                    )
                    n_fail += len(comparisons)
                    continue

                # Concurrent pair evaluation: each (anchor, later) pair
                # is independent, so we fan them out across threads.
                # The rotating client's queue serializes per-key access,
                # so this respects rate limits naturally.
                def _judge_pair(later: EntityCrop) -> Tuple[EntityCrop, Optional[Dict[str, Any]]]:
                    try:
                        later_img = np.array(Image.open(later.crop_path))
                    except Exception as e:
                        logger.warning(
                            f"[LLM-pairwise] {name}: failed to load later crop "
                            f"{later.crop_path!r}: {e}"
                        )
                        return (later, None)
                    verdict = vlm_judge_entity(
                        anchor_img, later_img,
                        anchor.entity_description, name, etype,
                        BENCHMARK_CONFIG["vlm_model_entity"],
                    )
                    if verdict.get("reason") == "VLM call failed":
                        return (later, None)
                    return (later, verdict)

                from concurrent.futures import ThreadPoolExecutor
                pair_results: List[Tuple[EntityCrop, Optional[Dict[str, Any]]]] = []
                with ThreadPoolExecutor(max_workers=llm_concurrency) as ex:
                    pair_results = list(ex.map(_judge_pair, comparisons))

                # Sort by shot position so the audit `vlm_per_pair_sims`
                # list is deterministic (threading completion order is not).
                pair_results.sort(key=lambda r: shot_pos_map.get(r[0].shot_key, 0))

                per_pair_sims: List[Dict[str, Any]] = []
                pair_n_eval = 0
                pair_n_correct = 0
                for later, verdict in pair_results:
                    if verdict is None:
                        n_fail += 1
                        continue
                    n_eval += 1
                    pair_n_eval += 1
                    if verdict["is_same"]:
                        correct += 1
                        pair_n_correct += 1
                    sims.append(verdict["similarity"])
                    for ck in criteria_keys:
                        v = verdict.get("criteria", {}).get(ck)
                        if v is not None:
                            crit_sims[ck].append(float(v))
                    per_pair_sims.append({
                        "later_shot": later.shot_key,
                        "similarity": verdict["similarity"],
                        "is_same": verdict["is_same"],
                        "criteria": verdict.get("criteria", {}),
                    })

                # Surface per-entity pairwise breakdown in audit
                if ent_key in result.per_entity_scores:
                    result.per_entity_scores[ent_key]["vlm_anchor_shot"] = anchor.shot_key
                    result.per_entity_scores[ent_key]["vlm_anchor_clip_sim"] = float(anchor.clip_text_sim)
                    result.per_entity_scores[ent_key]["vlm_pair_count"] = pair_n_eval
                    result.per_entity_scores[ent_key]["vlm_pair_accuracy"] = (
                        pair_n_correct / pair_n_eval if pair_n_eval > 0 else None
                    )
                    result.per_entity_scores[ent_key]["vlm_per_pair_sims"] = per_pair_sims

            # accuracy: fraction of PAIRS (anchor-vs-each) Gemini said is_same
            result.metrics[f"{prefix}_accuracy"] = MetricValue(
                value=(correct / n_eval) if n_eval > 0 else None,
                n_evaluated=n_eval, n_failed=n_fail, n_skipped=n_skip,
            )
            # mean_score: mean similarity across all pairs
            result.metrics[f"{prefix}_mean_score"] = MetricValue(
                value=float(np.mean(sims)) if sims else None,
                n_evaluated=n_eval, n_failed=n_fail, n_skipped=n_skip,
            )
            # Per-criterion: same denominator as mean_score; missing
            # criterion in a Gemini response is recorded as a per-criterion
            # skip, NOT as a 0.
            for ck in criteria_keys:
                vals = crit_sims[ck]
                result.metrics[f"{prefix}_{ck}"] = MetricValue(
                    value=float(np.mean(vals)) if vals else None,
                    n_evaluated=len(vals),
                    n_skipped=n_skip + max(0, n_eval - len(vals)),
                    n_failed=n_fail,
                )

        # ---- VLM-Judge: scenes/locations (N-at-once, full frames) ----
        # Same set-judge pattern as characters/objects, but uses up to 2
        # full frames per shot (selected by sharpness, may include
        # foreground characters — Gemini is told to ignore them).
        scene_crit_keys = ["layout", "color_mood", "landmarks", "perspective"]
        scene_n_eval, scene_n_fail, scene_n_skip = 0, 0, 0
        n_consistent_locations = 0
        scene_consistencies: List[float] = []
        scene_crit_sims: Dict[str, List[float]] = {k: [] for k in scene_crit_keys}

        for name, crops in deduped.items():
            typed = [c for c in crops if c.entity_type == "location"]
            if len(typed) < 2:
                scene_n_skip += 1
                continue
            typed_sorted = sorted(typed, key=lambda c: shot_pos_map.get(c.shot_key, 0))
            # Per-shot frame lists (top-3 sharpest already cached)
            shot_frame_lists = [shot_frames_top.get(c.shot_key, []) for c in typed_sorted]
            shot_keys_for_set = [c.shot_key for c in typed_sorted]
            desc = typed_sorted[0].entity_description

            verdict = vlm_judge_scene_set(
                shot_frames=shot_frame_lists,
                shot_keys=shot_keys_for_set,
                description=desc,
                location_name=name,
                model=BENCHMARK_CONFIG["vlm_model_scene"],
                max_shots=BENCHMARK_CONFIG["vlm_n_max_images_per_call"] // 2,
                max_frames_per_shot=2,
            )
            if verdict.get("vlm_failed"):
                scene_n_fail += 1
                continue
            scene_n_eval += 1
            if verdict.get("all_consistent"):
                n_consistent_locations += 1
            if verdict.get("consistency") is not None:
                scene_consistencies.append(verdict["consistency"])
            for ck in scene_crit_keys:
                v = verdict.get("criteria", {}).get(ck)
                if v is not None:
                    scene_crit_sims[ck].append(float(v))

        result.metrics["llm_scene_accuracy"] = MetricValue(
            value=(n_consistent_locations / scene_n_eval) if scene_n_eval > 0 else None,
            n_evaluated=scene_n_eval, n_failed=scene_n_fail, n_skipped=scene_n_skip,
        )
        result.metrics["llm_scene_mean_score"] = MetricValue(
            value=float(np.mean(scene_consistencies)) if scene_consistencies else None,
            n_evaluated=len(scene_consistencies),
            n_failed=scene_n_fail,
            n_skipped=scene_n_skip + (scene_n_eval - len(scene_consistencies)),
        )
        for ck in scene_crit_keys:
            vals = scene_crit_sims[ck]
            result.metrics[f"llm_scene_{ck}"] = MetricValue(
                value=float(np.mean(vals)) if vals else None,
                n_evaluated=len(vals),
                n_skipped=scene_n_skip + max(0, scene_n_eval - len(vals)),
                n_failed=scene_n_fail,
            )

        # NOTE: LLM-judged style metrics and cs_style were dropped from
        # the benchmark suite. The signal was noisy and overlapped with
        # other cross-shot metrics; we prefer fewer, more interpretable
        # metrics for the published numbers.

        # ---- Gap decay (all-pairs, characters + objects) ----
        result.gap_decay_data = compute_gap_decay(
            deduped, shot_order,
            entity_types=("character", "object"),
        )

    # Clean up VBench scratch
    if work_dir:
        import shutil
        for subdir in ["vbench_batch_", "vbench_out_"]:
            ep_work = os.path.join(work_dir, f"{subdir}{episode_id}")
            if os.path.isdir(ep_work):
                shutil.rmtree(ep_work, ignore_errors=True)

    return result


# ============================================================
# Aggregation + reporting (runs automatically after evaluation)
# ============================================================

ALL_METRICS = [
    # ---- Pillar 1B: Intra-shot prompt-following ----
    # Presence (fraction of scheduled entities actually rendered)
    "intra_character_presence", "intra_object_presence", "intra_location_presence",
    # Per-criteria fidelity for characters
    "intra_face_fidelity", "intra_face_face", "intra_face_hair",
    "intra_face_clothing", "intra_face_build",
    # Per-criteria fidelity for objects
    "intra_object_fidelity", "intra_object_shape", "intra_object_color_texture",
    "intra_object_proportions", "intra_object_details",
    # Per-criteria fidelity for locations
    "intra_location_fidelity", "intra_location_layout",
    "intra_location_color_mood", "intra_location_landmarks",
    "intra_location_perspective",
    # Action fidelity
    "intra_action_overall", "intra_action_subject_identity",
    "intra_action_subject_action", "intra_action_object_interaction",
    "intra_action_motion_quality", "intra_action_depicted",
    # ---- Pillar 2: Cross-shot consistency (DINOv2) ----
    "cs_face", "cs_object", "cs_transition_boundary",
    # ---- Pillar 2: Cross-shot consistency (LLM-judge) ----
    # Per-criteria for characters
    "llm_face_accuracy", "llm_face_mean_score",
    "llm_face_face", "llm_face_hair", "llm_face_clothing", "llm_face_build",
    # Per-criteria for objects
    "llm_object_accuracy", "llm_object_mean_score",
    "llm_object_shape", "llm_object_color_texture",
    "llm_object_proportions", "llm_object_details",
    # Per-criteria for locations
    "llm_scene_accuracy", "llm_scene_mean_score",
    "llm_scene_layout", "llm_scene_color_mood",
    "llm_scene_landmarks", "llm_scene_perspective",
]


# Subgroups for the report writer — keep related metrics together.
INTRA_SHOT_METRICS = [m for m in ALL_METRICS if m.startswith("intra_")]
CROSS_SHOT_DINO_METRICS = ["cs_face", "cs_object", "cs_transition_boundary"]
CROSS_SHOT_LLM_METRICS = [m for m in ALL_METRICS if m.startswith("llm_")]


def aggregate_results(
    episode_results: List[EpisodeResult],
    split_ids: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Aggregate per-episode metrics into per-tier statistics.

    STRICT contract:
      - A metric with value=None contributes NOTHING to mean/std.
        It does NOT silently become 0.
      - Aggregate reports n_episodes_total vs n_episodes_evaluated for
        each metric so the reader sees coverage explicitly.
      - "all" tier is union of easy+medium+hard, not a separate split.
    """
    report = {}
    id_to_r = {r.episode_id: r for r in episode_results}

    # Build "all" by union if not explicitly provided
    explicit_all = split_ids.get("all", [])
    if not explicit_all:
        explicit_all = (split_ids.get("easy", [])
                        + split_ids.get("medium", [])
                        + split_ids.get("hard", []))

    tier_episode_lists = {
        "all": explicit_all,
        "easy": split_ids.get("easy", []),
        "medium": split_ids.get("medium", []),
        "hard": split_ids.get("hard", []),
    }

    def _s(vals: List[float], n_episodes_total: int) -> Dict[str, Any]:
        """Summary stats with explicit coverage tracking."""
        if not vals:
            return {
                "value": None,
                "n_episodes_evaluated": 0,
                "n_episodes_total": n_episodes_total,
            }
        a = np.array(vals, dtype=np.float64)
        return {
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4),
            "n_episodes_evaluated": len(vals),
            "n_episodes_total": n_episodes_total,
        }

    for tier, ep_list in tier_episode_lists.items():
        trs = [id_to_r[e] for e in ep_list if e in id_to_r]
        if not trs:
            continue
        n_total = len(trs)

        # VBench: aggregate per dim, ignoring None values
        vba: Dict[str, List[float]] = defaultdict(list)
        vbench_n_failed: Dict[str, int] = defaultdict(int)
        for r in trs:
            for d, mv in r.vbench_means.items():
                if mv.value is not None:
                    vba[d].append(mv.value)
                if mv.n_failed > 0:
                    vbench_n_failed[d] += 1

        # Pillar 2: aggregate across episodes, ignoring None values.
        # NOTE: a metric value of 0.0 IS valid (e.g., VLM said "different"
        # for every pair) and IS included. Only None is excluded.
        per_metric_vals: Dict[str, List[float]] = {}
        for m in ALL_METRICS:
            vals = []
            for r in trs:
                mv = r.metrics.get(m)
                if mv is not None and mv.value is not None:
                    vals.append(mv.value)
            per_metric_vals[m] = vals

        report[tier] = {
            "num_episodes": n_total,
            "num_shots": sum(r.num_shots for r in trs),
            "vbench": {
                d: dict(_s(vba.get(d, []), n_total),
                        n_episodes_with_failures=vbench_n_failed.get(d, 0))
                for d in VBENCH_DIMENSIONS
            },
            **{m: _s(per_metric_vals[m], n_total) for m in ALL_METRICS},
        }

        gd = []
        for r in trs:
            gd.extend(r.gap_decay_data)
        if gd:
            # Combined curve (all entity types) — preserve for backward compat
            bins_all = defaultdict(list)
            # Per-type curves — characters and objects often decay differently
            bins_by_type: Dict[str, Dict[int, List[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for d in gd:
                gap = int(d["gap"])
                sim = d["similarity"]
                bins_all[gap].append(sim)
                etype = d.get("entity_type", "character")
                bins_by_type[etype][gap].append(sim)

            report[tier]["gap_decay_curve"] = {
                str(g): {"mean_sim": round(float(np.mean(s)), 4),
                         "std_sim": round(float(np.std(s)), 4),
                         "n_pairs": len(s)}
                for g, s in sorted(bins_all.items())
            }
            report[tier]["gap_decay_curve_by_type"] = {
                etype: {
                    str(g): {"mean_sim": round(float(np.mean(s)), 4),
                             "std_sim": round(float(np.std(s)), 4),
                             "n_pairs": len(s)}
                    for g, s in sorted(bins.items())
                }
                for etype, bins in bins_by_type.items()
            }

    return report


def write_report(report: Dict[str, Any], method_name: str, out_path: str):
    L = [f"# Evaluation Report: {method_name}", ""]

    # Manifest summary at top — critical for reproducibility audits
    if "_manifest" in report:
        m = report["_manifest"]
        L.append("## Run manifest")
        L.append("")
        L.append(f"- Method: `{m.get('method_name')}`")
        L.append(f"- Timestamp (UTC): {m.get('timestamp_utc')}")
        L.append(f"- Strict: {m.get('strict')} | Allow partial: {m.get('allow_partial')}")
        L.append(f"- Pillars: {m.get('pillars')}")
        cfg = m.get("config", {})
        L.append(f"- Config: gdino_box={cfg.get('gdino_box_threshold')}, "
                 f"gdino_text={cfg.get('gdino_text_threshold')}, "
                 f"clip_thresh={cfg.get('clip_match_threshold')}, "
                 f"n_frames={cfg.get('n_frames_per_shot')}")
        L.append(f"- VLM models: entity={cfg.get('vlm_model_entity')}, "
                 f"scene={cfg.get('vlm_model_scene')}, style={cfg.get('vlm_model_style')}")
        libs = m.get("libraries", {})
        L.append(f"- Libraries: vbench={libs.get('vbench')}, "
                 f"transformers={libs.get('transformers')}, torch={libs.get('torch')}")
        L.append("")
        L.append("Two runs are comparable IFF their `run_manifest.json` files match. "
                 "See `--diagnose_vbench` if any VBench coverage row shows failures.")
        L.append("")

    def _row(name: str, s: Dict[str, Any]) -> str:
        if s.get("value", "missing") is None or "mean" not in s:
            cov = f"0/{s.get('n_episodes_total','?')}"
            return f"| {name} | --- | --- | --- | --- | {cov} |"
        cov = f"{s.get('n_episodes_evaluated','?')}/{s.get('n_episodes_total','?')}"
        return (f"| {name} | {s['mean']} | {s.get('std','-')} | "
                f"{s.get('min','-')} | {s.get('max','-')} | {cov} |")

    for tier in ["all", "easy", "medium", "hard"]:
        r = report.get(tier)
        if not r:
            continue
        L.append(f"## {tier.title()} ({r['num_episodes']} episodes, {r['num_shots']} shots)")
        L.append("")

        L.append("### Cross-shot consistency (DINOv2)")
        L.append("")
        L.append("| Metric | Mean | Std | Min | Max | Coverage |")
        L.append("|--------|------|-----|-----|-----|----------|")
        for m in ["cs_face", "cs_object", "cs_transition_boundary"]:
            L.append(_row(m, r.get(m, {})))
        L.append("")

        L.append("### Cross-shot consistency (LLM-judge)")
        L.append("")
        L.append("| Metric | Mean | Std | Min | Max | Coverage |")
        L.append("|--------|------|-----|-----|-----|----------|")
        for m in ["llm_face_accuracy", "llm_face_mean_score",
                   "llm_object_accuracy", "llm_object_mean_score",
                   "llm_scene_accuracy", "llm_scene_mean_score"]:
            L.append(_row(m, r.get(m, {})))
        L.append("")

        vb = r.get("vbench", {})
        if vb:
            L.append("### VBench intra-shot quality")
            L.append("")
            L.append("VBench dimensions are reported on their CANONICAL scales "
                     "(matching the broader literature). Most are in [0, 1]; "
                     "`imaging_quality` is the raw MUSIQ score on [0, 100]. "
                     "Do NOT normalize before comparing across papers.")
            L.append("")
            L.append("| Dimension | Scale | Mean | Std | Coverage | Episodes with failures |")
            L.append("|-----------|-------|------|-----|----------|------------------------|")
            for d in VBENCH_DIMENSIONS:
                s = vb.get(d, {})
                scale = VBENCH_DIM_SCALE.get(d, "?")
                if s.get("value", "missing") is None or "mean" not in s:
                    L.append(f"| {d} | {scale} | --- | --- | "
                             f"0/{s.get('n_episodes_total','?')} | "
                             f"{s.get('n_episodes_with_failures', 0)} |")
                else:
                    cov = f"{s.get('n_episodes_evaluated','?')}/{s.get('n_episodes_total','?')}"
                    L.append(f"| {d} | {scale} | {s['mean']} | {s.get('std','-')} | "
                             f"{cov} | {s.get('n_episodes_with_failures', 0)} |")
            L.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def diagnose_vbench(cache_dir: str, device: str = "cuda"):
    """Check VBench installation, checkpoints, and run a quick test."""
    print("=" * 60)
    print("VBench Diagnostic")
    print("=" * 60)

    # 1. Check import
    ok, ver = _check_vbench_available()
    print(f"\n1. Import: {'OK (v' + ver + ')' if ok else 'FAILED — pip install vbench'}")
    if not ok:
        return

    # 2. Check cache dir
    print(f"\n2. Cache dir: {cache_dir}")
    if not os.path.isdir(cache_dir):
        print(f"   MISSING — directory does not exist")
        return

    # 3. Check VBench_full_info.json
    info_file = os.path.join(cache_dir, "VBench_full_info.json")
    if os.path.isfile(info_file):
        print(f"   VBench_full_info.json: OK ({os.path.getsize(info_file)} bytes)")
    else:
        # Try to find it from vbench package
        try:
            import vbench
            pkg_dir = os.path.dirname(vbench.__file__)
            pkg_info = os.path.join(pkg_dir, "VBench_full_info.json")
            if os.path.isfile(pkg_info):
                print(f"   VBench_full_info.json: not in cache, but found in package at {pkg_info}")
                print(f"   → Copy it: cp {pkg_info} {cache_dir}/")
            else:
                print(f"   VBench_full_info.json: MISSING (not in cache or package)")
        except Exception:
            print(f"   VBench_full_info.json: MISSING")

    # 4. Check each checkpoint
    print(f"\n3. Checkpoints for custom-video dims:")
    checkpoint_map = {
        "subject_consistency":    [("DINO", "clip_model")],
        "temporal_flickering":    [],
        "motion_smoothness":      [("RAFT", "raft_model")],
        "dynamic_degree":         [("RAFT", "raft_model")],
        "aesthetic_quality":      [("LAION aesthetic", "aesthetic_model"),
                                   ("LAION linear head", "sa_0_4_vit_l_14_linear.pth")],
        "imaging_quality":        [("MUSIQ", "musiq_spaq_ckpt-358bb6af.pth")],
    }

    all_ok = True
    for dim, deps in checkpoint_map.items():
        if not deps:
            print(f"   {dim:28s}: OK (no model needed)")
            continue
        issues = []
        for dep_name, dep_path in deps:
            full_path = os.path.join(cache_dir, dep_path)
            if os.path.exists(full_path):
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    if size < 1000:
                        issues.append(f"{dep_path} too small ({size} bytes, likely corrupted)")
                    else:
                        size_mb = size / (1024 * 1024)
                        # Check if it's a valid zip/pytorch file
                        try:
                            with open(full_path, "rb") as f:
                                header = f.read(4)
                            if header[:2] == b"PK" or header == b"\x80\x02":  # zip or pickle
                                pass  # looks valid
                            issues.append(None)  # mark as checked
                        except Exception:
                            issues.append(f"{dep_path} unreadable")
                elif os.path.isdir(full_path):
                    n_files = sum(1 for _ in os.listdir(full_path))
                    issues.append(None)  # dir exists
            else:
                issues.append(f"{dep_path} MISSING")

        real_issues = [i for i in issues if i is not None]
        if real_issues:
            print(f"   {dim:28s}: ISSUE — {'; '.join(real_issues)}")
            all_ok = False
        else:
            print(f"   {dim:28s}: OK")

    # 5. Try loading VBench
    print(f"\n4. VBench initialization:")
    try:
        from vbench import VBench
        test_out = os.path.join(cache_dir, "_diagnostic_test")
        os.makedirs(test_out, exist_ok=True)
        vb = VBench(device, cache_dir, test_out)
        print(f"   VBench(device={device}, cache={cache_dir}): OK")
        import shutil
        shutil.rmtree(test_out, ignore_errors=True)
    except Exception as e:
        print(f"   VBench init FAILED: {e}")

    # 6. Summary
    print(f"\n{'=' * 60}")
    if all_ok:
        print("All checkpoints present. Ready to run.")
    else:
        print("Some checkpoints have issues. Fix them and re-run.")
        print("\nTo re-download a corrupted checkpoint (e.g., musiq):")
        print("  wget https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth")
        print(f"  mv musiq_spaq_ckpt-358bb6af.pth {cache_dir}/")
    print()


# ============================================================
# CLI
# ============================================================

def _episode_result_to_json(r: EpisodeResult) -> Dict[str, Any]:
    """Serialize EpisodeResult including MetricValue provenance."""
    return {
        "episode_id": r.episode_id,
        "tier": r.tier,
        "num_shots": r.num_shots,
        "vbench_means": {d: mv.to_json() for d, mv in r.vbench_means.items()},
        "metrics": {m: mv.to_json() for m, mv in r.metrics.items()},
        "per_entity_scores": {k: v for k, v in r.per_entity_scores.items()
                              if isinstance(v, dict)},
        "gap_decay_data": r.gap_decay_data,
    }


def _episode_result_from_json(eid: str, tier: str, c: Dict[str, Any]) -> EpisodeResult:
    """Deserialize EpisodeResult, tolerating both new and legacy formats.

    Legacy cached per-episode JSONs may use the `vlm_*` prefix for the
    multimodal-LLM cross-shot metrics; we now publish them as `llm_*`.
    Remap on read so old caches re-aggregate into the new key namespace.
    """
    def _rekey(name: str) -> str:
        return ("llm_" + name[len("vlm_"):]) if name.startswith("vlm_") else name

    r = EpisodeResult(episode_id=eid, tier=tier, num_shots=c.get("num_shots", 0))
    # VBench
    vb = c.get("vbench_means", {})
    r.vbench_means = {d: MetricValue.from_json(v) for d, v in vb.items()}
    # Pillar 2
    if "metrics" in c:
        r.metrics = {_rekey(m): MetricValue.from_json(v)
                     for m, v in c["metrics"].items()}
    else:
        # Legacy format — flat float fields
        r.metrics = {}
        for m in ALL_METRICS:
            if m in c:
                r.metrics[m] = MetricValue.from_json(c[m])
            # accept legacy vlm_* spelling for the same metric
            legacy = "vlm_" + m[len("llm_"):] if m.startswith("llm_") else None
            if legacy and legacy in c:
                r.metrics.setdefault(m, MetricValue.from_json(c[legacy]))
    r.gap_decay_data = c.get("gap_decay_data", [])
    r.per_entity_scores = c.get("per_entity_scores", {})
    return r


def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-shot consistency benchmark")
    parser.add_argument("--results_dir", required=False, default="")
    parser.add_argument("--scripts_dir", required=False, default="")
    parser.add_argument("--split_json", required=False, default="")
    parser.add_argument("--out_dir", required=False, default="")
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--pillars", default="1,2,3",
                        help="Comma-separated pillar IDs to run. "
                             "1=VBench (intra-shot quality), "
                             "2=cross-shot consistency, "
                             "3=intra-shot prompt-following (presence + fidelity + action). "
                             "Default: 1,2,3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vlm_model", default=None,
                        help="Override default VLM model. STRICT mode rejects this.")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Override n_frames_per_shot. STRICT mode rejects this.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vbench_cache_dir", default="")
    parser.add_argument("--diagnose_vbench", action="store_true",
                        help="Run VBench diagnostic check and exit")
    # Strict-mode controls
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Strict mode: no fallbacks, no silent skips. (DEFAULT)")
    parser.add_argument("--no_strict", dest="strict", action="store_false",
                        help="Disable strict mode (allows fallbacks/silent skips). "
                             "Results are NOT comparable to canonical benchmark numbers.")
    parser.add_argument("--allow_partial", action="store_true",
                        help="In strict mode, record per-item failures as null "
                             "instead of aborting the episode. Aggregate still excludes "
                             "nulls from the mean. Use for long distributed runs.")
    parser.add_argument("--allow_config_override", action="store_true",
                        help="Permit CLI overrides of BENCHMARK_CONFIG values. "
                             "Disables strict reproducibility guarantees.")
    parser.add_argument("--save_action_grids", action="store_true",
                        help="Persist Pillar 3 action-grid PNGs to disk for "
                             "manual inspection. Disabled by default — at "
                             "~2500 grids per method run, this can use 100GB+ "
                             "of disk. Use only for debugging a small subset.")
    parser.add_argument("--llm_concurrency", type=int, default=5,
                        help="Number of concurrent LLM calls per episode "
                             "(default 5, matching typical n_keys). The "
                             "rotating client serializes per-key access so "
                             "this is safe to set up to n_keys; setting it "
                             "higher than n_keys gives no benefit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # Diagnostic mode
    if args.diagnose_vbench:
        cache = args.vbench_cache_dir or os.environ.get("VBENCH_CACHE_DIR") or os.path.expanduser("~/.cache/vbench")
        diagnose_vbench(cache, args.device)
        return

    # Normal mode — require the usual args
    if not args.results_dir or not args.scripts_dir or not args.split_json or not args.out_dir:
        parser.error("--results_dir, --scripts_dir, --split_json, and --out_dir are required (unless --diagnose_vbench)")

    # Build effective config: start from BENCHMARK_CONFIG, optionally override.
    # Only treat a CLI value as an "override" when it actually differs from
    # the canonical default — this lets the distributed launcher pass these
    # flags through unconditionally without tripping strict-mode.
    effective_config = dict(BENCHMARK_CONFIG)
    overrides = {}
    if (args.vlm_model is not None
            and args.vlm_model != BENCHMARK_CONFIG["vlm_model_entity"]):
        overrides["vlm_model_entity"] = args.vlm_model
    if (args.n_frames is not None
            and args.n_frames != BENCHMARK_CONFIG["n_frames_per_shot"]):
        overrides["n_frames_per_shot"] = args.n_frames

    if overrides and args.strict and not args.allow_config_override:
        parser.error(
            f"Strict mode rejects CLI config overrides: {overrides}. "
            f"Pass --allow_config_override to permit them (results will not "
            f"be comparable to canonical benchmark numbers), or remove the flags."
        )
    effective_config.update(overrides)

    os.makedirs(args.out_dir, exist_ok=True)
    pillars = set(int(p) for p in args.pillars.split(","))

    with open(args.split_json, encoding="utf-8") as f:
        split_ids = json.load(f)

    episodes = [(eid, tier) for tier in ["easy", "medium", "hard"] for eid in split_ids.get(tier, [])]
    if args.limit > 0:
        episodes = episodes[:args.limit]

    # ---- Build manifest BEFORE evaluation; validate against existing ----
    manifest = build_run_manifest(args.method_name, effective_config)
    manifest["strict"] = args.strict
    manifest["allow_partial"] = args.allow_partial
    manifest["allow_config_override"] = args.allow_config_override
    manifest["pillars"] = sorted(pillars)
    # Record key COUNT only (never the keys themselves). Two runs are
    # comparable regardless of how many keys were used; the count is
    # purely informational so reviewers can see whether 429s were
    # likely (1 key under heavy load) or unlikely (5 keys).
    keys_env = os.environ.get("GEMINI_API_KEYS", "").strip()
    if keys_env:
        n_keys = len([k for k in keys_env.split(",") if k.strip()])
    elif os.environ.get("AZURE_OPENAI_API_KEY", "").strip():
        n_keys = 1
    else:
        n_keys = 0
    manifest["n_vlm_keys"] = n_keys
    manifest_path = os.path.join(args.out_dir, "run_manifest.json")

    # Manifest read/write is shared across distributed shards. We need
    # to handle three race conditions:
    #   (1) Two shards observe no manifest at the same time and both write.
    #       Solved by atomic-replace + skip-if-already-comparable.
    #   (2) One shard reads while another is writing (partial/empty file).
    #       Solved by tolerant reader: small retry loop, fall back to
    #       "not yet present" if all reads fail.
    #   (3) Shards racing to update vlm_call_stats at the end of the run.
    #       Solved by atomic-replace + last-writer-wins (stats are advisory,
    #       not a comparability check input).

    def _read_manifest_tolerant(path: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
        """Read a JSON manifest, retrying briefly if another shard is mid-write."""
        for attempt in range(max_retries):
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    # File exists but is empty (concurrent writer truncated it)
                    time.sleep(0.2 * (attempt + 1))
                    continue
                return json.loads(content)
            except FileNotFoundError:
                return None
            except json.JSONDecodeError:
                time.sleep(0.2 * (attempt + 1))
                continue
        # Couldn't read after retries — treat as "not present" rather than crash;
        # another shard's write will land soon.
        logger.warning(
            f"Could not parse {path} after {max_retries} retries — likely "
            f"a concurrent shard mid-write. Continuing without comparability check."
        )
        return None

    def _write_manifest_atomic(path: str, data: Dict[str, Any]) -> None:
        """Write JSON via tmp file + os.replace so readers never see partial."""
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    if os.path.exists(manifest_path) and args.resume:
        existing_manifest = _read_manifest_tolerant(manifest_path)
        if existing_manifest is not None:
            ok, diffs = manifest_comparable(manifest, existing_manifest)
            if not ok:
                msg = (
                    f"Resuming from incompatible manifest:\n"
                    + "\n".join(f"  - {d}" for d in diffs[:20])
                )
                if args.strict:
                    parser.error(
                        msg + "\n\nResume aborted. Either use a fresh --out_dir, "
                        "or pass --no_strict to override (NOT recommended)."
                    )
                logger.warning(msg)
            else:
                # Already comparable — don't rewrite. This avoids 8 shards
                # racing to overwrite the same file at startup.
                logger.info(f"Manifest already present and comparable; skipping write")
                manifest = existing_manifest

    # Always (re-)write atomically. If we read an existing comparable
    # manifest above, we kept it; this path is for first-shard-to-arrive
    # or for filling in missing fields.
    _write_manifest_atomic(manifest_path, manifest)

    vbench_eval = None
    if 1 in pillars:
        vbench_eval = VBenchEvaluator(
            device=args.device,
            cache_dir=args.vbench_cache_dir,
            strict=args.strict,
        )
        ok, ver = _check_vbench_available()
        if args.strict and not ok:
            parser.error(
                "Strict mode + Pillar 1 requested but VBench is not installed. "
                "Run: pip install vbench. (Or pass --no_strict, but results "
                "won't be comparable.)"
            )
        print(f"VBench: {'v' + ver if ok else 'NOT INSTALLED'}")

    work_dir = os.path.join(args.out_dir, ".work")
    os.makedirs(work_dir, exist_ok=True)

    print(f"\nEvaluation: {args.method_name}")
    print(f"  Episodes: {len(episodes)}, Pillars: {pillars}, Device: {args.device}")
    print(f"  Strict: {args.strict}, Allow partial: {args.allow_partial}")
    print(f"  Manifest: {manifest_path}\n")

    # Resolve effective config values once for evaluate_episode
    n_frames = effective_config["n_frames_per_shot"]
    vlm_model = effective_config["vlm_model_entity"]

    all_results = []
    pbar = tqdm(list(enumerate(episodes)), desc="Episodes", unit="ep",
                total=len(episodes))
    for idx, (eid, tier) in pbar:
        # Update bar postfix with current episode info (right side of bar)
        try:
            pbar.set_postfix_str(f"{tier:6s} {eid[:24]}")
        except AttributeError:
            pass  # tqdm fallback (a no-op iterator)

        rp = os.path.join(args.out_dir, f"{eid}.json")
        if args.resume and os.path.exists(rp):
            try:
                with open(rp, encoding="utf-8") as f:
                    c = json.load(f)
                r = _episode_result_from_json(eid, tier, c)
                all_results.append(r)
                continue
            except Exception as e:
                logger.warning(f"Failed to resume {eid}, re-evaluating: {e}")

        sp = os.path.join(args.scripts_dir, f"{eid}.json")
        if not os.path.isfile(sp):
            if args.strict:
                raise StrictModeError(
                    f"Script not found for episode {eid}: {sp}. "
                    f"In strict mode every episode in the split must have a script."
                )
            logger.warning(f"Script not found: {sp}")
            continue

        with open(sp, encoding="utf-8") as f:
            script = json.load(f)

        r = evaluate_episode(
            eid, args.results_dir, script, tier, pillars, args.device,
            vlm_model, n_frames, vbench_evaluator=vbench_eval, work_dir=work_dir,
            save_action_grids=args.save_action_grids,
            llm_concurrency=args.llm_concurrency,
        )
        all_results.append(r)

        ep_out = _episode_result_to_json(r)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(ep_out, f, indent=2, ensure_ascii=False)

    # Capture VLM stats AFTER the eval loop, BEFORE saving the report.
    # Stats include rate-limit hits per key, total successful calls, and
    # how often we exhausted all keys (returning None → recorded as
    # n_failed in the affected metrics). Embedding these in the report
    # lets reviewers correlate any unusual n_failed counts with key
    # exhaustion rather than methodological issues.
    vlm_stats = vlm_call_stats()
    manifest["vlm_call_stats"] = vlm_stats
    # Re-write manifest with final stats — atomic to avoid racing other
    # shards that may also be finishing around the same time.
    _write_manifest_atomic(manifest_path, manifest)

    print(f"\nAggregating {len(all_results)} episodes...")
    report = aggregate_results(all_results, split_ids)
    report["_manifest"] = manifest

    # Detect whether this is a shard of a distributed run. The launcher
    # writes shard splits to {out_dir}/.shards/shard_N.json, so we can
    # see it in the split path. When we ARE a shard, we skip writing
    # the report — the launcher's post-run aggregation will produce the
    # canonical report from ALL shards' per-episode JSONs. Multiple
    # shards racing to write the same report file at the same time
    # would otherwise produce incomplete/corrupt output.
    is_shard_run = ".shards" in os.path.normpath(args.split_json).split(os.sep)
    if is_shard_run:
        logger.info(
            f"[Eval] Detected shard run (split={args.split_json}); skipping "
            f"per-shard report write. Run the launcher's aggregation step "
            f"to produce the canonical report from all shards."
        )
    else:
        rj = os.path.join(args.out_dir, f"report_{args.method_name}.json")
        # Atomic write for the same race-safety reasons as the manifest
        rj_tmp = f"{rj}.tmp.{os.getpid()}"
        with open(rj_tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        os.replace(rj_tmp, rj)
        rm = os.path.join(args.out_dir, f"report_{args.method_name}.md")
        write_report(report, args.method_name, rm)

    print(f"\n{'='*60}\nEVALUATION COMPLETE: {args.method_name}\n{'='*60}")

    # VLM stats summary (helps decide whether to add more keys next run)
    if vlm_stats["total_calls"] > 0:
        sr = vlm_stats["successes"] / max(1, vlm_stats["total_calls"])
        print(f"\n  VLM API ({vlm_stats['n_keys']} key(s)):")
        print(f"    total calls:       {vlm_stats['total_calls']}")
        print(f"    successful:        {vlm_stats['successes']}  "
              f"({100*sr:.1f}%)")
        print(f"    rate-limit hits:   {vlm_stats['rate_limit_hits']}  "
              f"(retried via key rotation)")
        if vlm_stats["exhausted_after_max_rounds"] > 0:
            print(f"    EXHAUSTED:         {vlm_stats['exhausted_after_max_rounds']}  "
                  f"(all keys 429'd 3x in a row — recorded as n_failed)")
            print(f"      ↳ consider adding more keys to GEMINI_API_KEYS for next run")
        if vlm_stats["non_rate_limit_errors"] > 0:
            print(f"    other errors:      {vlm_stats['non_rate_limit_errors']}")
    for tier in ["all", "easy", "medium", "hard"]:
        r = report.get(tier, {})
        if not r:
            continue
        print(f"\n  {tier.upper()} ({r['num_episodes']} eps, {r['num_shots']} shots)")
        for label, metrics in [
            ("DINOv2", ["cs_face", "cs_object", "cs_transition_boundary"]),
            ("LLM-judge (similarity)", ["llm_face_mean_score", "llm_object_mean_score",
                                         "llm_scene_mean_score"]),
            ("LLM-judge (accuracy)", ["llm_face_accuracy", "llm_object_accuracy",
                                       "llm_scene_accuracy"]),
        ]:
            print(f"    --- {label} ---")
            for m in metrics:
                s = r.get(m, {})
                # Show coverage explicitly. value=None means "not evaluated".
                if s.get("value", "missing") is None:
                    print(f"    {m:25s}: --- (0/{s.get('n_episodes_total', '?')} episodes evaluated)")
                elif "mean" in s:
                    cov = f"({s.get('n_episodes_evaluated','?')}/{s.get('n_episodes_total','?')})"
                    print(f"    {m:25s}: {s['mean']:.4f} ± {s.get('std',0):.4f}  {cov}")

    print(f"\nOutputs → {args.out_dir}/")


if __name__ == "__main__":
    main()