<p align="center">
  <h2 align="center"><i>EntityBench</i>: Towards Entity-Consistent Long-Range 
  
  Multi-Shot Video Generation</h2>

  <p align="center">
    <span class="author-block">Ruozhen He<sup>1,3</sup>,</span>
    <span class="author-block">Meng Wei<sup>1</sup>,</span>
    <span class="author-block">Ziyan Yang<sup>2</sup>,</span>
    <span class="author-block">Vicente Ordonez<sup>3</sup></span>
  </p>
  <p align="center">
    <sup>1</sup>ByteDance &nbsp;·&nbsp;
    <sup>2</sup>ByteDance Seed &nbsp;·&nbsp;
    <sup>3</sup>Rice University
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2605.15199"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.15199-b31b1b.svg"></a>
    <a href="https://catherine-r-he.github.io/EntityBench/"><img alt="Project page" src="https://img.shields.io/badge/Project-Website-orange"></a>
    <a href="#"><img alt="Code / Data" src="https://img.shields.io/badge/GitHub-Code%20%2F%20Data-black?logo=github"></a>
  </p>
</p>

---

## Abstract

Multi-shot video generation extends single-shot generation to coherent visual
narratives, yet maintaining consistent characters, objects, and locations
across shots remains a challenge over long sequences. Existing evaluations
typically use independently generated prompt sets with limited entity coverage
and simple consistency metrics, making standardized comparison across methods
difficult.
We introduce **EntityBench**, a benchmark of **140 episodes (2,491 shots)**
derived from real narrative media, with explicit per-shot entity schedules
tracking characters, objects, and locations simultaneously across easy /
medium / hard tiers of up to 50 shots, 13 cross-shot characters, 8 cross-shot
locations, 22 cross-shot objects, and recurrence gaps spanning up to 48 shots.
EntityBench pairs the dataset with a three-pillar evaluation suite that
disentangles intra-shot visual quality, prompt-following alignment, and
cross-shot entity consistency, with a **fidelity gate** that admits only
accurate entity appearances into cross-shot scoring.
To establish baselines, we propose **EntityMem**, a memory-augmented
generation system that stores verified per-entity visual references in a
persistent memory bank *before* generation begins, enabling the video backbone
to retrieve each entity's appearance across shots. Experiments show that
cross-shot entity consistency degrades sharply with recurrence distance in
existing methods, and that explicit per-entity memory yields the highest
character fidelity (Cohen's *d* = +2.33) and presence among methods evaluated.

---

## What's in this release

- `data/scripts/` — 140 episode JSONs (~2,491 shots), each with the full
  per-shot entity schedule and registry prompts.
- `data/splits/` — easy / medium / hard tier splits.
- `eval/` — three Python files: `evaluate_benchmark.py` (single-GPU
  evaluator), `run_eval_distributed.py` (multi-GPU launcher),
  `compare_methods.py` (method-vs-method comparison).
- `examples/run_eval_example.sh` — ready-to-edit launch script.

## Setup

```bash
conda create -n entitybench python=3.11 -y
conda activate entitybench

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.36.0 transformers tqdm openai imageio imageio-ffmpeg
pip install groundingdino-py vbench pyiqa
```

Pillar 2 / Pillar 3 LLM scoring uses an Azure-OpenAI-compatible endpoint:

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export GEMINI_API_KEYS="key1,key2,key3"     # comma-separated, rotated round-robin
```

## Run

Generated videos must be laid out as
`<results_dir>/<episode_id>/<scene>_<shot>.mp4` (zero-padded, e.g.
`001_001.mp4`).

### Multi-GPU (recommended)

```bash
python eval/run_eval_distributed.py \
  --n_gpus 8 \
  --results_dir <generated videos> \
  --scripts_dir data/scripts \
  --split_json  data/splits/final_split_validated_ids.json \
  --out_dir     eval_results/<method_name> \
  --method_name <method_name> \
  --pillars 1,2,3 \
  --llm_concurrency 5 \
  --resume
```

### Single GPU

```bash
python eval/evaluate_benchmark.py \
  --results_dir <generated videos> \
  --scripts_dir data/scripts \
  --split_json  data/splits/final_split_validated_ids.json \
  --out_dir     eval_results/<method_name> \
  --method_name <method_name> \
  --pillars 1,2,3 \
  --llm_concurrency 5 \
  --resume
```

### Comparisons

```bash
python eval/compare_methods.py \
  --method_a eval_results/method_a      --label_a method_a \
  --method_b eval_results/method_b  --label_b method_b \
  --out_dir  comparison/
```

## Citation

```bibtex
@article{he2026entitybench,
  title   = {EntityBench: Towards Entity-Consistent Long-Range Multi-Shot Video Generation},
  author  = {He, Ruozhen and Meng, Wei and Yang, Ziyan and Ordonez, Vicente},
  journal = {Preprint},
  year    = {2026},
}
```
