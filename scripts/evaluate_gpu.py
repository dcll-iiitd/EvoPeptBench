"""
PeptideBench-MBPP — GPU Inference Script (single V100)
=======================================================
Scorer  : peptide.py (PeptideBLEU) — no dependency on peptide_bench.py.
Model   : microsoft/BioGPT-Large loaded in float16 on cuda:0.

Key differences vs the CPU script
----------------------------------
  1. BioGPT-Large loaded in float16 on cuda:0 (~750 MB VRAM).
  2. Batched generation — gpu_batch_size tasks are processed in one
     model.generate() call, saturating V100 tensor cores.
  3. Scoring is parallelized across CPU cores (multiprocessing.Pool)
     while the GPU generates the next batch.
  4. All scoring logic comes from peptide.py (PeptideBLEU).

V100 memory budget:
  Weights  : ~750 MB (fp16)
  Prompts  : ~100-150 tokens × batch_size × k
  Total    : ~3-4 GB at batch_size=32, k=5  (well within 16 GB)
  → If OOM: halve --gpu-batch-size

Usage:
  pip install transformers torch numpy
  python peptide_bench_mbpp_gpu.py --tasks tasks.jsonl --mode WITH_LENGTH --k 5
  python peptide_bench_mbpp_gpu.py --tasks tasks.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ---------------------------------------------------------------------------
# Import from the CPU script (all shared logic lives there)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from evaluate_cpu import (
        DEFAULT_MODEL_ID,
        DEFAULT_K,
        PASS_THRESHOLD,
        PeptideTask,
        TaskResult,
        AggregateResult,
        load_tasks,
        _parse_sequence,
        _score_worker_init,
        _score_one_task,
        _parallel_score,
        print_results,
        save_results,
        diversity_score,
        novelty_score,
        PeptideBenchMBPP,   # reuse _aggregate and _dummy
    )
    from peptide_bleu.core import (
        peptide_metric,
        score_components,
        is_valid_sequence,
        sanitize,
        _COMPONENT_KEYS,
    )
except ImportError as exc:
    sys.exit(
        f"[ERROR] Cannot import required modules: {exc}\n"
    )


# =============================================================================
# SECTION 1 — GPU MODEL LOADER
# =============================================================================

def load_model_gpu(model_path: str):
    """Load BioGPT-Large in float16 on cuda:0. Returns (model, tokenizer, device)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Run:  pip install transformers torch  (CUDA build)")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device found.\n"
            "Use peptide_bench_mbpp.py for CPU inference."
        )

    device   = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"  GPU : {gpu_name}  ({vram_gb:.1f} GB VRAM)", flush=True)
    print(f"  Loading '{model_path}' in float32 …", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left-pad for causal batched generation

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    used = torch.cuda.memory_allocated(0) / 1e9
    print(f"  Model loaded.  VRAM used: {used:.2f} GB", flush=True)
    return model, tok, device


# =============================================================================
# SECTION 2 — BATCHED GPU GENERATION
# =============================================================================

def _generate_batch_gpu(
    model,
    tok,
    device,
    tasks:       List[PeptideTask],
    k:           int,
    temperature: float,
    mode:        str,
) -> Dict[int, List[str]]:
    """
    Generate k sequences for every task in one GPU call.
    Returns {task_id: [seq, …]}.

    Each task's NL prompt is replicated k times and left-padded to the
    longest prompt in the batch. Only newly generated tokens (after the
    prompt) are decoded so the returned strings are pure AA sequences.
    """
    import torch

    prompts: List[str] = []
    for t in tasks:
        p = (t.nlp_prompt_with_length() if mode == "WITH_LENGTH"
             else t.nlp_prompt_without_length())
        prompts.extend([p] * k)

    max_target   = max(t.length for t in tasks)
    max_new_toks = (max(max_target + 10, 20) if mode == "WITH_LENGTH"
                    else max(max_target + 30, 40))

    enc = tok(prompts, return_tensors="pt", padding=True,
              truncation=True, max_length=512).to(device)
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids          = enc["input_ids"],
            attention_mask     = enc["attention_mask"],
            max_new_tokens     = max_new_toks,
            do_sample          = True,
            temperature        = temperature,
            repetition_penalty = 1.2,
            pad_token_id       = tok.eos_token_id,
        )

    results: Dict[int, List[str]] = {t.task_id: [] for t in tasks}
    flat_idx = 0
    for task in tasks:
        for _ in range(k):
            raw = tok.decode(out[flat_idx][prompt_len:], skip_special_tokens=True)
            seq = _parse_sequence(raw)
            if mode == "WITH_LENGTH":
                seq = seq[: task.length]
            results[task.task_id].append(seq)
            flat_idx += 1

    return results


# =============================================================================
# SECTION 3 — GPU EVALUATOR
# =============================================================================

class PeptideBenchMBPP_GPU:
    """
    GPU-accelerated MBPP harness using PeptideBLEU (peptide.py) for scoring.

    Parameters
    ----------
    tasks          : list of PeptideTask
    mode           : "WITH_LENGTH" | "WITHOUT_LENGTH"
    k              : samples per task
    temperature    : sampling temperature
    pass_threshold : pass@k score threshold
    gpu_batch_size : tasks per GPU call (halve if OOM; default 32)
    n_workers      : CPU scoring workers (default: all cores)
    """

    def __init__(
        self,
        tasks:          List[PeptideTask],
        mode:           str           = "WITH_LENGTH",
        k:              int           = DEFAULT_K,
        temperature:    float         = 0.8,
        pass_threshold: float         = PASS_THRESHOLD,
        gpu_batch_size: int           = 32,
        n_workers:      Optional[int] = None,
    ):
        assert mode in ("WITH_LENGTH", "WITHOUT_LENGTH")
        self.tasks, self.mode           = tasks, mode
        self.k, self.temperature        = k, temperature
        self.pass_threshold             = pass_threshold
        self.gpu_batch_size             = gpu_batch_size
        self.n_workers                  = n_workers or max(1, os.cpu_count() or 1)
        # Reuse CPU harness for _dummy and _aggregate
        self._cpu = PeptideBenchMBPP(tasks=tasks, mode=mode, k=k,
                                      temperature=temperature,
                                      pass_threshold=pass_threshold,
                                      n_workers=self.n_workers)
        self.out_path = None

    def evaluate_dry_run(self) -> Tuple[List[TaskResult], AggregateResult]:
        print(f"  [DRY-RUN] Scoring {len(self.tasks)} tasks "
              f"across {self.n_workers} workers …", flush=True)
        jobs = [(t.task_id, t.functional_classes, t.activity, t.sequence, t.length,
                 self._cpu._dummy(t), self._cpu._dummy(t)) for t in self.tasks]
        results = _parallel_score(jobs, self.pass_threshold, self.n_workers)
        return results, self._cpu._aggregate(results)

    def evaluate(
        self, model_path: str = DEFAULT_MODEL_ID
    ) -> Tuple[List[TaskResult], AggregateResult]:
        """
        Batched GPU generation + parallel CPU scoring.
        GPU generates gpu_batch_size tasks at once; CPU pool scores them
        concurrently while the GPU processes the next batch.
        """
        model, tok, device = load_model_gpu(model_path)
        n_tasks = len(self.tasks)
        all_r:  List[TaskResult] = []

        for batch_start in range(0, n_tasks, self.gpu_batch_size):
            batch = self.tasks[batch_start: batch_start + self.gpu_batch_size]

            seqs_by_id = _generate_batch_gpu(
                model, tok, device, batch, self.k, self.temperature, self.mode
            )
            jobs = [(t.task_id, t.functional_classes, t.activity, t.sequence,
                     t.length, seqs_by_id[t.task_id], seqs_by_id[t.task_id])
                    for t in batch]

            batch_r = _parallel_score(jobs, self.pass_threshold, self.n_workers)
            all_r.extend(batch_r)

            done = min(batch_start + self.gpu_batch_size, n_tasks)
            last = batch_r[-1]
            print(f"  [{done:6d}/{n_tasks}]  task_id={last.task_id}  "
                  f"best={last.best_score:.3f}  "
                  f"passed={'✓' if last.passed else '✗'}", flush=True)


            done = min(batch_start + self.gpu_batch_size, n_tasks)
            last = batch_r[-1]

            print(
                f"  [{done:6d}/{n_tasks}]  "
                f"last task_id={last.task_id}  "
                f"best={last.best_score:.3f}  "
                f"passed={'✓' if last.passed else '✗'}",
                flush=True,
            )

            # 🔥 NEW: incremental saving
            if hasattr(self, "out_path") and self.out_path:
                if batch_start % self.gpu_batch_size == 0:
                    try:
                        agg_partial = self._cpu._aggregate(all_r)
                        from evaluate_cpu import save_results
                        save_results(
                            all_r,
                            agg_partial,
                            self.out_path,
                            model="partial [GPU]",
                            mode=self.mode,
                        )
                        print(f"  💾 checkpoint saved at {done} tasks", flush=True)
                    except Exception as e:
                        print(f"  ⚠️ checkpoint failed: {e}", flush=True)

        return all_r, self._cpu._aggregate(all_r)


# =============================================================================
# SECTION 4 — CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PeptideBench-MBPP GPU — BioGPT-Large + PeptideBLEU scorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tasks",          required=True)
    p.add_argument("--model-path",     default=DEFAULT_MODEL_ID)
    p.add_argument("--mode",           default="WITH_LENGTH",
                   choices=["WITH_LENGTH","WITHOUT_LENGTH"])
    p.add_argument("--k",              type=int,   default=DEFAULT_K)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--threshold",      type=float, default=PASS_THRESHOLD)
    p.add_argument("--gpu-batch-size", type=int,   default=32,
                   help="Tasks per GPU call — halve if OOM")
    p.add_argument("--workers",        type=int,   default=None)
    p.add_argument("--out",            default=None)
    p.add_argument("--dry-run",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"\nLoading tasks from: {args.tasks}")
    tasks = load_tasks(args.tasks)
    print(f"  {len(tasks)} tasks loaded.")

    n_workers     = args.workers or max(1, os.cpu_count() or 1)
    seqs_per_call = args.gpu_batch_size * args.k
    print(f"  GPU batch size           : {args.gpu_batch_size}  tasks per GPU call")
    print(f"  Sequences per GPU call   : {seqs_per_call}  ({args.gpu_batch_size} × k={args.k})")
    print(f"  Parallel scoring workers : {n_workers}")

    harness = PeptideBenchMBPP_GPU(
        tasks          = tasks,
        mode           = args.mode,
        k              = args.k,
        temperature    = args.temperature,
        pass_threshold = args.threshold,
        gpu_batch_size = args.gpu_batch_size,
        n_workers      = n_workers,
    )
    harness.out_path = args.out

    if args.dry_run:
        print("\n[DRY-RUN] Skipping GPU, using dummy sequences.\n")
        results, agg = harness.evaluate_dry_run()
        model_name = "dry-run"
    else:
        print(f"\nRunning GPU evaluation  (model={args.model_path}, mode={args.mode})\n")
        results, agg = harness.evaluate(model_path=args.model_path)
        model_name = args.model_path

    print()
    print_results(results, agg, model=f"{model_name} [GPU fp16]")

    if args.out:
        save_results(results, agg, args.out,
                     model=f"{model_name} [GPU fp16]", mode=args.mode)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # safe CUDA fork avoidance
    main()