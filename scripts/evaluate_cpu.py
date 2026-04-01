"""
PeptideBench-MBPP: MBPP-style Evaluation Harness for LLM Peptide Generation
=============================================================================
Scorer: peptide.py (PeptideBLEU) — no dependency on peptide_bench.py.

Each generated sequence is scored with peptide_metric(ref, pred, activity=X)
where ref is the task's ground-truth sequence and the activity preset is
mapped from the task's functional class labels.

Task JSON format:
  {"task_id":1, "sequence":"YLGYLE", "length":6, "prompt":"...", ...}

Prompting modes:
  WITH_LENGTH    — prompt states the exact target length
  WITHOUT_LENGTH — prompt gives only functional property description

Scoring:
  pass@1  : mean PeptideBLEU of the best sample per task
  pass@k  : fraction of tasks where >=1 sample scores >= threshold
  mean    : mean across all samples x tasks

Activity preset mapping (peptide.py):
  anti-bacterial / anti-fungal / anti-parasitic / toxic  → amp
  drug-delivery                                          → cpp
  signal-peptide                                         → signal
  anti-viral / anti-cancer / immunological               → immunological
  inhibitor / metabolic / other-functional / …           → default weights

Usage:
  pip install transformers torch numpy
  python peptide_bench_mbpp.py --tasks tasks.jsonl --mode WITH_LENGTH --k 5
  python peptide_bench_mbpp.py --tasks tasks.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Scorer import
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import peptide_bleu.core as _pep
    from peptide_bleu.core import (
        peptide_metric,
        score_components,
        is_valid_sequence,
        sanitize,
        _COMPONENT_KEYS,
    )
except ImportError as exc:
    sys.exit(
        f"[ERROR] Cannot import peptide_bleu.core: {exc}\n"
    )


# =============================================================================
# SECTION 1 — CONSTANTS & ACTIVITY MAPPING
# =============================================================================

PASS_THRESHOLD: float = 0.60
DEFAULT_K:      int   = 5

ALL_PROPERTIES: List[str] = [
    "anti-bacterial", "anti-cancer", "anti-fungal", "anti-parasitic",
    "anti-viral", "cell-cell-communication", "drug-delivery",
    "immunological", "inhibitor", "metabolic", "other-functional",
    "signal-peptide", "toxic",
]

PROPERTY_LABELS: Dict[str, str] = {
    "anti-bacterial":          "anti-bacterial",
    "anti-cancer":             "anti-cancer",
    "anti-fungal":             "anti-fungal",
    "anti-parasitic":          "anti-parasitic",
    "anti-viral":              "anti-viral",
    "cell-cell-communication": "involved in cell-cell communication",
    "drug-delivery":           "suitable for drug delivery",
    "immunological":           "immunological",
    "inhibitor":               "an inhibitor",
    "metabolic":               "metabolic",
    "other-functional":        "bioactive / other-functional",
    "signal-peptide":          "a signal peptide",
    "toxic":                   "toxic",
}

# Maps functional class → peptide.py activity preset
# Classes not listed → None → DEFAULT_WEIGHTS
CLASS_TO_ACTIVITY: Dict[str, str] = {
    "anti-bacterial": "amp",
    "anti-fungal":    "amp",
    "anti-parasitic": "amp",
    "toxic":          "amp",
    "drug-delivery":  "cpp",
    "signal-peptide": "signal",
    "anti-viral":     "immunological",
    "anti-cancer":    "immunological",
    "immunological":  "immunological",
}

DEFAULT_MODEL_ID = "microsoft/BioGPT-Large"


def resolve_activity(functional_classes: List[str]) -> Optional[str]:
    """Return the first recognised activity preset, or None for default weights."""
    for cls in functional_classes:
        if cls in CLASS_TO_ACTIVITY:
            return CLASS_TO_ACTIVITY[cls]
    return None


# =============================================================================
# SECTION 2 — TASK DATACLASS & LOADER
# =============================================================================

@dataclass
class PeptideTask:
    task_id:               int
    sequence:              str
    length:                int
    prompt:                str
    properties:            Dict[str, bool]
    ref_net_charge:        Optional[float] = None
    ref_avg_hydrophobicity:Optional[float] = None
    ref_peptide_mw_da:     Optional[float] = None
    ref_hydrophobic_pct:   Optional[float] = None
    ref_aromaticity_pct:   Optional[float] = None
    functional_classes:    List[str] = None   # type: ignore[assignment]
    activity:              Optional[str] = None

    def __post_init__(self):
        self.sequence = self.sequence.strip().upper()
        if self.functional_classes is None:
            self.functional_classes = [p for p, v in self.properties.items() if v]
        if self.activity is None:
            self.activity = resolve_activity(self.functional_classes)

    def _property_sentence(self) -> str:
        parts = [
            f"{PROPERTY_LABELS[p]}: {'yes' if self.properties.get(p) else 'no'}"
            for p in ALL_PROPERTIES
        ]
        return "; ".join(parts)

    def nlp_prompt_with_length(self) -> str:
        return (
            f"Generate a peptide amino acid sequence with the following properties: "
            f"{self._property_sentence()}. "
            f"The sequence must be exactly {self.length} amino acids long. "
            f"Output only the amino acid sequence using uppercase single-letter codes "
            f"with no spaces, punctuation, or explanation."
        )

    def nlp_prompt_without_length(self) -> str:
        return (
            f"Generate a peptide amino acid sequence with the following properties: "
            f"{self._property_sentence()}. "
            f"Output only the amino acid sequence using uppercase single-letter codes "
            f"with no spaces, punctuation, or explanation."
        )

    def prompt_with_length(self)    -> str: return self.nlp_prompt_with_length()
    def prompt_without_length(self) -> str: return self.nlp_prompt_without_length()


def _parse_properties_from_prompt(prompt: str) -> Dict[str, bool]:
    text = prompt.lower()
    neg  = re.compile(r"\bnot\b|\bnon[-\s]")
    result: Dict[str, bool] = {}
    for prop in ALL_PROPERTIES:
        patterns = list({prop.lower(), PROPERTY_LABELS[prop].lower()})
        pos, neg_found = False, False
        for pat in patterns:
            for m in re.finditer(re.escape(pat), text):
                prefix = text[max(0, m.start()-20): m.start()]
                if neg.search(prefix):
                    neg_found = True
                else:
                    pos = True
        result[prop] = pos and not neg_found
    return result


def load_tasks(path: str) -> List[PeptideTask]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    records = json.loads(raw) if raw.startswith("[") else [
        json.loads(line) for line in raw.splitlines() if line.strip()
    ]
    tasks = []
    for rec in records:
        prompt = str(rec["prompt"])
        tasks.append(PeptideTask(
            task_id               = int(rec["task_id"]),
            sequence              = str(rec["sequence"]),
            length                = int(rec.get("length", len(rec["sequence"]))),
            prompt                = prompt,
            properties            = _parse_properties_from_prompt(prompt),
            ref_net_charge        = rec.get("ref_net_charge"),
            ref_avg_hydrophobicity= rec.get("ref_avg_hydrophobicity"),
            ref_peptide_mw_da     = rec.get("ref_peptide_mw_da"),
            ref_hydrophobic_pct   = rec.get("ref_hydrophobic_pct"),
            ref_aromaticity_pct   = rec.get("ref_aromaticity_pct"),
        ))
    return tasks


# =============================================================================
# SECTION 3 — MODEL (BioGPT-Large, CPU)
# =============================================================================

def _load_model(model_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Run:  pip install transformers torch")

    print(f"  Loading '{model_name}' …", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    import torch
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"  Model loaded (threads={torch.get_num_threads()}).", flush=True)
    return model, tok


def _parse_sequence(raw: str) -> str:
    lines = [l for l in raw.splitlines() if not l.startswith(">")]
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", " ".join(lines).upper())


def _generate_sequences(
    model_and_tok,
    task:        PeptideTask,
    k:           int,
    temperature: float,
    mode:        str,
) -> List[str]:
    import torch
    model, tok = model_and_tok
    prompt = task.nlp_prompt_with_length() if mode == "WITH_LENGTH" \
             else task.nlp_prompt_without_length()
    max_new = max(task.length + 10, 20) if mode == "WITH_LENGTH" \
              else max(task.length + 30, 40)

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(
            input_ids           = enc["input_ids"],
            attention_mask      = enc["attention_mask"],
            max_new_tokens      = max_new,
            do_sample           = True,
            temperature         = temperature,
            num_return_sequences= k,
            repetition_penalty  = 1.2,
            pad_token_id        = tok.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    seqs = []
    for ids in out:
        raw = tok.decode(ids[prompt_len:], skip_special_tokens=True)
        seq = _parse_sequence(raw)
        if mode == "WITH_LENGTH":
            seq = seq[:task.length]
        seqs.append(seq)
    return seqs


# =============================================================================
# SECTION 4 — CORPUS METRICS (self-contained, no external dep)
# =============================================================================

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    n, m = len(a), len(b)
    if not n: return m
    if not m: return n
    prev = list(range(m+1))
    for i in range(1, n+1):
        curr = [i] + [0]*m
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
        prev = curr
    return prev[m]


def _ned(a: str, b: str) -> float:
    ml = max(len(a), len(b))
    return 0.0 if not ml else _levenshtein(a, b) / ml


def diversity_score(seqs: List[str]) -> float:
    seqs = [s for s in seqs if s]
    if len(seqs) < 2: return 0.0
    if len(seqs) > 200:
        rng  = np.random.default_rng(seed=42)
        seqs = [seqs[i] for i in rng.choice(len(seqs), 200, replace=False)]
    total, count = 0.0, 0
    for i in range(len(seqs)):
        for j in range(i+1, len(seqs)):
            total += _ned(seqs[i], seqs[j]); count += 1
    return round(total/count, 4) if count else 0.0


def novelty_score(hyps: List[str], refs: List[str], threshold: float = 0.30) -> float:
    if not hyps or not refs: return 1.0
    return round(
        sum(1 for h in hyps if all(_ned(h, r) > threshold for r in refs))
        / len(hyps), 4
    )


# =============================================================================
# SECTION 5 — PARALLEL SCORING
# =============================================================================

def _score_worker_init(pass_threshold: float):
    global _W_THRESH
    _W_THRESH = pass_threshold


def _score_one_task(args: tuple) -> "TaskResult":
    task_id, func_classes, activity, ref_seq, ref_len, seqs, raw = args
    scores, comp_sums, n_valid = [], {k: 0.0 for k in _COMPONENT_KEYS}, 0

    for seq in seqs:
        clean = sanitize(seq)
        if seq and is_valid_sequence(clean):
            s = peptide_metric(ref_seq, clean, activity=activity)
            scores.append(s)
            for k, v in score_components(ref_seq, clean).items():
                comp_sums[k] += v
            n_valid += 1
        else:
            scores.append(0.0)

    best_i  = int(np.argmax(scores)) if scores else 0
    n_v     = max(n_valid, 1)
    return TaskResult(
        task_id            = task_id,
        functional_classes = func_classes,
        activity           = activity,
        reference_seq      = ref_seq,
        ref_length         = ref_len,
        generated          = seqs,
        raw_responses      = raw,
        scores             = [round(s, 4) for s in scores],
        best_score         = round(scores[best_i], 4) if scores else 0.0,
        mean_score         = round(float(np.mean(scores)), 4) if scores else 0.0,
        best_seq           = seqs[best_i] if seqs else "",
        passed             = any(s >= _W_THRESH for s in scores),
        component_means    = {k: round(v/n_v, 4) for k, v in comp_sums.items()},
    )


def _parallel_score(jobs: List[tuple], threshold: float, n_workers: int) -> List["TaskResult"]:
    with mp.Pool(n_workers, initializer=_score_worker_init, initargs=(threshold,)) as pool:
        return pool.map(_score_one_task, jobs)


# =============================================================================
# SECTION 6 — RESULT DATACLASSES
# =============================================================================

@dataclass
class TaskResult:
    task_id:            int
    functional_classes: List[str]
    activity:           Optional[str]
    reference_seq:      str
    ref_length:         int
    generated:          List[str]
    raw_responses:      List[str]
    scores:             List[float]
    best_score:         float
    mean_score:         float
    best_seq:           str
    passed:             bool
    component_means:    Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateResult:
    n_tasks:         int
    k:               int
    mode:            str
    pass_threshold:  float
    pass_at_1:       float
    pass_at_k:       float
    mean_score:      float
    pass_rate:       float
    diversity:       float
    novelty:         float
    mean_components: Dict[str, float]


# =============================================================================
# SECTION 7 — EVALUATOR
# =============================================================================

class PeptideBenchMBPP:
    def __init__(
        self,
        tasks:          List[PeptideTask],
        mode:           str           = "WITH_LENGTH",
        k:              int           = DEFAULT_K,
        temperature:    float         = 0.8,
        pass_threshold: float         = PASS_THRESHOLD,
        n_workers:      Optional[int] = None,
        gen_batch_size: int           = 512,
    ):
        assert mode in ("WITH_LENGTH", "WITHOUT_LENGTH")
        self.tasks, self.mode           = tasks, mode
        self.k, self.temperature        = k, temperature
        self.pass_threshold             = pass_threshold
        self.n_workers                  = n_workers or max(1, os.cpu_count() or 1)
        self.gen_batch_size             = gen_batch_size

    def evaluate_dry_run(self) -> Tuple[List[TaskResult], AggregateResult]:
        print(f"  Scoring {len(self.tasks)} tasks across {self.n_workers} workers …",
              flush=True)
        jobs = [(t.task_id, t.functional_classes, t.activity, t.sequence, t.length,
                 self._dummy(t), self._dummy(t)) for t in self.tasks]
        results = _parallel_score(jobs, self.pass_threshold, self.n_workers)
        return results, self._aggregate(results)

    def evaluate(self, model_name: str = DEFAULT_MODEL_ID
                 ) -> Tuple[List[TaskResult], AggregateResult]:
        mt    = _load_model(model_name)
        n     = len(self.tasks)
        all_r: List[TaskResult] = []
        pend:  List[tuple]      = []

        def _flush():
            if not pend: return
            all_r.extend(_parallel_score(pend, self.pass_threshold, self.n_workers))
            pend.clear()

        for task in self.tasks:
            seqs = _generate_sequences(mt, task, self.k, self.temperature, self.mode)
            pend.append((task.task_id, task.functional_classes, task.activity,
                         task.sequence, task.length, seqs, seqs))
            if len(pend) >= self.gen_batch_size:
                _flush()
                last = all_r[-1]
                print(f"  [{len(all_r):6d}/{n}] task_id={last.task_id}  "
                      f"best={last.best_score:.3f}  "
                      f"passed={'✓' if last.passed else '✗'}", flush=True)

        _flush()
        print(f"  [{n}/{n}] complete.", flush=True)
        return all_r, self._aggregate(all_r)

    def _dummy(self, t: PeptideTask) -> List[str]:
        templates = [t.sequence,
                     ("KLAKLAK" * max(1, t.length//7))[:t.length],
                     ("AAAAAAA" * max(1, t.length//7))[:t.length],
                     "KWKLFKKIEKVGQ"[:t.length],
                     t.sequence[::-1]]
        seqs = (templates * math.ceil(self.k / len(templates)))[: self.k]
        return [s[:max(t.length, 4)] for s in seqs]

    def _aggregate(self, results: List[TaskResult]) -> AggregateResult:
        if not results: raise ValueError("No results.")
        all_scores = [s for r in results for s in r.scores]
        all_seqs   = [s for r in results for s in r.generated]
        ref_seqs   = [r.reference_seq for r in results]

        comp_sums = {k: 0.0 for k in _COMPONENT_KEYS}
        n_comp    = 0
        for r in results:
            clean = sanitize(r.best_seq) if r.best_seq else ""
            if clean and is_valid_sequence(clean):
                for k, v in score_components(r.reference_seq, clean).items():
                    comp_sums[k] += v
                n_comp += 1
        n_c = max(n_comp, 1)

        return AggregateResult(
            n_tasks         = len(results),
            k               = self.k,
            mode            = self.mode,
            pass_threshold  = self.pass_threshold,
            pass_at_1       = round(float(np.mean([r.best_score for r in results])), 4),
            pass_at_k       = round(float(np.mean([r.passed for r in results])), 4),
            mean_score      = round(float(np.mean(all_scores)), 4),
            pass_rate       = round(sum(1 for s in all_seqs
                                        if s and is_valid_sequence(sanitize(s)))
                                    / max(len(all_seqs), 1), 4),
            diversity       = diversity_score([s for s in all_seqs if s]),
            novelty         = novelty_score([s for s in all_seqs if s], ref_seqs),
            mean_components = {k: round(v/n_c, 4) for k, v in comp_sums.items()},
        )


# =============================================================================
# SECTION 8 — OUTPUT & SERIALISATION
# =============================================================================

W = 80

def print_results(
    results: List[TaskResult],
    agg:     AggregateResult,
    model:   str = "—",
) -> None:
    bar = "═" * W; dash = "─" * W
    print(bar)
    print("  PeptideBench-MBPP  (scorer: PeptideBLEU / peptide.py)")
    print(f"  Model : {model}")
    print(f"  Mode  : {agg.mode}  │  Tasks: {agg.n_tasks}  │  "
          f"k={agg.k}  │  threshold≥{agg.pass_threshold:.2f}")
    print(dash)
    print(f"  {'task_id':>8}  {'activity':<14}  {'best':>6}  "
          f"{'mean':>6}  {'pass':>5}  best_generated")
    print("  " + "─"*(W-2))
    for r in results:
        act = r.activity or "default"
        seq = r.best_seq[:26] + ("…" if len(r.best_seq) > 26 else "")
        print(f"  {r.task_id:>8}  {act:<14}  {r.best_score:>6.4f}  "
              f"{r.mean_score:>6.4f}  {'✓' if r.passed else '✗':>5}  {seq}")
    print(dash)
    print("  AGGREGATE RESULTS")
    print(dash)
    print(f"  {'pass@1  (mean best-sample score)':<42}: {agg.pass_at_1:.4f}")
    print(f"  {'pass@k  (tasks with ≥1 pass)':<42}: {agg.pass_at_k:.4f}"
          f"  ({agg.pass_at_k*agg.n_tasks:.0f}/{agg.n_tasks} tasks)")
    print(f"  {'mean score (all samples)':<42}: {agg.mean_score:.4f}")
    print(dash)
    print("  PeptideBLEU component breakdown (best sample per task):")
    for comp, val in agg.mean_components.items():
        print(f"    {comp:<30}: {val:.4f}")
    print(dash)
    print("  Corpus health:")
    print(f"    {'pass_rate':<42}: {agg.pass_rate:.4f}")
    print(f"    {'diversity':<42}: {agg.diversity:.4f}")
    print(f"    {'novelty':<42}: {agg.novelty:.4f}")
    print(bar)


def save_results(
    results:  List[TaskResult],
    agg:      AggregateResult,
    out_path: str,
    model:    str = "—",
    mode:     str = "—",
) -> None:
    payload = {
        "metadata": {"model": model, "scorer": "PeptideBLEU/peptide.py",
                     "mode": mode, "k": agg.k,
                     "pass_threshold": agg.pass_threshold, "n_tasks": agg.n_tasks},
        "aggregate": asdict(agg),
        "per_task":  [r.to_dict() for r in results],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}")


# =============================================================================
# SECTION 9 — CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PeptideBench-MBPP with PeptideBLEU scorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tasks",          required=True)
    p.add_argument("--model_name",     default=DEFAULT_MODEL_ID,
                   help="HuggingFace model string (e.g., mistralai/Mistral-7B, microsoft/BioGPT-Large)")
    p.add_argument("--mode",           default="WITH_LENGTH",
                   choices=["WITH_LENGTH","WITHOUT_LENGTH"])
    p.add_argument("--k",              type=int,   default=DEFAULT_K)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--threshold",      type=float, default=PASS_THRESHOLD)
    p.add_argument("--workers",        type=int,   default=None)
    p.add_argument("--gen-batch-size", type=int,   default=512)
    p.add_argument("--out",            default=None)
    p.add_argument("--dry-run",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"\nLoading tasks from: {args.tasks}")
    tasks = load_tasks(args.tasks)
    print(f"  {len(tasks)} tasks loaded.")

    n_workers = args.workers or max(1, os.cpu_count() or 1)
    print(f"  Parallel scoring workers : {n_workers}")
    print(f"  Generation batch size    : {args.gen_batch_size}")

    harness = PeptideBenchMBPP(
        tasks=tasks, mode=args.mode, k=args.k,
        temperature=args.temperature, pass_threshold=args.threshold,
        n_workers=n_workers, gen_batch_size=args.gen_batch_size,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Using dummy sequences.\n")
        results, agg = harness.evaluate_dry_run()
        model_name = "dry-run"
    else:
        print(f"\nRunning (model={args.model_name}, mode={args.mode}, k={args.k})\n")
        results, agg = harness.evaluate(model_name=args.model_name)
        model_name = args.model_name

    print()
    print_results(results, agg, model=model_name)
    if args.out:
        save_results(results, agg, args.out, model=model_name, mode=args.mode)


# =============================================================================
# SECTION 10 — SELF-TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        import tempfile, os as _os

        _demo_json = json.dumps([
            {"task_id": 1, "sequence": "YLGYLE", "length": 6,
             "prompt": ("This peptide has the following properties: not anti-bacterial, "
                        "not anti-cancer, not anti-fungal, not anti-parasitic, not anti-viral, "
                        "not involved in cell-cell communication, not suitable for drug delivery, "
                        "not immunological, an inhibitor, metabolic, bioactive (multifunctional), "
                        "not a signal peptide, non-toxic. The peptide length is 6 amino acids.")},
            {"task_id": 2, "sequence": "KLAKLAKKLAKLAK", "length": 14,
             "prompt": ("This peptide has the following properties: anti-bacterial, "
                        "anti-fungal, not anti-cancer, not anti-viral, not immunological, "
                        "not toxic. The peptide length is 14 amino acids.")},
            {"task_id": 3, "sequence": "SIINFEKL", "length": 8,
             "prompt": ("This peptide has the following properties: immunological, "
                        "not anti-bacterial, not anti-fungal, not anti-cancer, not anti-viral, "
                        "not toxic, not a signal peptide. The peptide length is 8 amino acids.")},
        ])

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        tmp.write(_demo_json); tmp.close()

        print("=" * W)
        print("  PeptideBench-MBPP — Demo  (scorer: PeptideBLEU / peptide.py)")
        print("=" * W)

        try:
            tasks = load_tasks(tmp.name)
            print(f"\n  {len(tasks)} tasks  |  activities: {[t.activity for t in tasks]}")

            # Show a couple of prompts
            print(f"\n  [WITH_LENGTH prompt for task 1]\n  {tasks[0].nlp_prompt_with_length()}\n")
            print(f"  [WITHOUT_LENGTH prompt for task 1]\n  {tasks[0].nlp_prompt_without_length()}\n")

            # Manual scoring demo
            t2 = tasks[1]  # anti-bacterial, activity=amp
            print("─" * W)
            print(f"  Manual scoring (task 2, ref={t2.sequence}, activity={t2.activity}):")
            print(f"  {'Sequence':<22} {'Description':<26} {'PeptideBLEU':>12}")
            print("  " + "─" * 62)
            for seq, desc in [
                (t2.sequence,           "exact reference"),
                ("KRRKKLLLKKR",         "good cationic AMP"),
                ("AAAAAAAAAAAAA",       "poly-A"),
                ("DDDDDDDDDDDDD",       "poly-D (wrong charge)"),
                ("XYZXYZ",             "invalid residues"),
            ]:
                s = peptide_metric(t2.sequence, seq, activity=t2.activity)
                print(f"  {seq:<22} {desc:<26} {s:>12.4f}")

            # Dry-run MBPP
            print("\n" + "─" * W)
            harness = PeptideBenchMBPP(tasks=tasks, mode="WITH_LENGTH", k=5, n_workers=2)
            results, agg = harness.evaluate_dry_run()
            print_results(results, agg, model="dummy-sequences")

        finally:
            _os.unlink(tmp.name)
