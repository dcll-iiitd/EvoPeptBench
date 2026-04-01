"""
peptide_bleu.py
===============
PeptideBLEU — A CodeBLEU-style evaluation metric for peptide sequences.

Designed to score AI-generated peptide sequences against ground-truth
references using biochemical rules sourced from "The amino acid property
table" rule document.

QUICK START
-----------
    from peptide_bleu import peptide_metric, batch_peptide_metric

    score = peptide_metric("KLLKLLKLLK", "KLLKLFKLLK", verbose=True)
    results = batch_peptide_metric(refs, preds)

PACKAGE CONVERSION
------------------
To turn this single file into an installable package later:

    peptidebleu/
    ├── __init__.py          ← re-exports peptide_metric, batch_peptide_metric
    ├── core.py              ← this file, renamed
    ├── tables.py            ← move VALID_AA / AA_CHARGE / … lookup dicts here
    ├── components/
    │   ├── __init__.py
    │   ├── ngram.py         ← ngram_bleu_score
    │   ├── physicochemical.py ← charge, hydrophobicity
    │   ├── functional.py    ← functional_group, property_distribution
    │   └── structural.py    ← structural_penalty
    ├── weights/
    │   ├── __init__.py
    │   └── learner.py       ← move weight_learner.py content here
    └── align.py             ← align_sequences stub → NW implementation

EXTENSIBILITY HOOKS (grep "# HOOK:" to find all extension points)
-------------------------------------------------------------------
HOOK:BLOSUM   — replace modified_precision() with BLOSUM62-weighted version
HOOK:ALIGN    — replace align_sequences() body with Needleman-Wunsch
HOOK:WEIGHTS  — replace DEFAULT_WEIGHTS with output of weight_learner.py
HOOK:PKA      — replace AA_CHARGE with Henderson-Hasselbalch fractional charges

BIOCHEMICAL DATA SOURCE
-----------------------
All AA properties (charge, hydrophobicity, property class, functional group)
are taken verbatim from the attached rule document (The amino acid property
table). Length/charge class thresholds follow Table 4-5 of the same document.

TESTED ON
---------
Ground-truth dataset: peptides_curated.csv  (105,510 sequences, lengths 6-100)
Activity labels: anti-bacterial, anti-cancer, anti-fungal, anti-parasitic,
                 anti-viral, cell-cell-communication, drug-delivery,
                 immunological, inhibitor, metabolic, other-functional,
                 signal-peptide, toxic

Version : 1.1.0
"""

# ---------------------------------------------------------------------------
# Standard library only — no external dependencies
# ---------------------------------------------------------------------------
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


# ===========================================================================
# SECTION 1 — BIOCHEMICAL LOOKUP TABLES
# (source: rule document "The amino acid property table")
# ===========================================================================

#: The 20 canonical amino acids accepted by this metric.
#: Any sequence containing characters outside this set is rejected.
VALID_AA: frozenset = frozenset("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# Charge at pH 7  (integer per residue)
# Histidine (H): pKa 6.0 — treated as uncharged at physiological pH 7.
# ---------------------------------------------------------------------------
AA_CHARGE: Dict[str, int] = {
    "A":  0, "C":  0, "D": -1, "E": -1, "F":  0,
    "G":  0, "H":  0, "I":  0, "K": +1, "L":  0,
    "M":  0, "N":  0, "P":  0, "Q":  0, "R": +1,
    "S":  0, "T":  0, "V":  0, "W":  0, "Y":  0,
}
# HOOK:PKA — replace with Henderson-Hasselbalch fractional charges:
#   AA_CHARGE["H"] = 1 / (1 + 10 ** (pH - 6.0))   at a given pH
#   Add a `pH` parameter to net_charge() and charge_similarity_score().

# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale
# ---------------------------------------------------------------------------
AA_HYDROPHOBICITY: Dict[str, float] = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# ---------------------------------------------------------------------------
# Property class (used by functional_group and property_distribution scores)
# ---------------------------------------------------------------------------
AA_PROPERTY_CLASS: Dict[str, str] = {
    "A": "hydrophobic",      "C": "special",           "D": "negative",
    "E": "negative",         "F": "hydrophobic",        "G": "special",
    "H": "positive",         "I": "hydrophobic",        "K": "positive",
    "L": "hydrophobic",      "M": "hydrophobic",        "N": "polar_uncharged",
    "P": "special",          "Q": "polar_uncharged",    "R": "positive",
    "S": "polar_uncharged",  "T": "polar_uncharged",    "V": "hydrophobic",
    "W": "hydrophobic",      "Y": "polar_aromatic",
}

# Ordered list of all property classes (used as stable vector axis)
PROPERTY_CLASSES: List[str] = [
    "hydrophobic", "polar_uncharged", "positive",
    "negative",    "special",         "polar_aromatic",
]

# ---------------------------------------------------------------------------
# Functional group (for soft-matching credit)
# ---------------------------------------------------------------------------
AA_FUNCTIONAL_GROUP: Dict[str, str] = {
    "A": "methyl",         "C": "thiol",           "D": "carboxylate",
    "E": "carboxylate",    "F": "phenyl",           "G": "none",
    "H": "imidazole",      "I": "branched_alkyl",  "K": "epsilon_amino",
    "L": "branched_alkyl", "M": "thioether",        "N": "amide",
    "P": "cyclic",         "Q": "amide",            "R": "guanidinium",
    "S": "hydroxyl",       "T": "hydroxyl",         "V": "isopropyl",
    "W": "indole",         "Y": "phenol",
}

# ---------------------------------------------------------------------------
# Soft-matching similarity between property classes  [0.0 – 1.0]
# Based on biochemical relatedness; missing pairs default to 0.0.
# ---------------------------------------------------------------------------
PROPERTY_CLASS_SIMILARITY: Dict[Tuple[str, str], float] = {
    # Exact matches
    ("hydrophobic",     "hydrophobic"):     1.0,
    ("polar_uncharged", "polar_uncharged"): 1.0,
    ("positive",        "positive"):        1.0,
    ("negative",        "negative"):        1.0,
    ("special",         "special"):         1.0,
    ("polar_aromatic",  "polar_aromatic"):  1.0,
    # Cross-class partial credits
    ("polar_aromatic",  "polar_uncharged"): 0.6,
    ("polar_uncharged", "polar_aromatic"):  0.6,
    ("polar_aromatic",  "hydrophobic"):     0.3,
    ("hydrophobic",     "polar_aromatic"):  0.3,
    ("positive",        "polar_uncharged"): 0.2,
    ("polar_uncharged", "positive"):        0.2,
    ("negative",        "polar_uncharged"): 0.2,
    ("polar_uncharged", "negative"):        0.2,
    ("special",         "hydrophobic"):     0.1,
    ("hydrophobic",     "special"):         0.1,
}

# ---------------------------------------------------------------------------
# Activity-class thresholds  (from rule document Table 4-5)
# Used by classify_sequence() and for informational reporting.
# ---------------------------------------------------------------------------
LENGTH_CLASSES = [
    (2,   9,   "very_short",    "Cell-cell communication, neuropeptides"),
    (10,  50,  "core_window",   "AMP, CPP, immunological, inhibitor"),
    (15,  30,  "signal_window", "Signal peptide"),
    (51,  100, "long",          "Metabolic, predominantly non-functional"),
    (101, None,"protein",       "Protein (excluded from peptide benchmarks)"),
]

CHARGE_CLASSES = [
    (None, -1,  "anionic",           "Non-functional or highly specific (6% AMPs)"),
    (-1,   1,   "near_neutral",      "Cell-cell comm., metabolic hormones, immunological"),
    (1,    3,   "weak_cationic",     "Anti-viral, anti-cancer, immunological"),
    (2,    5,   "moderate_cationic", "Core AMP: anti-bacterial, anti-fungal, anti-parasitic"),
    (3,    9,   "strongly_cationic", "Drug-delivery (CPP)"),
]


# ===========================================================================
# SECTION 2 — UTILITY FUNCTIONS
# ===========================================================================

def is_valid_sequence(seq: str) -> bool:
    """Return True iff *seq* is non-empty and every character is in VALID_AA."""
    return bool(seq) and all(aa in VALID_AA for aa in seq)


def sanitize(seq: str) -> str:
    """Strip whitespace and upper-case. Does NOT filter invalid characters."""
    return seq.strip().upper()


def net_charge(seq: str) -> float:
    """
    Net charge at pH 7:  Q_net = Σ C(rᵢ)
    Equivalent to: count(K) + count(R) − count(D) − count(E).
    Histidine treated as uncharged (pKa 6.0 < pH 7).
    """
    return float(sum(AA_CHARGE.get(aa, 0) for aa in seq))


def mean_hydrophobicity(seq: str) -> float:
    """Mean Kyte-Doolittle hydrophobicity: H_peptide = (1/n) Σ H(rᵢ)."""
    if not seq:
        return 0.0
    return sum(AA_HYDROPHOBICITY.get(aa, 0.0) for aa in seq) / len(seq)


def hydrophobic_fraction(seq: str) -> float:
    """Fraction of residues whose property class is 'hydrophobic'."""
    if not seq:
        return 0.0
    return sum(1 for aa in seq
               if AA_PROPERTY_CLASS.get(aa) == "hydrophobic") / len(seq)


def property_class_vector(seq: str) -> Dict[str, float]:
    """
    Normalised frequency distribution over PROPERTY_CLASSES.
    Returns a dict {class: fraction} summing to 1.0.
    """
    counts = Counter(AA_PROPERTY_CLASS.get(aa, "special") for aa in seq)
    n = len(seq) or 1
    return {c: counts.get(c, 0) / n for c in PROPERTY_CLASSES}


def classify_sequence(seq: str) -> Dict[str, str]:
    """
    Return a human-readable dictionary of biochemical class assignments
    for *seq* according to the rule document thresholds.
    Useful for debugging and reporting.
    """
    n     = len(seq)
    q     = net_charge(seq)
    hf    = hydrophobic_fraction(seq)

    # Length class
    lclass = "unknown"
    for lo, hi, name, _ in LENGTH_CLASSES:
        if hi is None:
            lclass = name if n >= lo else lclass
        elif lo <= n <= hi:
            lclass = name

    # Charge class
    cclass = "anionic" if q < 0 else "neutral"
    for lo, hi, name, _ in CHARGE_CLASSES:
        lo_ok = (lo is None) or (q >= lo)
        hi_ok = (hi is None) or (q <= hi)
        if lo_ok and hi_ok:
            cclass = name

    # Hydrophobicity class
    if hf >= 0.50:
        hclass = "high_hydrophobic (toxic / signal h-region)"
    elif hf >= 0.40:
        hclass = "AMP_core (40-49%)"
    else:
        hclass = f"below_AMP_core ({hf:.0%})"

    return {
        "length":        str(n),
        "length_class":  lclass,
        "net_charge":    f"{q:+.0f}",
        "charge_class":  cclass,
        "hydro_fraction": f"{hf:.2%}",
        "hydro_class":   hclass,
        "mean_hydro":    f"{mean_hydrophobicity(seq):+.3f}",
    }


# ===========================================================================
# SECTION 3 — ALIGNMENT HOOK
# ===========================================================================

def align_sequences(ref: str, pred: str,
                    gap_char: str = "-",
                    gap_penalty: float = -1.0) -> Tuple[str, str]:
    """
    Global sequence alignment.

    Currently a *stub* returning the sequences unchanged (identity alignment).
    Replace the body with a Needleman-Wunsch implementation to make
    functional_group_similarity_score alignment-aware.

    # HOOK:ALIGN — Example NW skeleton:
    #
    #   n, m   = len(ref), len(pred)
    #   score  = [[0]*(m+1) for _ in range(n+1)]
    #   trace  = [[None]*(m+1) for _ in range(n+1)]
    #   for i in range(1, n+1): score[i][0] = i * gap_penalty
    #   for j in range(1, m+1): score[0][j] = j * gap_penalty
    #   for i in range(1, n+1):
    #       for j in range(1, m+1):
    #           match = score[i-1][j-1] + substitution(ref[i-1], pred[j-1])
    #           delete= score[i-1][j]   + gap_penalty
    #           insert= score[i][j-1]   + gap_penalty
    #           score[i][j] = max(match, delete, insert)
    #           trace[i][j] = ... (traceback pointer)
    #   # traceback → aligned_ref, aligned_pred
    #
    # `substitution(a, b)` can use BLOSUM62 for residue-aware scoring.

    Parameters
    ----------
    ref, pred      : raw peptide sequences
    gap_char       : character used for inserted gaps (default '-')
    gap_penalty    : linear gap score (not used in stub)

    Returns
    -------
    (aligned_ref, aligned_pred) — same as input in stub mode
    """
    return ref, pred


# ===========================================================================
# SECTION 4 — METRIC COMPONENTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Component 1 · N-gram BLEU with Brevity Penalty
# ---------------------------------------------------------------------------

def _get_ngrams(seq: str, n: int) -> Counter:
    """Extract all contiguous n-grams as a Counter."""
    return Counter(seq[i:i+n] for i in range(len(seq) - n + 1))


def brevity_penalty(ref_len: int, pred_len: int) -> float:
    """
    BLEU-style brevity penalty (BP):
        BP = 1                         if pred_len >= ref_len
        BP = exp(1 − ref_len/pred_len) otherwise
    Penalises under-generation (prediction shorter than reference).
    """
    if pred_len == 0:
        return 0.0
    if pred_len >= ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / pred_len)


def modified_precision(ref: str, pred: str, n: int) -> float:
    """
    Clipped n-gram precision.
    For each n-gram in the prediction, its count is capped by its count in
    the reference.  Divides by total prediction n-gram count.

    # HOOK:BLOSUM — Replace exact gram matching with BLOSUM62-weighted:
    #
    #   For each pred_gram, find the ref_gram that maximises
    #       BLOSUM62_score(pred_gram, ref_gram) / BLOSUM62_max_score
    #   and use that fractional credit instead of 0/1 clipping.
    #   This gives partial credit to biochemically conservative substitutions
    #   (e.g. I/L/V all score high against each other).
    """
    pred_ngrams = _get_ngrams(pred, n)
    ref_ngrams  = _get_ngrams(ref,  n)
    total_pred  = sum(pred_ngrams.values())
    if total_pred == 0:
        return 0.0
    clipped = sum(
        min(cnt, ref_ngrams.get(gram, 0))
        for gram, cnt in pred_ngrams.items()
    )
    return clipped / total_pred


def ngram_bleu_score(
    ref:     str,
    pred:    str,
    max_n:   int = 3,
    weights: Optional[List[float]] = None,
) -> float:
    """
    Modified BLEU over amino-acid n-grams (n = 1 … max_n).

    Captures:
    - Unigrams  (n=1): residue identity frequency
    - Bigrams   (n=2): local dipeptide motifs
    - Trigrams  (n=3): short structural/functional motifs

    Parameters
    ----------
    ref, pred : peptide sequences
    max_n     : maximum n-gram order (default 3)
    weights   : per-order weights summing to 1. Default: uniform 1/max_n.

    Returns
    -------
    float in [0, 1]
    """
    if not pred or not ref:
        return 0.0
    if weights is None:
        weights = [1.0 / max_n] * max_n

    log_avg = 0.0
    for n, w in enumerate(weights, start=1):
        if n > len(pred) or n > len(ref):
            log_avg += w * math.log(1e-10)
            continue
        p = modified_precision(ref, pred, n)
        log_avg += w * math.log(max(p, 1e-10))

    bp    = brevity_penalty(len(ref), len(pred))
    score = bp * math.exp(log_avg)
    return float(min(max(score, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Component 2 · Charge Similarity
# ---------------------------------------------------------------------------

def charge_similarity_score(
    ref:   str,
    pred:  str,
    decay: float = 0.5,
) -> float:
    """
    Similarity based on net charge at pH 7.

        score = exp(−decay × |Q_net(ref) − Q_net(pred)|)

    The exponential decay maps absolute charge differences to [0, 1]:
    - |ΔQ| = 0  →  1.00  (identical charge)
    - |ΔQ| = 1  →  0.61  (one residue off)
    - |ΔQ| = 2  →  0.37
    - |ΔQ| = 4  →  0.14

    Dataset statistics (peptides_curated.csv):
        mean Q = +2.54,  std = 5.19

    Parameters
    ----------
    decay : float
        Sensitivity to charge difference. Larger → stricter penalty.
        Recommended range: 0.3 – 0.8.
        Can be learned via weight_learner.py.

    Returns
    -------
    float in (0, 1]
    """
    q_ref  = net_charge(ref)
    q_pred = net_charge(pred)
    return math.exp(-decay * abs(q_ref - q_pred))


# ---------------------------------------------------------------------------
# Component 3 · Hydrophobicity Similarity
# ---------------------------------------------------------------------------

def hydrophobicity_similarity_score(
    ref:   str,
    pred:  str,
    decay: float = 0.3,
) -> float:
    """
    Similarity based on mean Kyte-Doolittle hydrophobicity.

        score = exp(−decay × |H_peptide(ref) − H_peptide(pred)|)

    Dataset statistics (peptides_curated.csv):
        mean H = -0.26,  std = 0.87

    Parameters
    ----------
    decay : float
        Sensitivity to hydrophobicity difference.
        Recommended range: 0.2 – 0.5.

    Returns
    -------
    float in (0, 1]
    """
    h_ref  = mean_hydrophobicity(ref)
    h_pred = mean_hydrophobicity(pred)
    return math.exp(-decay * abs(h_ref - h_pred))


# ---------------------------------------------------------------------------
# Component 4 · Functional Group / Soft Class Matching
# ---------------------------------------------------------------------------

def functional_group_similarity_score(ref: str, pred: str) -> float:
    """
    Positional soft matching using PROPERTY_CLASS_SIMILARITY.

    For each aligned position (up to the shorter sequence length), the
    similarity between the reference and prediction residue's property class
    is looked up from PROPERTY_CLASS_SIMILARITY.  The average is then
    multiplied by a length-ratio penalty to discourage over/under-generation.

    This is the **soft matching** advanced component.  Unlike binary n-gram
    exact matching, biochemically conservative substitutions (e.g. K→R,
    I→L, S→T) receive substantial partial credit.

    Advanced component type: SOFT MATCHING (same functional group = partial credit)

    # HOOK:ALIGN — Replace positional pairing with alignment-aware version:
    #
    #   aligned_ref, aligned_pred = align_sequences(ref, pred)
    #   for r_aa, p_aa in zip(aligned_ref, aligned_pred):
    #       if r_aa == gap_char or p_aa == gap_char:  continue
    #       ...

    Returns
    -------
    float in [0, 1]
    """
    if not ref or not pred:
        return 0.0

    # Alignment hook (identity stub unless replaced)
    aligned_ref, aligned_pred = align_sequences(ref, pred)

    total_sim = 0.0
    n_compared = min(len(aligned_ref), len(aligned_pred))

    for r_aa, p_aa in zip(aligned_ref, aligned_pred):
        r_class = AA_PROPERTY_CLASS.get(r_aa, "special")
        p_class = AA_PROPERTY_CLASS.get(p_aa, "special")
        total_sim += PROPERTY_CLASS_SIMILARITY.get((r_class, p_class), 0.0)

    avg_sim    = (total_sim / n_compared) if n_compared > 0 else 0.0
    length_pen = (min(len(ref), len(pred)) /
                  max(len(ref), len(pred)))
    return float(avg_sim * length_pen)


# ---------------------------------------------------------------------------
# Component 5 · Property Distribution Similarity
# ---------------------------------------------------------------------------

def _smooth_distribution(dist: Dict[str, float],
                          epsilon: float = 1e-6) -> Dict[str, float]:
    """Add epsilon to all classes to avoid log(0) in KL computation."""
    total = sum(dist.values()) + epsilon * len(dist)
    return {k: (v + epsilon) / total for k, v in dist.items()}


def property_distribution_score(ref: str, pred: str) -> float:
    """
    Distribution-level comparison via symmetric KL-divergence.

        KL_sym = 0.5 × [KL(P_ref ‖ P_pred) + KL(P_pred ‖ P_ref)]
        score  = exp(−KL_sym)

    This captures whether the prediction has a similar **proportion** of
    each residue class (hydrophobic, polar, charged, special) — order-
    independent.  A prediction with all-hydrophobic residues substituted
    for all-positive residues would score zero here even if positional
    soft matching gave it some credit.

    Advanced component type: DISTRIBUTION-LEVEL COMPARISON

    Returns
    -------
    float in (0, 1]
    """
    if not ref or not pred:
        return 0.0

    p_ref  = _smooth_distribution(property_class_vector(ref))
    p_pred = _smooth_distribution(property_class_vector(pred))
    all_classes = set(p_ref) | set(p_pred)

    def kl(p: Dict, q: Dict) -> float:
        return sum(
            p.get(c, 1e-9) * math.log(p.get(c, 1e-9) / q.get(c, 1e-9))
            for c in all_classes
        )

    kl_sym = 0.5 * (kl(p_ref, p_pred) + kl(p_pred, p_ref))
    return float(math.exp(-kl_sym))


# ---------------------------------------------------------------------------
# Component 6 · Structural Penalty
# ---------------------------------------------------------------------------

def structural_penalty_score(ref: str, pred: str) -> float:
    """
    Structural plausibility score based on:

    1. Proline positional penalty
       Proline locks the backbone nitrogen (cyclic side chain), disrupting
       α-helices and β-turns at internal positions.  Spurious prolines in the
       prediction (positions not in the reference) that are internal (pos > 1)
       incur a fractional penalty.

    2. Cysteine pairing awareness
       An even cysteine count supports disulphide bonding.  If the prediction
       introduces an odd Cys count while the reference has an even count, a
       fixed penalty is applied (unpaired thiol → structural instability,
       aberrant redox chemistry).

    Advanced component types: STRUCTURAL PENALTIES

        raw = 1 − (pro_penalty + cys_penalty),   clamped to [0, 1]

    Returns
    -------
    float in [0, 1]
    """
    if not pred:
        return 0.0

    # ── Proline penalty ──────────────────────────────────────────────────────
    pred_pro = {i for i, aa in enumerate(pred) if aa == "P"}
    ref_pro  = {i for i, aa in enumerate(ref)  if aa == "P"}
    # Internal spurious prolines only (N-terminal Pro is tolerated)
    spurious_internal = {p for p in (pred_pro - ref_pro) if p > 1}
    pro_penalty = len(spurious_internal) / max(len(pred), 1)

    # ── Cysteine pairing penalty ─────────────────────────────────────────────
    cys_penalty = 0.0
    if (pred.count("C") % 2 == 1) and (ref.count("C") % 2 == 0):
        cys_penalty = 0.15

    return float(min(max(1.0 - (pro_penalty + cys_penalty), 0.0), 1.0))


# ===========================================================================
# SECTION 5 — WEIGHT CONFIGURATION
# ===========================================================================

#: Default weights — biochemically motivated, tuned for the AMP/CPP task.
#: Weights sum to exactly 1.0.
#:
#: Rationale:
#:   ngram_bleu (0.25)  — sequence-level fidelity, motif preservation
#:   charge     (0.20)  — primary AMP determinant (membrane disruption depends
#:                        on cationic character); mean Q=+2.54 in dataset
#:   hydrophobicity (0.20) — membrane insertion / signal peptide function
#:   functional_group (0.15) — residue-level biochemical fitness
#:   property_distribution (0.10) — global composition sanity
#:   structural (0.10)  — structural plausibility penalty
#:
#: To use learned weights:
#:   from weight_learner import run_weight_learning
#:   learned = run_weight_learning("peptides_curated.csv")
#:   score = peptide_metric(ref, pred, weights=learned["weights"])
#:
#: HOOK:WEIGHTS — replace this dict with output of weight_learner.py

DEFAULT_WEIGHTS: Dict[str, float] = {
    "ngram_bleu":            0.25,
    "charge":                0.20,
    "hydrophobicity":        0.20,
    "functional_group":      0.15,
    "property_distribution": 0.10,
    "structural":            0.10,
}
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, \
    "DEFAULT_WEIGHTS must sum to 1.0"

# Activity-class preset weights — use when the target activity is known.
# Each preset emphasises the physicochemical properties most relevant to
# that activity class per the rule document.
ACTIVITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Anti-bacterial / anti-fungal / anti-parasitic  →  charge-dominant AMP window
    "amp": {
        "ngram_bleu":            0.15,
        "charge":                0.30,
        "hydrophobicity":        0.25,
        "functional_group":      0.15,
        "property_distribution": 0.10,
        "structural":            0.05,
    },
    # Drug-delivery / CPP  →  strongly cationic (+3 to +9)
    "cpp": {
        "ngram_bleu":            0.10,
        "charge":                0.40,
        "hydrophobicity":        0.15,
        "functional_group":      0.15,
        "property_distribution": 0.10,
        "structural":            0.10,
    },
    # Signal peptide  →  high hydrophobicity (h-region), moderate length
    "signal": {
        "ngram_bleu":            0.15,
        "charge":                0.10,
        "hydrophobicity":        0.35,
        "functional_group":      0.15,
        "property_distribution": 0.15,
        "structural":            0.10,
    },
    # Anti-cancer / anti-viral / immunological  →  moderate cationic
    "immunological": {
        "ngram_bleu":            0.20,
        "charge":                0.25,
        "hydrophobicity":        0.20,
        "functional_group":      0.15,
        "property_distribution": 0.10,
        "structural":            0.10,
    },
}
# Verify all preset weights
for _act, _wts in ACTIVITY_WEIGHTS.items():
    assert abs(sum(_wts.values()) - 1.0) < 1e-9, \
        f"ACTIVITY_WEIGHTS['{_act}'] must sum to 1.0"


def get_weights(
    activity: Optional[str] = None,
    custom:   Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Retrieve the weight dict to use for scoring.

    Priority order: custom > activity preset > DEFAULT_WEIGHTS.

    Parameters
    ----------
    activity : str | None
        One of 'amp', 'cpp', 'signal', 'immunological' (or None).
    custom   : dict | None
        A full weight dict whose values sum to 1.0.

    Returns
    -------
    Dict[str, float]
    """
    if custom is not None:
        if abs(sum(custom.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"Custom weights sum to {sum(custom.values()):.6f}, expected 1.0"
            )
        return custom
    if activity is not None:
        if activity not in ACTIVITY_WEIGHTS:
            raise ValueError(
                f"Unknown activity '{activity}'. "
                f"Valid options: {list(ACTIVITY_WEIGHTS.keys())}"
            )
        return ACTIVITY_WEIGHTS[activity]
    return DEFAULT_WEIGHTS


# ===========================================================================
# SECTION 6 — MAIN SCORING FUNCTIONS
# ===========================================================================

#: Internal ordered list of component keys (defines computation order)
_COMPONENT_KEYS: List[str] = [
    "ngram_bleu",
    "charge",
    "hydrophobicity",
    "functional_group",
    "property_distribution",
    "structural",
]

#: Human-readable component names for display
_COMPONENT_LABELS: Dict[str, str] = {
    "ngram_bleu":            "N-gram BLEU (w/ brevity penalty)",
    "charge":                "Charge similarity (K+R−D−E rule)",
    "hydrophobicity":        "Hydrophobicity similarity (Kyte-Doolittle)",
    "functional_group":      "Functional group / soft class matching",
    "property_distribution": "Property distribution (KL-divergence)",
    "structural":            "Structural penalty (Pro+Cys)",
}


def _compute_components(ref: str, pred: str) -> Dict[str, float]:
    """Compute all six components. Assumes both sequences are valid."""
    return {
        "ngram_bleu":            ngram_bleu_score(ref, pred),
        "charge":                charge_similarity_score(ref, pred),
        "hydrophobicity":        hydrophobicity_similarity_score(ref, pred),
        "functional_group":      functional_group_similarity_score(ref, pred),
        "property_distribution": property_distribution_score(ref, pred),
        "structural":            structural_penalty_score(ref, pred),
    }


def peptide_metric(
    reference:  str,
    prediction: str,
    weights:    Optional[Dict[str, float]] = None,
    activity:   Optional[str]              = None,
    verbose:    bool                       = False,
) -> float:
    """
    Compute the PeptideBLEU composite score.

    Score = Σ (wᵢ × componentᵢ),   wᵢ ∈ DEFAULT_WEIGHTS

    Parameters
    ----------
    reference  : Ground-truth peptide sequence (single-letter codes).
    prediction : AI-generated peptide sequence to evaluate.
    weights    : Custom weight dict (values must sum to 1.0).
                 Overrides *activity* if both are provided.
    activity   : Activity-class preset ('amp','cpp','signal','immunological').
                 Uses DEFAULT_WEIGHTS if None.
    verbose    : Print per-component breakdown.

    Returns
    -------
    float in [0, 1].
    Returns 0.0 if either sequence contains invalid amino acids.

    Examples
    --------
    >>> peptide_metric("KLLKLLKLLK", "KLLKLFKLLK")
    0.9339
    >>> peptide_metric("RRWWKK", "RRWWDD", activity="amp", verbose=True)
    """
    active_weights = get_weights(activity=activity, custom=weights)

    # Sanitize
    ref  = sanitize(reference)
    pred = sanitize(prediction)

    # Hard validity guard — reject sequences with non-canonical amino acids
    if not is_valid_sequence(ref):
        invalid = sorted(set(aa for aa in ref if aa not in VALID_AA))
        if verbose:
            print(f"[INVALID REFERENCE]  Contains: {invalid}  → score = 0.0")
        return 0.0
    if not is_valid_sequence(pred):
        invalid = sorted(set(aa for aa in pred if aa not in VALID_AA))
        if verbose:
            print(f"[INVALID PREDICTION] Contains: {invalid}  → score = 0.0")
        return 0.0

    # Compute all components
    components = _compute_components(ref, pred)

    # Weighted sum
    final_score = sum(active_weights[k] * components[k]
                      for k in _COMPONENT_KEYS)

    # Verbose report
    if verbose:
        ref_info  = classify_sequence(ref)
        pred_info = classify_sequence(pred)
        w_name    = activity or "default"
        print("=" * 68)
        print(f"  PeptideBLEU — weights: [{w_name}]")
        print("-" * 68)
        print(f"  Reference : {ref}")
        print(f"    len={ref_info['length']}  Q={ref_info['net_charge']}  "
              f"({ref_info['charge_class']})  "
              f"H={ref_info['mean_hydro']}  "
              f"hydro%={ref_info['hydro_fraction']}")
        print(f"  Prediction: {pred}")
        print(f"    len={pred_info['length']}  Q={pred_info['net_charge']}  "
              f"({pred_info['charge_class']})  "
              f"H={pred_info['mean_hydro']}  "
              f"hydro%={pred_info['hydro_fraction']}")
        print("-" * 68)
        print(f"  {'Component':<42s} {'raw':>6}  {'w':>5}  {'contrib':>7}")
        print(f"  {'-'*42} {'------':>6}  {'-----':>5}  {'-------':>7}")
        for k in _COMPONENT_KEYS:
            raw  = components[k]
            w    = active_weights[k]
            cont = w * raw
            lbl  = _COMPONENT_LABELS[k]
            print(f"  {lbl:<42s} {raw:>6.4f}  {w:>5.2f}  {cont:>7.4f}")
        print("=" * 68)
        print(f"  FINAL PeptideBLEU Score :  {final_score:.4f}")
        print("=" * 68)

    return round(final_score, 6)


def score_components(
    reference:  str,
    prediction: str,
) -> Dict[str, float]:
    """
    Return the raw (unweighted) value of each component.

    Useful for inspecting where a prediction loses points, training
    a learned weight model, or building feature vectors.

    Returns
    -------
    dict with keys matching _COMPONENT_KEYS, each value in [0, 1].
    Returns all-zero dict if either sequence is invalid.
    """
    ref  = sanitize(reference)
    pred = sanitize(prediction)
    if not is_valid_sequence(ref) or not is_valid_sequence(pred):
        return {k: 0.0 for k in _COMPONENT_KEYS}
    return _compute_components(ref, pred)


# ===========================================================================
# SECTION 7 — BATCH EVALUATION
# ===========================================================================

def batch_peptide_metric(
    references:  List[str],
    predictions: List[str],
    weights:     Optional[Dict[str, float]] = None,
    activity:    Optional[str]              = None,
    verbose:     bool                       = False,
) -> Dict[str, object]:
    """
    Evaluate a list of (reference, prediction) pairs.

    Parameters
    ----------
    references, predictions : parallel lists of peptide sequences.
    weights    : custom weight dict (see peptide_metric).
    activity   : activity-class preset (see peptide_metric).
    verbose    : print each pair's breakdown.

    Returns
    -------
    dict:
        scores       — list of per-pair PeptideBLEU scores
        mean         — corpus-level mean score
        std          — standard deviation
        min          — worst-case score
        max          — best-case score
        valid_frac   — fraction of predictions with only valid AAs
        components   — dict of per-component mean scores (useful for analysis)
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"references ({len(references)}) and predictions "
            f"({len(predictions)}) must have equal length."
        )

    scores: List[float] = []
    n_valid = 0
    component_sums: Dict[str, float] = {k: 0.0 for k in _COMPONENT_KEYS}

    for ref, pred in zip(references, predictions):
        s = peptide_metric(ref, pred, weights=weights,
                           activity=activity, verbose=verbose)
        scores.append(s)

        pred_clean = sanitize(pred)
        if is_valid_sequence(pred_clean):
            n_valid += 1
            for k, v in score_components(ref, pred).items():
                component_sums[k] += v

    n = len(scores)
    mean_score = sum(scores) / n if n else 0.0
    variance   = (sum((s - mean_score) ** 2 for s in scores) / n) if n else 0.0
    n_valid_   = max(n_valid, 1)

    return {
        "scores":     scores,
        "mean":       round(mean_score, 6),
        "std":        round(math.sqrt(variance), 6),
        "min":        round(min(scores), 6) if scores else 0.0,
        "max":        round(max(scores), 6) if scores else 0.0,
        "valid_frac": round(n_valid / n, 4) if n else 0.0,
        "components": {
            k: round(component_sums[k] / n_valid_, 4)
            for k in _COMPONENT_KEYS
        },
    }


# ===========================================================================
# SECTION 8 — EXAMPLE USAGE  (python peptide_bleu.py)
# ===========================================================================

if __name__ == "__main__":
    SEP = "#" * 70

    # ── Example 1: Conservative substitution ────────────────────────────────
    print(f"\n{SEP}\n  Example 1 — Conservative substitution  (L→F, hydrophobic→hydrophobic)\n{SEP}")
    peptide_metric("KLLKLLKLLK", "KLLKLFKLLK", verbose=True)

    # ── Example 2: Charge-altering substitution ──────────────────────────────
    print(f"\n{SEP}\n  Example 2 — Charge-altering substitution  (K→D: +1 → -1)\n{SEP}")
    peptide_metric("KLLKLLKLLK", "DLLKLLKLLK", verbose=True)

    # ── Example 3: Length mismatch (brevity penalty) ─────────────────────────
    print(f"\n{SEP}\n  Example 3 — Length mismatch  (prediction under-generates)\n{SEP}")
    peptide_metric("KLLKLLKLLKLL", "KLLKLLK", verbose=True)

    # ── Example 4: Invalid amino acid ────────────────────────────────────────
    print(f"\n{SEP}\n  Example 4 — Invalid amino acid  ('B' not in VALID_AA)\n{SEP}")
    peptide_metric("KLLKLLK", "KLLBLLK", verbose=True)

    # ── Example 5: Activity-class preset weights ─────────────────────────────
    print(f"\n{SEP}\n  Example 5 — Activity-class preset weights  [amp]\n{SEP}")
    peptide_metric("RRWWKK", "RRWWDD", activity="amp", verbose=True)

    # ── Example 6: Signal peptide preset ─────────────────────────────────────
    print(f"\n{SEP}\n  Example 6 — Signal peptide preset  [signal]\n{SEP}")
    peptide_metric("MLLLLLLLLLAAA", "MLLLLLLLLLAAG", activity="signal", verbose=True)

    # ── Example 7: score_components (feature vector) ─────────────────────────
    print(f"\n{SEP}\n  Example 7 — score_components()  (raw feature vector)\n{SEP}")
    feats = score_components("KLLKLLKLLK", "KLLKLFKLLK")
    for k, v in feats.items():
        print(f"    {k:<28s}  {v:.4f}")

    # ── Example 8: Batch evaluation ──────────────────────────────────────────
    print(f"\n{SEP}\n  Example 8 — batch_peptide_metric()\n{SEP}")
    refs  = ["KLLKLLKLLK", "RRWWKK",  "ACDEFGHIKLMN", "WNLLRQAQEKFGKDKSPK"]
    preds = ["KLLKLFKLLK", "RRWWDD",  "ACDEFGHIKLMN", "WNLLRQAQEKFGKNKSPK"]
    res   = batch_peptide_metric(refs, preds)
    print(f"  Per-pair scores : {[round(s, 4) for s in res['scores']]}")
    print(f"  Corpus mean     : {res['mean']:.4f}  ±{res['std']:.4f}")
    print(f"  Best / Worst    : {res['max']:.4f} / {res['min']:.4f}")
    print(f"  Valid fraction  : {res['valid_frac']:.2%}")
    print(f"  Component means : {res['components']}")
