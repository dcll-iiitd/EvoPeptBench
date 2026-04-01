# EvoPeptBench

EvoPeptBench is a specialized benchmarking framework for evaluating Large Language Models (LLMs) on peptide sequence generation tasks.

## Setup

First, make sure to clean up any incompatible libraries (especially if switching to GPU evaluation). You can remove related default packages via:
```bash
pip uninstall -y torch torchvision torchaudio numpy transformers
```

Install the dependencies:
```bash
pip install -r requirements.txt
```
This strictly enforces:
- `numpy==1.26.4`
- `torch==2.1.2` (CUDA 11.8 compatible)
- `transformers==4.38.2`

## Run Evaluation

### GPU Evaluation (Recommended)
You can run evaluation on a GPU device explicitly using `CUDA_VISIBLE_DEVICES`. The following runs the evaluation in batched mode on a single GPU. Note that your tasks dataset is now located under `data/processed/`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_gpu.py \
    --tasks data/processed/peptides_with_length.jsonl \
    --mode WITH_LENGTH \
    --k 5 \
    --gpu-batch-size 64 \
    --out results_gpu.json
```

### CPU Evaluation
For debugging or environments without a GPU, run the CPU-only evaluation script:

```bash
python scripts/evaluate_cpu.py \
    --tasks data/processed/peptides_with_length.jsonl \
    --mode WITH_LENGTH \
    --k 5 \
    --out results_cpu.json
```

## Documentation

Information on the biochemical rules and property tables used to score metric outputs can be found in `docs/aminoacid-rules.md`. Complete dataset information resides in `data/`.
