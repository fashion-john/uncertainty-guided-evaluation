# Uncertainty-Guided Low-Resource LLM Evaluation

Minimal code for three-view answer evaluation and uncertainty-guided question selection.

## Reproduce the reported results

Python 3.10+; no dependencies, GPU, network access or API key needed:

```bash
python reproduce.py --check
python -m unittest -q
```

`python reproduce.py` regenerates the five CSV tables in `results/` from
`data/predictions.csv.gz`. The archive contains 21,924 per-question records:
parsed predictions for all three views, gold labels, option identity mappings,
and original candidate probabilities where used. Question texts, credentials,
model weights and server logs are not included.

| Dataset | Questions |
| --- | ---: |
| MedQA | 1,259 |
| MedMCQA | 1,000 |
| MMLU-Med | 895 |
| OpenBookQA (validation) | 500 |

Full evaluation uses Gemini-2.5-Flash, Qwen2.5-7B-Instruct, Gemma-3-12B-IT,
Llama3-Med42-8B, DeepSeek-V4-Flash and DeepSeek-V4-Pro. Risk selection uses
the first four models. The archived DeepSeek candidate distributions are
insufficient for this comparison; no DeepSeek risk results are generated.

| Output | Experiment |
| --- | --- |
| `table51.csv` | Original accuracy, three-view strict accuracy, gap and answer consistency |
| `table52.csv` | Original-error recall at 10%, 20%, 30%, 40% entropy selection |
| `table53.csv` | Absolute strict-accuracy approximation errors at the four budgets |
| `table54.csv` | Entropy / Margin / Max-prob: new-error recall@30, Lift@10 and error@30 |
| `table55.csv` | Original accuracy and two-view strict accuracies for each perturbation |

## Protocol

`N` counts original questions. `C` counts correct original answers, `S` counts
questions correct in all three views, and `H = C - S` counts newly exposed
errors: original correct, at least one perturbation wrong, counted once.

Select `k = round(x*N)` questions from all `N` questions, using Python's
ties-to-even rounding. Entropy ranks high to low; margin and maximum probability
rank low to high. Score ties retain `source_order` from the archived item list.
If the selection contains `h` newly exposed errors:

```text
full strict accuracy = 100*S/N
approximate strict accuracy = 100*(C-h)/N
absolute approximation error = 100*(H-h)/N       [percentage points]
new-error recall = 100*h/H                      [%]
Lift = (h/k)/(H/N)
model calls = N + 2*k                           [full evaluation: 3*N]
```

Table 5.2 uses original errors, with denominator `N-C`. Table 5.4 uses newly
exposed errors, with denominator `H`. Undefined recall/lift values are blank.
Per-dataset error is an absolute difference; Table 5.3's macro row is the
equal-weight mean of four unrounded dataset errors. Table 5.4's Entropy
error@30 equals Table 5.3's error@30 for each model and dataset.

Labels A/B/C/D become I/II/III/IV. Medical position perturbations retain the
benchmark's first-matching-correct-text swap with D; original D answers are
unchanged. OpenBookQA uses its archived nonidentity permutation seeded by
`fixed-pos-{item_id}`. Original populations are retained, including duplicate
option texts and positional references. Consistency compares mapped original
option identities and includes consistent wrong answers. Empty or ambiguous
answers count as incorrect and remain in `N`.

The four-label probabilities are normalized before scoring. Thirteen archived
records contain missing-label probability imputation, identified by
`probability_imputed`. Entropy, margin and maximum probability are recalculated
from the probabilities; no saved score or metric columns are trusted as input.

## Run a new evaluation

Prepare the same question populations from [ReMedQA](https://github.com/disi-unibo-nlp/remedqa)
and [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa).
Benchmark and model licenses apply to their respective upstream resources.

```bash
git clone https://github.com/disi-unibo-nlp/remedqa.git
git -C remedqa checkout 2caa66392f5d7c9632d8c24d6a9391df79c6555a
pip install datasets
python prepare.py --remedqa remedqa
```

Only the four datasets and three views are prepared; content checksums must
match `data/manifest.json`. `--openbookqa PATH` accepts an existing three-view
OpenBookQA JSON instead of downloading it.

For the three local models, use a Linux CUDA environment with vLLM:

```bash
pip install vllm
python run.py --backend vllm --model /path/to/checkpoint --model-name Qwen2.5-7B-Instruct --dataset medqa --data data/benchmarks/medqa.json --output runs/qwen_medqa.csv
```

Local inference uses bfloat16, greedy decoding, the checkpoint's chat template
when available, and the original one-token constrained label decoding. Provider
inference uses greedy direct-label prompts; Gemini thinking is disabled. For
Gemini-native or OpenAI-compatible endpoints, set `MODEL_API_KEY` in the
environment and use `--backend gemini` or `--backend openai`, supplying the
provider's model ID with `--model` and endpoint with `--api-base`.

`--dry-run` validates a benchmark and prints a prompt without calling a model.
Completed questions are resumed automatically. `--limit` supports smoke runs;
table generation requires complete dataset populations. New inference results
depend on checkpoint/provider versions; the bundled archive fixes the results
used for these tables.

```bash
python reproduce.py --input runs/qwen_medqa.csv --output results_new
```

The four Python entry points are `prepare.py`, `run.py`, `reproduce.py` and
`test_protocol.py`; their shared definitions are in `protocol.py`.
