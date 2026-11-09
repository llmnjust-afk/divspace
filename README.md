# divspace

Locating the latent subspace that governs output diversity in language models, and
testing whether intervening on it is reliable.

The repository implements an end-to-end, inference-only pipeline: it samples
generations, ranks them by intrinsic dispersion, fits a per-layer direction that
separates high- from low-dispersion responses, attributes the effect across depth
with attribution patching, and then measures what happens when that direction is
injected, ablated, or replaced by a placebo.

Everything runs against locally hosted open-weight checkpoints. There are no API
calls, no paid services, and no newly collected human annotations: the contrast
signal comes from the model's own outputs, and every diversity metric is either
lexical or computed from the model's own hidden states.

## What the code is for

Three questions drive the implementation, and each maps to one stage:

1. **Where** does output diversity change inside the network? Answered by
   attribution patching over depth (`divspace attribute`), which is cheap enough
   to run on every layer of a 7B model on a single 24GB card.
2. **What** is the axis? Answered by a contrastive fit over response activations
   (`divspace extract`), with a split-half stability check that flags directions
   that are noise.
3. **Is the intervention real?** Answered by placebo controls (`--control random`,
   `shuffled`, `orthogonal`) and by ablation (`--mode ablate`).

The third point is where most steering papers are weakest. Recent work has shown
that SAE-based interventions can be undone downstream (`Cui et al., 2026`) and
that residual-stream steering behaves differently in attention than in the
residual stream (`Kang et al., 2026`). A direction that raises a diversity metric
without surviving a placebo control is not a finding, so the controls are part of
the default workflow rather than an afterthought.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# optional: pip install -e '.[metrics]'   # enables MAUVE
```

The test suite runs on CPU with a randomly initialised 4-layer model and downloads
nothing:

```bash
pytest -q
```

## Quickstart on one GPU

```bash
# 1. what fits, and how fast
python -m divspace.cli probe --model Qwen/Qwen2.5-1.5B-Instruct --device cuda:0

# 2. fit directions at three depths
python -m divspace.cli extract \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda:0 \
  --prompt-set all --n-samples 16 \
  --out results/qwen15b/directions.pt

# 3. attribute the contrast to layers
python -m divspace.cli attribute \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda:0 \
  --artifact results/qwen15b/directions.pt \
  --out results/qwen15b/attribution.json

# 4. sweep the layer x coefficient grid
python -m divspace.cli steer \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda:0 \
  --artifact results/qwen15b/directions.pt \
  --coefs -3 -2 -1 1 2 3 --include-baseline \
  --out results/qwen15b/sweep.jsonl

# 5. score it
python -m divspace.cli metrics \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda:0 \
  --results results/qwen15b/sweep.jsonl \
  --artifact results/qwen15b/directions.pt \
  --out results/qwen15b/metrics_sweep.json
```

## Using all four GPUs

Two axes of parallelism are exposed, and they compose.

**Model-parallel**: one process per card, one model per process. This is the
useful axis here, because the workloads are inference-bound and each card can host
a complete 7B model in bf16.

```bash
./scripts/01_extract.sh          # 4 models over 4 GPUs
./scripts/02_attribute.sh
./scripts/03_steer.sh
./scripts/04_controls.sh
./scripts/05_metrics.sh
```

**Prompt-sharded**: split the prompt set inside a single model.

```bash
for S in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$S python -m divspace.cli steer \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
    --artifact results/qwen7b/directions.pt \
    --shard-id $S --num-shards 4 \
    --out results/qwen7b/sweep_shard$S.jsonl &
done
wait
python -m divspace.cli merge results/qwen7b/sweep_shard*.jsonl --out results/qwen7b/sweep.jsonl
```

Prompt ids are remapped to their global index before writing, so shards can be
concatenated without collisions.

For a declarative queue, `divspace-launch` pins one task per card:

```bash
divspace-launch --tasks configs/tasks.example.json --gpus 0 1 2 3 --log-dir runs
```

## Method details

**Intrinsic contrast.** For each prompt we draw `n_samples` generations at
temperature 0.9 and embed each response by mean-pooling the residual stream at a
mid-depth layer of the same model that produced it. A sample's score is its mean
cosine distance to the other samples of that prompt. The top quantile forms the
positive class, the bottom quantile the negative class, and selection is done
*within* a prompt so that prompt difficulty cannot leak into the contrast. No
external labelling is involved.

**Direction.** `d_l = normalise(mean(A+_l) - mean(A-_l))`, where `A_l` are
teacher-forced activations pooled over response tokens at layer `l`. With
`--raw-direction` the unnormalised vector is kept instead.

**Reliability.** `split_half_stability` refits the direction on two disjoint
halves of the contrast set and reports their cosine. A direction whose halves
disagree is a sampling artefact, not an axis, and should be dropped before any
sweep is interpreted.

**Attribution.** Two forward passes and one backward pass. The gradient is taken
at the low-dispersion run and multiplied by the clean-minus-corrupt activation
difference, summed over response-token positions. This is the first-order
approximation of Kramar et al. (2024), which avoids one forward pass per layer.

## Metrics

| Family | Metrics | Why |
|---|---|---|
| Lexical | distinct-1/2/3, rep-3, type-token ratio, self-BLEU | cheap, familiar, easy to game |
| Distributional | Vendi score, mean pairwise distance, effective rank, anisotropy | treat the sample set, not the n-grams |
| Semantic | cluster ratio, normalised cluster entropy | the quantity "homogenisation" usually means |
| Quality | completion rate, empty rate, length spread, degeneracy | diversity from broken text is not diversity |

MAUVE is available as an optional comparison against a reference set
(`--mauve-reference`), since it needs a second distribution.

## Layout

```
divspace/
  models.py        tokenizer and model loading, architecture-agnostic layer lookup
  hooks.py         capture / inject / ablate on decoder layers
  extract/         direction fitting and intrinsic contrast selection
  attribute/       attribution patching and logit lens
  intervene/       grids and placebo controls
  generate/        batched sampling with optional intervention
  metrics/         lexical, distributional, semantic, quality
  cli.py           one subcommand per stage
  launch.py        one task per GPU
configs/           model registry, extraction and steering defaults
scripts/           the four-GPU workflow, stage by stage
analysis/          paper figures
tests/             CPU-only, no downloads
```

## Deliberate omissions

No API-backed models, no human evaluation, and no new annotated dataset. The
consequence is a limit on the claims that can be made: everything here measures
*self-consistency and self-diversity*, not how a human would rate two responses as
meaningfully different. Any paper built on this code must say so in its
limitations section, and should report the quality column alongside every
diversity number.

## Prior work this is built against

- Karouzos, Tan & Aletras (2026), *Where does output diversity collapse in post-training?* — temporal attribution; this repository addresses the spatial axis instead.
- Jain et al. (2025), *Task-Dependent Evaluation of LLM Output Homogenization* — taxonomy-driven measurement.
- Yun et al. (2025), *The Price of Format: Diversity Collapse in LLMs* — format tokens as a mechanism.
- Zhu et al. (2026), *Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering* — steering for diversity.
- Cui et al. (2026), *SAE Interventions are Unreliable* — motivates the reliability controls.
- Kang et al. (2026), *Prompt-Activation Duality* — motivates attention-level as well as residual-level injection.
- Kramar et al. (2024), attribution patching — the attribution estimator used here.

## License

MIT.
