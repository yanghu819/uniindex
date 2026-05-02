# Robot Pre-Research Pivot

Timestamp: 2026-05-02T11:05:00+08:00

## Goal

Use the current A100 experiments as pre-research for a later large-scale robot setting, not as a target benchmark on a small toy dataset.

Short-term objective:
- turn the current pipeline into a reliable probe for image-to-text understanding mechanisms;
- identify which knobs should transfer to larger robot data and multi-GPU training;
- keep smoke tests cheap, but stop optimizing for toy benchmark leaderboard numbers.

Long-term objective:
- support scalable multimodal robot experiments where visual observations, language/state descriptions, and action-relevant semantics share one training/evaluation path;
- prepare the code and logging discipline needed before moving to hundreds or thousands of GPUs.

## What The MNIST Runs Mean

The 30k digit runs showed that the image-to-text pathway can learn a real conditional signal:

- `seq2, steps=8, temperature=1.0`: exact/constrained `0.8046`, token accuracy `0.9023`.
- `seq2, steps=16`: exact/constrained `0.8052`, token accuracy `0.9026`.
- `seq2, temperature=0.7`: exact/constrained `0.8068`, token accuracy `0.9034`.
- `seq4`: exact/constrained `0.7250`, token accuracy `0.8625`.

These are not the final research result. They are evidence for mechanism and failure modes:

- scaling data from 10k to 30k helped;
- doubling sampling steps barely helped, so inference steps are probably not the main bottleneck here;
- stronger sequence loss was not monotonic and biased the generated label distribution;
- temperature can slightly stabilize decoding but is a small effect.

## Main Constraint

The current bottleneck is not overfitting small datasets. The bottleneck is whether the method remains useful when:

- labels become longer and less canonical;
- observations are robot frames or embodied state, not clean MNIST digits;
- semantics require spatial, temporal, and action-relevant grounding;
- training moves from one A100 to distributed runs with strict reproducibility.

## What To Stop Doing

Do not spend more GPU time chasing small gains on MNIST exact match.

Cut for now:
- repeated temperature/step sweeps on toy data;
- longer MNIST-only runs unless they test a specific scalable mechanism;
- full image decode/classifier eval when token-only understanding eval answers the immediate question.

## Next Experiments

Priority 1: scale-relevant probes.
- Replace one-character labels with longer structured text targets.
- Add controlled ambiguity and synonym formats to test canonicalization.
- Measure exact match, constrained label accuracy, token accuracy, calibration, and class distribution skew.

Priority 2: data/interface readiness.
- Define a tokenized dataset contract for robot-like examples: image tokens, text/state tokens, optional action/task metadata, episode/time index.
- Add a minimal adapter so later robot datasets can enter the same train/eval loop without rewriting the model.

Priority 3: distributed readiness.
- Add run metadata that records git SHA, config hash, dataset artifact namespace, checkpoint namespace, GPU type, elapsed time, and offline/cache flags.
- Keep artifacts, checkpoints, and runs outside GitHub, but commit configs, scripts, and summarized metrics.

Priority 4: failure-mode diagnostics.
- Track generated text distribution skew.
- Track per-position text accuracy.
- Compare unconstrained decoding against constrained candidate scoring.
- Treat sequence loss as a sweepable risk, not an always-good regularizer.

## Reusable Decisions

- Use token-only eval for fast understanding checks.
- Keep tokenizer/data/checkpoint caches under `/fangxueji/Projects/PG/uniindex`.
- Use GitHub commits as the source of truth for code/config changes.
- Use toy data only as a smoke test and mechanism probe.
- Promote only findings that plausibly transfer to longer text, robot frames, temporal data, or distributed training.

## Current Action

Stop optimizing toy exact match. Next implementation should add a robot-style structured-text probe and dataset contract, then run it on A100 as a scalable proxy.
