# Active Baseline

The active baseline is now the clean SigLIP-VQ unified FLM mainline described
in `docs/MAINLINE.md` and implemented by `unified.py`.

## Current Main Result

- code SHA: `7ff026bc1a939c2f7e5f377bb72b00012e4807f0`
- image tokens: `inclusionAI/LLaDA2.0-Uni` SigLIP-VQ
- text: label text tokens
- model: shared bidirectional Transformer denoiser
- head: one shared output head
- objective: shared FLM denoising loss
- image-to-text exact at progress `0.5`: `0.8671875`
- image-to-text token accuracy at progress `0.5`: `0.93359375`
- text-to-image conditioned token-label accuracy: `0.07500000298023224`
- generated unique token count: `769`
- generated-vs-real histogram L1: `0.4066070318222046`

## Decision

Keep the project centered on the unified FLM idea. The clean run has a real
image-to-text signal, but text-to-image conditioning remains the bottleneck.

The old diagnostic routes are intentionally no longer active. They moved short
term numbers, but they injected task-specific side channels or changed the
problem into a smaller supervised side task.

## Canonical Execution

```bash
python unified.py about
python unified.py config --out configs/main.yaml
python unified.py prepare
python unified.py train --stage stage1
python unified.py train --stage stage2
python unified.py eval
python unified.py visualize
```
