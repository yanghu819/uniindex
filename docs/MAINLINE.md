# Mainline

The main branch is the clean SigLIP-VQ unified FLM route.

## Why This Is Main

The useful result is not a special MNIST shortcut. The current mainline keeps
the scalable pieces:

- SigLIP-VQ image tokens
- label text tokens as normal text input
- one shared sequence
- one bidirectional Transformer denoiser
- one shared output head
- one FLM denoising objective

Deleted paths include auxiliary supervised side channels, alternate text decoders,
mid-sampler projection tricks, label-only scoring gates, and per-sequence
scoring losses. Those were useful as diagnostics, but they are not the scalable
idea.

## Best Clean Result

- code SHA: `7ff026bc1a939c2f7e5f377bb72b00012e4807f0`
- config: `configs/flm_joint_work_siglipvq_generation_labeltoken.yaml`
- image-to-text exact at progress `0.5`: `0.8671875`
- image-to-text token accuracy at progress `0.5`: `0.93359375`
- text-to-image conditioned token-label accuracy: `0.07500000298023224`
- generated unique token count: `769`
- generated-vs-real histogram L1: `0.4066070318222046`

## Current Reproduction

Use the single-file entrypoint:

```bash
python unified.py about
python unified.py config --out configs/main.yaml
python unified.py prepare
python unified.py train --stage stage1
python unified.py train --stage stage2
python unified.py eval
python unified.py visualize
```

`configs/main.yaml` is generated from `unified.py`.

## What Is Not Solved

Text-to-image generation is still the active blocker. Image-to-text has a real
signal in the clean setup, but text conditioning does not yet reliably control
decoded generation.

`python unified.py acceptance` is expected to return a nonzero exit status until
the clean generation gate is actually met.
