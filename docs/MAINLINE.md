# Mainline

The main branch is the SigLIP-VQ unified FLM semantic-token route.

## Why This Is Main

The Emu3.5 route made decoded generation visible, but understanding only
became reliable after switching image tokens to SigLIP-VQ and adding one
semantic image token sourced directly from clean VQ-token distributions.

Best recorded understanding result:

- code SHA: `b96d5f195340650e0082725b57af1aa9ad85420c`
- config: `configs/flm_joint_work_siglipvq_generation_labeltoken_semantic_vqtoken_probe.yaml`
- i2t exact at progress `0.5`: `0.8828125`
- i2t token accuracy at progress `0.5`: `0.94140625`
- semantic hidden label probe: `0.9609375`

## Current Reproduction

Use the single-file entrypoint:

```bash
python unified.py about
python unified.py config --out configs/main.yaml
python unified.py prepare
python unified.py train --stage stage1
python unified.py train --stage stage2
python unified.py diagnose
python unified.py probe
```

`configs/main.yaml` keeps the same algorithmic recipe with cleaner path names.

## What Is Not Solved

Text-to-image generation is still the active blocker. The full SigLIP-VQ guard
confirmed that image-to-text understanding is real, while pixel-level
generation remained broken. The next research step should improve unified
generation without replacing the shared FLM backbone.

`python unified.py acceptance` is expected to return a nonzero exit status until
the clean generation gate is actually met.
