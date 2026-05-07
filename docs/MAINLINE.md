# UniIndex Mainline

Updated: 2026-05-07 Asia/Shanghai

## Active Version

Use `configs/main.yaml` as the active mainline config.

This is an alias of the current best clean unified FLM route:

- historical config: `configs/flm_words_joint_long_rope_p8_gen124_gpu80.yaml`
- model: unified denoiser with a shared RoPE Transformer backbone
- objective: shared FLM denoising objective
- task mix: `joint:text_to_image:image_to_text = 1:2:4`
- text representation: class names as text, for example `zero`, `one`, `two`
- current best checkpoint family: `gen124` step 5000

## Current State

- `image -> text` understanding is working and is used as a protection metric.
- `text -> image` generation has real decoder output, but still suffers late-stage collapse into `9/1/7`.
- The next research question is not whether the decoder works; it is whether late denoise / final readout collapse can be fixed in a scalable unified FLM setup.

## Deprecated Paths

Deprecated files are kept for reproducibility, but they are not active entrypoints:

- `scripts/deprecated/qwen_fallback/`: Qwen or causal-style fallback attempts. Do not use for the FLM mainline.
- `configs/deprecated/anti_collapse_failed/`: low-LR reset, joint-anchor, and t2i time-power experiments that did not solve collapse.
- `configs/deprecated/superseded_generation_mixes/`: generation mix variants superseded by `gen124`.

Do not silently switch to a deprecated route. Any paradigm fallback needs an explicit decision first.
