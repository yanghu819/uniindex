# uniindex

`uniindex` is a pre-research implementation of a unified FLM multimodal model:

- vision uses a frozen `BAAI/Emu3.5-VisionTokenizer`
- labels are encoded as text tokens such as `zero`, `one`, `two`
- one RoPE Transformer denoiser models the full unified image/text sequence
- training runs in two stages:
  - `stage1`: joint denoising over image tokens plus text tokens
  - `stage2`: mixed replay with `joint`, `text->image`, and `image->text`

The repository is designed for single-GPU execution with all caches and outputs kept inside the repo root.

## Mainline

The current clean mainline is:

- config: `configs/main.yaml`
- equivalent historical config: `configs/flm_words_joint_long_rope_p8_gen124_gpu80.yaml`
- architecture: unified FLM / RoPE Transformer / shared backbone / shared denoising objective
- current best generation checkpoint family: `gen124` step 5000

Deprecated experiments are kept under `configs/deprecated/` and `scripts/deprecated/`.
They are not deleted, but they should not be used as the active path.

## Layout

- `setup.sh`: create the `uv` environment and install pinned dependencies
- `down.sh`: download and prepare datasets, tokenizer assets, and the evaluation classifier
- `run.sh`: run `smoke`, `stage1`, `stage2`, or `eval`
- `src/uniindex/`: Python package
- `configs/main.yaml`: main config for the A100 unified FLM path
- `configs/default.yaml`: older baseline config retained for compatibility
- `configs/smoke.yaml`: tiny local config with a dummy tokenizer for fast checks

## Quick start

```bash
./setup.sh
./down.sh --config configs/main.yaml
./run.sh smoke
./run.sh stage1 --config configs/main.yaml
./run.sh stage2 --config configs/main.yaml
./run.sh eval --config configs/main.yaml
```

## Notes

- The main config uses the official Emu3.5 VisionTokenizer via repo-local cache; do not rely on online downloads during remote experiments.
- The smoke config swaps in a tiny deterministic dummy tokenizer so local validation does not depend on a large model download.
- All runtime metadata is written under `runs/<run_id>/metadata.json`.
