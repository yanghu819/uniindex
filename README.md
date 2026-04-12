# uniindex

`uniindex` is a toy implementation of a unified-index multimodal model:

- vision uses a frozen `BAAI/Emu3.5-VisionTokenizer`
- labels use one learned token per MNIST class
- one Transformer denoiser models the full unified sequence
- training runs in two stages:
  - `stage1`: joint denoising over image tokens plus label token
  - `stage2`: mixed replay with `joint`, `label->image`, and `image->label`

The repository is designed for single-GPU execution with all caches and outputs kept inside the repo root.

## Layout

- `setup.sh`: create the `uv` environment and install pinned dependencies
- `down.sh`: download and prepare datasets, tokenizer assets, and the evaluation classifier
- `run.sh`: run `smoke`, `stage1`, `stage2`, or `eval`
- `src/uniindex/`: Python package
- `configs/default.yaml`: main config for the A100 path
- `configs/smoke.yaml`: tiny local config with a dummy tokenizer for fast checks

## Quick start

```bash
./setup.sh
./down.sh --config configs/default.yaml
./run.sh smoke
./run.sh stage1
./run.sh stage2
./run.sh eval
```

## Notes

- The default config uses the official Emu3.5 VisionTokenizer via `transformers` remote code.
- The smoke config swaps in a tiny deterministic dummy tokenizer so local validation does not depend on a large model download.
- All runtime metadata is written under `runs/<run_id>/metadata.json`.
