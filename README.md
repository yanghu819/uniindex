# uniindex

The mainline is the minimal SigLIP-VQ unified FLM reproduction in
`unified.py`.

Current clean result:

- tokenizer: `inclusionAI/LLaDA2.0-Uni` SigLIP-VQ image tokens
- model: one shared bidirectional Transformer denoiser, one shared output head
- text: label text tokens in the same sequence as image tokens
- objective: shared FLM denoising loss on image and text positions
- best recorded image-to-text exact: `0.8671875` at progress `0.5`
- best recorded image-to-text token accuracy: `0.93359375`

Text-to-image generation is still not solved. The best clean long run improves
image-token diversity, but text-conditioned generation remains weak:
conditioned token-label accuracy `0.07500000298023224`, unique generated token
count `769`, generated-vs-real histogram L1 `0.4066070318222046`.
`unified.py acceptance` deliberately fails the generation gate instead of hiding
this.

## Layout

- `unified.py`: single-file mainline and reproduction contract
- `configs/main.yaml`: generated mainline config
- `setup.sh`: create the `uv` environment and install pinned dependencies
- `down.sh`: download and prepare datasets, tokenizer assets, and the evaluation classifier
- `run.sh`: compatibility wrapper around the package CLI
- `src/uniindex/`: implementation modules used by `unified.py`
- `configs/smoke.yaml`: tiny local config with a dummy tokenizer for fast checks
- `docs/MAINLINE.md`: current mainline decision record

## Quick Start

```bash
./setup.sh
python unified.py about
python unified.py acceptance
python unified.py config --out configs/main.yaml
python unified.py commands
python unified.py prepare
python unified.py train --stage stage1
python unified.py train --stage stage2
python unified.py eval
python unified.py visualize
```

## Notes

- The main config uses SigLIP-VQ assets cached under repo-local `.cache/`.
- The smoke config swaps in a tiny deterministic dummy tokenizer for fast local validation.
- All runtime metadata is written under `runs/<run_id>/metadata.json`.
- `run.sh` and `unified.py` pin `PYTHONPATH` to the local `src/` tree so detached worktrees do not accidentally import another editable install.
