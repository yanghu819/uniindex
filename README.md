# uniindex

The mainline is now the minimal SigLIP-VQ unified FLM reproduction in
`unified.py`.

Current best understanding result:

- tokenizer: `inclusionAI/LLaDA2.0-Uni` SigLIP-VQ image tokens
- model: one shared bidirectional Transformer denoiser, one shared output head
- text: one label token per class
- key change: one image semantic token sourced from clean VQ-token distributions
- best recorded i2t diagnostic: exact `0.8828125` at progress `0.5`
- semantic hidden label probe: `0.9609375`

Generation is still not solved. This main branch intentionally keeps the
working understanding route simple and reproducible before adding more moving
parts.

## Layout

- `unified.py`: single-file mainline and reproduction contract
- `configs/main.yaml`: generated mainline config
- `setup.sh`: create the `uv` environment and install pinned dependencies
- `down.sh`: download and prepare datasets, tokenizer assets, and the evaluation classifier
- `run.sh`: compatibility wrapper around the older package CLI
- `src/uniindex/`: historical implementation modules used by `unified.py`
- `configs/smoke.yaml`: tiny local config with a dummy tokenizer for fast checks
- `docs/active_baseline.md`: active branch, baseline config, and current best result

## Quick start

```bash
./setup.sh
python unified.py about
python unified.py config --out configs/main.yaml
python unified.py commands
python unified.py prepare
python unified.py train --stage stage1
python unified.py train --stage stage2
python unified.py diagnose
python unified.py probe
```

## Notes

- The main config uses SigLIP-VQ assets cached under repo-local `.cache/`.
- The smoke config swaps in a tiny deterministic dummy tokenizer for fast local validation.
- All runtime metadata is written under `runs/<run_id>/metadata.json`.
- `run.sh` and `unified.py` pin `PYTHONPATH` to the local `src/` tree so detached worktrees do not accidentally import another editable install.
