# 2026-05-07 Quick Visual Check: clean10k SigLIP-VQ Unified FLM

## Run

- Timestamp: 2026-05-07T06:29:13Z
- Code SHA: `9c09a1ea9729a3889f1c0cbf2e5460a101f6a23e`
- Remote: AIStation `A100`, `root@172.16.78.10 -p 30186`
- Worktree: `/fangxueji/Projects/PG/uniindex/worktrees/quickvis-9c09a1e`
- Config: `configs_generated/clean10k_quick_visualize.yaml`
- Base checkpoint:
  `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/models/siglipvq_minimal_unified_longer_img512/20260428T095050Z-minimal-unified-gaussian-continue8000/continue-2000-plus-8000/checkpoints/stage2_latest.pt`

## Command

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ../../.venv/bin/python -m uniindex.cli quick-visualize \
  --config configs_generated/clean10k_quick_visualize.yaml \
  --seeds-per-label 1 \
  --steps 32
```

## Output

- Remote grid:
  `/fangxueji/Projects/PG/uniindex/worktrees/quickvis-9c09a1e/runs/clean10k_quick_visualize/20260507T062913Z-quick-visualize/visuals/text_to_image_prompt_grid.png`
- Local display copy:
  `logs/visualizations/clean10k_quick_9c09a1e_20260507T062913Z/text_to_image_prompt_grid.png`

The local display copy is an artifact for inspection and is not committed.

## Result

- The generated decoded images are high-frequency glyph-like textures.
- They are not reliably recognizable as the conditioned digit prompts.
- There is no convincing per-prompt visual control from `zero` through `nine`.
- This supports the current conclusion: image-to-text understanding is working, but text-to-image generation remains the main open problem.

## Engineering Note

The quick visual path itself is useful, but the SigLIP-VQ decoder path is still too slow. The decoder emits repeated initialization warnings and appears to reload heavy decode components per image. The next engineering fix should cache/reuse the decoder model for visualization, without changing the FLM training objective or adding classifier/probe scoring.
