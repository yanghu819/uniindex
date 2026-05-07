# 2026-05-07 Minimal Old Generation Setting Verification

## Goal

Verify, without new training, which old setting produced MNIST-like text-to-image generations and whether the current SigLIP-VQ decoder is itself broken.

## Old Emu3.5 Candidate Baseline

- Remote worktree: `/fangxueji/Projects/PG/uniindex/worktrees/gpu-fd41e0b`
- Worktree state: detached `2c202b7614e9e9ce99b3b40af2bd871017abef4f` with uncommitted code edits and untracked generation configs/scripts.
- Config: `configs/flm_words_joint_long_rope_p8_gen124_gpu80.yaml`
- Checkpoint: `runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Tokenizer: `emu3p5`, `BAAI/Emu3.5-VisionTokenizer`
- Image token layout: `16x16`, `256` image tokens, `codebook_size=55553`, `compact_vocab=true`
- Data: MNIST `train_limit=30000`, `test_limit=5000`
- Model: `d_model=256`, `n_layers=6`, `n_heads=8`, `position_encoding=rope`
- Train setting: `batch_size=128`, `stage1_steps=1200`, `stage2_steps=8000`, `joint:t2i:i2t=1:2:4`
- Schedule: `text_time_power=0.25`, `image_to_text_text_time_power=8.0`
- Sampling: `256` steps, temperature `0.7`, `4` seeds per label for this quick verification
- Local grid copy: `logs/visualizations/old_gen124_minverify_20260507T_min/decoded_grid_steps256_temp0.70.png`
- Result: generated images are clearly MNIST-like, but prompt following is weak. Token-NN accuracy is `0.275`, and nearest labels collapse to `9/1/7` only: `9=17`, `1=12`, `7=11`.

## Current SigLIP-VQ Comparisons

- Clean10k visual:
  - Config: `configs_generated/clean10k_quick_visualize.yaml`
  - Checkpoint: `.../siglipvq_minimal_unified_longer_img512/.../checkpoints/stage2_latest.pt`
  - Tokenizer: `siglip_vq`, `inclusionAI/LLaDA2.0-Uni`
  - Image token layout: `32x32`, `1024` image tokens, `codebook_size=16384`, `compact_vocab=false`
  - Local grid copy: `logs/visualizations/clean10k_quick_9c09a1e_20260507T062913Z/text_to_image_prompt_grid.png`
  - Visual note: high-frequency texture/glyph-like outputs, not reliable digit prompt following.

- Scaled d512/l12 visual:
  - Config: `configs_generated/formal_d512_l12_b80_resume_from_step300_2h.yaml`
  - Checkpoint: `.../scale-d512-l12-batch90-63c4d99/.../checkpoints/stage2_latest.pt`
  - Tokenizer: `siglip_vq`, `inclusionAI/LLaDA2.0-Uni`
  - Model: `d_model=512`, `n_layers=12`, `n_heads=8`
  - Image token layout: `32x32`, `1024` image tokens
  - Local grid copy: `logs/visualizations/scale_d512_l12_b80_resume2h_20260507T100736Z/text_to_image_prompt_grid.png`
  - Visual note: smoother blob-like outputs than step300, but still not recognizable prompt-conditioned digits.

## Decoder Sanity

- Config: `configs_generated/formal_d512_l12_b80_resume_from_step300_2h.yaml`
- Tokenizer: `siglip_vq`, `inclusionAI/LLaDA2.0-Uni`
- Image token layout: `32x32`, `1024` image tokens, `codebook_size=16384`, `image_size=512`
- Test: decode real SigLIP-VQ MNIST test-set tokens, not generated tokens.
- Local grid copy: `logs/visualizations/siglip_real_token_decode_sanity_20260507T_min/siglip_real_token_reconstruction_grid.png`
- Result: real SigLIP-VQ tokens decode to clear MNIST digits. The decoder path is therefore functional; generated SigLIP-VQ outputs fail because the FLM generated token grids are off-manifold or weakly conditioned.

## Conclusion

The old setting can be reproduced as a MNIST-like generation baseline, but it is not solved prompt-following. The most likely useful old settings are Emu3.5 tokens, shorter `256`-token image sequence, `compact_vocab=true`, `30k` data, RoPE, stronger t2i ratio, and long `256`-step sampling. The current SigLIP-VQ route has a harder `1024`-token image space and produces off-manifold image token grids even though its real-token decoder works.
