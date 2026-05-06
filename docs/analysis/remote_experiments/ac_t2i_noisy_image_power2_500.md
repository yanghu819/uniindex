# ac_t2i_noisy_image_power2_500

- Status: `ok`
- Started: `2026-05-06T08:36:20Z`
- Finished: `2026-05-06T08:56:16Z`
- Commit SHA: `57781de6437096f2c3857688cef2f46be8e23a9b`
- Config: `configs/flm_words_joint_long_rope_p8_ac_t2i_noisy_image_power2_gpu80.yaml`
- Base checkpoint: `/fangxueji/Projects/PG/uniindex/worktrees/gpu-fd41e0b/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `True`
- Override LR: `0.0001`

## Hypothesis

Collapse may start in early denoising from high-noise image states; increasing the t2i image time-power during training may make text conditioning matter earlier.

## Metrics

- i2t 512 exact: `0.84765625`
- i2t 512 token acc: `0.8837485172004745`
- t2i 512 token-NN acc: `0.314453125`
- t2i nearest-label distribution: `{9: 201, 1: 156, 7: 145, 4: 10}`
- decoded 160 token-NN acc: `0.2750000059604645`
- decoded nearest-label distribution: `{9: 73, 7: 44, 1: 40, 4: 3}`
- decoded grid: `/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/logs/remote_experiments/ac_t2i_noisy_image_power2_500/decoded/decoded_grid_steps256_temp0.70.png`

## Acceptance

- i2t exact pass: `True`
- visual review: `pending`
- final acceptance: `pending_visual_review`
- note: Token metrics are only a first pass; final pass requires manual decoded-grid visual note.

## Next Step Basis

If decoded images follow prompts better and nearest labels spread beyond 9/1/7, run a small power sweep next.
