# ac_t2i_noisy_image_power2_500

- Status: `ok`
- Started: `2026-05-06T08:36:20Z`
- Finished: `2026-05-06T08:56:16Z`
- Commit SHA: `57781de6437096f2c3857688cef2f46be8e23a9b`
- Config: `configs/flm_words_joint_long_rope_p8_ac_t2i_noisy_image_power2_gpu80.yaml`
- Base checkpoint: `/fangxueji/Projects/PG/uniindex/worktrees/gpu-fd41e0b/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `true`
- Override LR: `0.0001`

## Hypothesis

Collapse may start in early denoising from high-noise image states; increasing the t2i image time-power during training may make text conditioning matter earlier.

## Metrics

- i2t 512 exact: `0.84765625`
- i2t 512 token acc: `0.8837485172004745`
- t2i 512 token-NN acc: `0.314453125`
- t2i nearest-label distribution: `{"1":156,"4":10,"7":145,"9":201}`
- decoded 160 token-NN acc: `0.2750000059604645`
- decoded nearest-label distribution: `{"1":40,"4":3,"7":44,"9":73}`
- decoded grid: `/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/logs/remote_experiments/ac_t2i_noisy_image_power2_500/decoded/decoded_grid_steps256_temp0.70.png`

## Acceptance

- i2t exact pass: `true`
- visual review: `done`
- final acceptance: `fail`
- visual note: Decoded grid keeps one and seven readable, but zero/two/three/four/five/six/eight/nine are still pulled toward 9/7/1/4. The image-time power change did not produce prompt-following generation.
- note: i2t stays above the protection threshold, but decoded images and nearest-label distributions still show strong text-to-image collapse. This run should not be scaled as-is.

## Next Step Basis

Do not run a power sweep yet; first add diagnostics for per-step t2i trajectories and image-token marginal drift under the same unified FLM sampler.
