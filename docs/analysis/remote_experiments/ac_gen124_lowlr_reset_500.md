# ac_gen124_lowlr_reset_500

- Status: `ok`
- Started: `2026-05-06T07:52:50Z`
- Finished: `2026-05-06T08:15:25Z`
- Commit SHA: `57781de6437096f2c3857688cef2f46be8e23a9b`
- Config: `configs/flm_words_joint_long_rope_p8_ac_gen124_lowlr_reset_gpu80.yaml`
- Base checkpoint: `/fangxueji/Projects/PG/uniindex/worktrees/gpu-fd41e0b/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `true`
- Override LR: `0.0001`

## Hypothesis

step 6500 generation regression may come from high LR or inherited optimizer momentum pushing the model toward a 9 attractor; low LR plus reset optimizer should preserve gen124 prompt-following while reducing drift.

## Metrics

- i2t 512 exact: `0.8515625`
- i2t 512 token acc: `0.8888888888888888`
- t2i 512 token-NN acc: `0.314453125`
- t2i nearest-label distribution: `{"1":150,"4":10,"7":129,"9":223}`
- decoded 160 token-NN acc: `0.29375001788139343`
- decoded nearest-label distribution: `{"1":44,"7":40,"9":76}`
- decoded grid: `/fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500/logs/remote_experiments/ac_gen124_lowlr_reset_500/decoded/decoded_grid_steps256_temp0.70.png`

## Acceptance

- i2t exact pass: `true`
- visual review: `done`
- final acceptance: `fail`
- visual note: Decoded grid shows partial prompt following for easy labels such as one/seven and some four/five/six/eight shapes, but zero/two/three/five/six/eight still frequently drift to 9/1/7. Decoded NN distribution is entirely 9/1/7, so generation collapse is not solved.
- note: i2t stays above the protection threshold, but decoded images and nearest-label distributions still show strong text-to-image collapse. This run should not be scaled as-is.

## Next Step Basis

Do not extend this setting yet; first diagnose where t2i denoising collapses and why the image-token vector field prefers 9/1/7 attractors.
