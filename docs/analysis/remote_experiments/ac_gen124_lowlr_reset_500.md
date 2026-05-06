# ac_gen124_lowlr_reset_500

- Status: `ok`
- Started: `2026-05-06T07:52:50Z`
- Finished: `2026-05-06T08:15:25Z`
- Commit SHA: `57781de6437096f2c3857688cef2f46be8e23a9b`
- Config: `configs/flm_words_joint_long_rope_p8_ac_gen124_lowlr_reset_gpu80.yaml`
- Base checkpoint: `/fangxueji/Projects/PG/uniindex/worktrees/gpu-fd41e0b/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `True`
- Override LR: `0.0001`

## Hypothesis

step 6500 generation regression may come from high LR or inherited optimizer momentum pushing the model toward a 9 attractor; low LR plus reset optimizer should preserve gen124 prompt-following while reducing drift.

## Metrics

- i2t 512 exact: `0.8515625`
- i2t 512 token acc: `0.8888888888888888`
- t2i 512 token-NN acc: `0.314453125`
- t2i nearest-label distribution: `{9: 223, 1: 150, 7: 129, 4: 10}`
- decoded 160 token-NN acc: `0.29375001788139343`
- decoded nearest-label distribution: `{9: 76, 1: 44, 7: 40}`
- decoded grid: `/fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500/logs/remote_experiments/ac_gen124_lowlr_reset_500/decoded/decoded_grid_steps256_temp0.70.png`

## Acceptance

- i2t exact pass: `True`
- visual review: `pending`
- final acceptance: `pending_visual_review`
- note: Token metrics are only a first pass; final pass requires manual decoded-grid visual note.

## Next Step Basis

If i2t stays >=80% and t2i does not regress, extend to 1000 steps before broader scale-up.
