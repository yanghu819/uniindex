# ac_joint_anchor_242_lowlr_500

- Status: `ok`
- Started: `2026-05-06T08:15:32Z`
- Finished: `2026-05-06T08:36:12Z`
- Commit SHA: `57781de6437096f2c3857688cef2f46be8e23a9b`
- Config: `configs/flm_words_joint_long_rope_p8_ac_joint_anchor_242_lowlr_gpu80.yaml`
- Base checkpoint: `/fangxueji/Projects/PG/uniindex/worktrees/gpu-fd41e0b/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `true`
- Override LR: `0.0001`

## Hypothesis

More joint tasks should anchor the unified multimodal manifold and reduce pure t2i drift into a few digit attractors.

## Metrics

- i2t 512 exact: `0.83203125`
- i2t 512 token acc: `0.8722815342032424`
- t2i 512 token-NN acc: `0.291015625`
- t2i nearest-label distribution: `{"1":135,"4":10,"7":166,"9":201}`
- decoded 160 token-NN acc: `0.26250001788139343`
- decoded nearest-label distribution: `{"1":39,"4":4,"7":50,"9":67}`
- decoded grid: `/fangxueji/Projects/PG/uniindex/worktrees/ac_joint_anchor_242_lowlr_500/logs/remote_experiments/ac_joint_anchor_242_lowlr_500/decoded/decoded_grid_steps256_temp0.70.png`

## Acceptance

- i2t exact pass: `true`
- visual review: `done`
- final acceptance: `fail`
- visual note: Decoded grid does not improve over gen124 low-LR reset. It is slightly more seven-heavy, with most prompts still visually and by NN falling into 9/7/1 plus a tiny 4 tail.
- note: i2t stays above the protection threshold, but decoded images and nearest-label distributions still show strong text-to-image collapse. This run should not be scaled as-is.

## Next Step Basis

Do not scale the 2:4:2 joint-anchor direction; more joint mass alone did not change the attractor structure.
