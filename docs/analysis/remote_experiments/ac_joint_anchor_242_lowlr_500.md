# ac_joint_anchor_242_lowlr_500

- Status: `running`
- Started: `2026-05-06T07:16:02Z`
- Finished: `2026-05-06T07:17:10Z`
- Commit SHA: `803af6887b51914ef4eb9e9e799e63335087b50a`
- Config: `configs/flm_words_joint_long_rope_p8_ac_joint_anchor_242_lowlr_gpu80.yaml`
- Base checkpoint: `runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `True`
- Override LR: `0.0001`

## Hypothesis

More joint tasks should anchor the unified multimodal manifold and reduce pure t2i drift into a few digit attractors.

## Metrics

- i2t 512 exact: `n/a`
- i2t 512 token acc: `n/a`
- t2i 512 token-NN acc: `n/a`
- t2i nearest-label distribution: `n/a`
- decoded 160 token-NN acc: `n/a`
- decoded nearest-label distribution: `n/a`
- decoded grid: `n/a`

## Acceptance

- i2t exact pass: `n/a`
- visual review: `pending`
- final acceptance: `n/a`
- note: n/a

## Next Step Basis

If nearest-label mass spreads beyond 9/1/7 while i2t remains >=80%, this becomes the first scale-up candidate.
