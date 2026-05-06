# ac_gen124_lowlr_reset_500

- Status: `failed`
- Started: `2026-05-06T07:33:07Z`
- Finished: `2026-05-06T07:44:07Z`
- Commit SHA: `bfad1b32ef84bea0fc27f8e5292fb4de6f61f5df`
- Config: `configs/flm_words_joint_long_rope_p8_ac_gen124_lowlr_reset_gpu80.yaml`
- Base checkpoint: `/fangxueji/Projects/PG/uniindex/worktrees/gpu-fd41e0b/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `True`
- Override LR: `0.0001`

## Hypothesis

step 6500 generation regression may come from high LR or inherited optimizer momentum pushing the model toward a 9 attractor; low LR plus reset optimizer should preserve gen124 prompt-following while reducing drift.

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

If i2t stays >=80% and t2i does not regress, extend to 1000 steps before broader scale-up.

## Failure

- Stage: `read_i2t`
- Exit code: `None`
- Reason: failed to parse remote JSON /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500/logs/remote_experiments/ac_gen124_lowlr_reset_500/eval/i2t_512.json: Expecting value: line 1 column 1 (char 0)
- Command: `None`

Recent output:

```text
      55
    ],
    [
      "five",
      54
    ],
    [
      "three",
      53
    ],
    [
      "nine",
      50
    ],
    [
      "seven",
      49
    ],
    [
      "two",
      46
    ],
    [
      "zero",
      42
    ],
    [
      "six",
      35
    ],
    [
      "sixht",
      3
    ],
    [
      "throe",
      1
    ]
  ]
}
```
