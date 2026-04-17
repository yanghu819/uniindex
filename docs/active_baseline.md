# Active Baseline

`schedule-fix-layout-integ` is the only active code line for fullvocab runs.

## Active configs

- Long baseline: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- i2t power short sweep:
  - `configs/flm_joint_work_fullvocab_short_i2tp20_tsw075.yaml`
  - `configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml`
  - `configs/flm_joint_work_fullvocab_short_i2tp60_tsw075.yaml`

## Current best result

- `text_sequence_weight = 0.75`
- `image_to_text_text_time_power = 4.0`
- Eval path: `runs/fullvocab_long_tsw075/20260416T132256Z-eval/metrics.json`
- Metrics:
  - `image_to_text_exact_match = 0.078125`
  - `image_to_text_label_accuracy_constrained = 0.1015625`
  - `text_to_image_accuracy = 0.84375`
  - `unconditional_consistency = 0.734375`

## Canonical execution

- Use `./run.sh prepare|stage1|stage2|eval --config ...`
- Use `./run.sh sweep-i2t-power` for the default 2/4/6 short sweep and automatic long-run promotion
- `run.sh` exports `PYTHONPATH=$ROOT/src`, so every worktree resolves the local code instead of an unrelated editable install
- `configs_generated/` is runtime-only and should stay untracked

## Archived local trees

- `visualize-joint-work`
- `schedule-fix-analysis`
- `uniindex-formal-run`

Treat them as read-only references unless they are explicitly reactivated.
