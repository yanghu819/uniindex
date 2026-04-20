# Active Baseline

`schedule-fix-layout-integ` is the only active code line for fullvocab runs.

## Active configs

- Long baseline: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- i2t power short sweep:
  - `configs/flm_joint_work_fullvocab_short_i2tp20_tsw075.yaml`
  - `configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml`
  - `configs/flm_joint_work_fullvocab_short_i2tp60_tsw075.yaml`
- i2t repeats short sweep: generated from `configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml` by `./run.sh sweep-i2t-repeats`

## Current best result

- `text_sequence_weight = 0.75`
- training `image_to_text_text_time_power = 4.0`
- `stage2_joint_repeats = 2`
- `stage2_text_to_image_repeats = 2`
- `stage2_image_to_text_repeats = 6`
- default sampling:
  - `sampling.steps = 32`
  - `sampling.temperature = 0.7`
  - `sampling.image_to_text_text_time_power = 4.0`
  - `sampling.integrator = legacy_progress_euler`
  - `sampling.final_decode = final_model_call`
- Remote eval path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/fullvocab_long_tsw075_i2tr06/20260417T110458Z-eval/metrics.json`
- Original checkpoint metrics with the old sampling default:
  - `image_to_text_exact_match = 0.12109375`
  - `image_to_text_token_accuracy = 0.35990528808208366`
  - `image_to_text_label_accuracy_constrained = 0.15625`
  - `text_to_image_accuracy = 0.94140625`
  - `unconditional_consistency = 0.8125`
- Active sampler A/B eval path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampler_ab_20260420T083644Z_aggressive_i2t_legacy/20260420T085118Z-eval/metrics.json`
- Active sampler A/B metrics:
  - `image_to_text_exact_match = 0.1484375`
  - `image_to_text_token_accuracy = 0.36306235201262826`
  - `image_to_text_label_accuracy_constrained = 0.16796875`
  - `text_to_image_accuracy = 0.93359375`
  - `unconditional_consistency = 0.796875`

## Canonical execution

- Use `./run.sh prepare|stage1|stage2|eval --config ...`
- Use `./run.sh sweep-i2t-power` for the 2/4/6 short sweep over `image_to_text_text_time_power`
- Use `./run.sh sweep-i2t-repeats` for the 4/6/8 short sweep over `stage2_image_to_text_repeats`
- `run.sh` exports `PYTHONPATH=$ROOT/src`, so every worktree resolves the local code instead of an unrelated editable install
- `configs_generated/` is runtime-only and should stay untracked

## Recent text-weight sweep

- Short sweep summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/text_weight_sweep/20260420T033259Z/summary.json`
- Short winner: `text_sequence_weight = 1.0`
  - `image_to_text_label_accuracy_constrained = 0.109375`
  - `text_to_image_accuracy = 0.13671875`
- Long promotion summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/text_weight_long/20260420T035303Z/summary.json`
- Long promotion metrics for `text_sequence_weight = 1.0`:
  - `image_to_text_exact_match = 0.06640625`
  - `image_to_text_label_accuracy_constrained = 0.1171875`
  - `text_to_image_accuracy = 0.84765625`
  - `unconditional_consistency = 0.671875`
- Decision: do not promote `text_sequence_weight = 1.0`; keep the current `0.75` baseline.

## Recent sampling sweeps

- Broad sampling summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampling_sweep/20260420T044724Z/summary.json`
- Broad best for image-to-text: `sampling.steps = 32`, `sampling.temperature = 0.7`, `sampling.image_to_text_text_time_power = 4.0`
  - `image_to_text_exact_match = 0.1484375`
  - `image_to_text_label_accuracy_constrained = 0.16796875`
  - `text_to_image_accuracy = 0.8984375`
  - `unconditional_consistency = 0.796875`
- T2I-guarded sampling summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampling_t2iguard_sweep/20260420T065716Z/summary.json`
- T2I-guarded winner: `sampling.steps = 32`, `sampling.temperature = 1.0`, `sampling.image_to_text_text_time_power = 1.0`
  - `image_to_text_exact_match = 0.140625`
  - `image_to_text_label_accuracy_constrained = 0.1640625`
  - `text_to_image_accuracy = 0.96484375`
  - `unconditional_consistency = 0.796875`
- Sampler A/B summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampler_ab/20260420T083644Z/summary.json`
- Best A/B result: `sampling.steps = 32`, `sampling.temperature = 0.7`, `sampling.image_to_text_text_time_power = 4.0`, `integrator = legacy_progress_euler`, `final_decode = final_model_call`
  - `image_to_text_exact_match = 0.1484375`
  - `image_to_text_label_accuracy_constrained = 0.16796875`
  - `text_to_image_accuracy = 0.93359375`
  - `unconditional_consistency = 0.796875`
- Scheduled-Euler diagnostic result: the paper-style `scheduled_euler + last_endpoint` sampler underperforms on the current checkpoint, e.g. `confirm_best_fixed` gets `image_to_text_exact_match = 0.12109375`, `text_to_image_accuracy = 0.87109375`, and `unconditional_consistency = 0.640625`.
- Denoiser-oracle diagnostics: direct image-conditioned `D_t` is strong at high text time. At `progress = 0.95`, exact is `0.94921875` and constrained label accuracy is `0.96875` in `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampler_ab_20260420T083644Z_confirm_best_fixed/20260420T085614Z-diagnose-i2t/i2t_diagnostics.json`.
- Decision: keep the legacy sampler as the active default for the current checkpoint. The low free-running i2t score is primarily a sampling/ODE issue, not evidence that the denoiser lacks image understanding.

## Next experiment

Do not continue increasing `text_sequence_weight` without a new reason. The next low-cost check should target sampler/time-conditioning alignment: evaluate a final-step endpoint projection or train with a time parameterization matching the scheduled Euler sampler before changing task mix.

## Archived local trees

- `visualize-joint-work`
- `schedule-fix-analysis`
- `uniindex-formal-run`

Treat them as read-only references unless they are explicitly reactivated.
