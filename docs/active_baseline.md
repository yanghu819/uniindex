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
  - `sampling.final_model_progress = 0.95`
  - `sampling.image_to_text_decoder = sample`
  - `sampling.image_to_text_projection = candidate_renoise`
  - `sampling.image_to_text_projection_progress = 0.5`
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
- Active decoder sweep path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/decoder_ab/20260420T105052Z/sample_fmp095/20260420T105823Z-eval/metrics.json`
- Active decoder sweep metrics:
  - `image_to_text_exact_match = 0.15234375`
  - `image_to_text_token_accuracy = 0.36779794790844517`
  - `image_to_text_label_accuracy_constrained = 0.17578125`
  - `text_to_image_accuracy = 0.96484375`
  - `unconditional_consistency = 0.8125`
- Active projection eval path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/fullvocab_long_tsw075_i2tr06_candidate_proj_p050/20260423T041348Z-eval/metrics.json`
- Active projection metrics:
  - `image_to_text_exact_match = 0.16796875`
  - `image_to_text_token_accuracy = 0.3820047355958958`
  - `image_to_text_label_accuracy_constrained = 0.19140625`
  - `text_to_image_accuracy = 0.97265625`
  - `unconditional_consistency = 0.8125`

## Canonical execution

- Use `./run.sh prepare|stage1|stage2|eval --config ...`
- Use `./run.sh diagnose-i2t-image-dependence --config ... --progress ...` to compare true-image, shuffled-image, and random-image-token controls before promoting another i2t training or sampler change.
- Use `./run.sh diagnose-i2t-sampler-trajectory --config ... --progress ...` to test whether the free i2t sampling state still responds to true-image controls at intermediate sampler steps.
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
- Decoder A/B summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/decoder_ab/20260420T105052Z/summary.json`
- Best sample-decoder result: `final_model_progress = 0.95`
  - `image_to_text_exact_match = 0.15234375`
  - `image_to_text_label_accuracy_constrained = 0.17578125`
  - `text_to_image_accuracy = 0.96484375`
  - `unconditional_consistency = 0.8125`
- Candidate denoiser-score result: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/decoder_ab/20260420T110918Z_candidate_chunked/summary.json`
  - `image_to_text_exact_match = 0.16796875`
  - `image_to_text_label_accuracy_constrained = 0.16796875`
  - `image_to_text_token_accuracy = 0.3291239147592739`
  - `text_to_image_accuracy = 0.921875`
  - `unconditional_consistency = 0.84375`
- Decision: promote `final_model_progress = 0.95` in the active config. Keep `image_to_text_decoder = sample` as default; `candidate_denoiser_score` is useful as a label-rerank diagnostic but hurts free text token accuracy.

## Active projection sampler

- Run timestamp: `2026-04-23T04:13:48Z` (`2026-04-23 12:13:48 CST`)
- Remote path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/fullvocab_long_tsw075_i2tr06_candidate_proj_p050/20260423T041348Z-eval/metrics.json`
- Code state: remote detached checkout `f0d20a7`.
- Config: `configs/flm_joint_work_fullvocab_tsw075_candidate_proj_p050.yaml`
- Sampling change: at i2t sampler progress `0.5`, select the best canonical label candidate from current text logits, re-noise that projected text state at the next schedule time, then continue the legacy-progress trajectory.
- Metrics:
  - `image_to_text_exact_match = 0.16796875`
  - `image_to_text_token_accuracy = 0.3820047355958958`
  - `image_to_text_label_accuracy_constrained = 0.19140625`
  - `text_to_image_accuracy = 0.97265625`
  - `unconditional_consistency = 0.8125`
- Decision: promote `candidate_renoise` at `progress = 0.5` into `configs/flm_joint_work_fullvocab_tsw075.yaml`. It beats the prior active decoder result on i2t exact, token accuracy, constrained label accuracy, and text-to-image accuracy while preserving unconditional consistency.
- Environment note: this eval again emitted Emu3.5 VisionTokenizer remote-code download messages despite offline env vars. Pinning or vendoring that tokenizer code remains necessary before a long unattended run.

## Recent projection sweep

- Run timestamp: `2026-04-23T04:19:39Z` (`2026-04-23 12:19:39 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/projection_sweep/20260423T041939Z-projection-sweep/summary.json`
- Base config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Results:
  - `candidate_renoise @ 0.4`: exact `0.1640625`, label `0.1796875`, token `0.3851617995264404`, t2i `0.97265625`, uncond `0.8125`
  - `candidate_renoise @ 0.5`: exact `0.16796875`, label `0.19140625`, token `0.3820047355958958`, t2i `0.97265625`, uncond `0.8125`
  - `candidate_renoise @ 0.6`: exact `0.1640625`, label `0.1796875`, token `0.388318863456985`, t2i `0.97265625`, uncond `0.8125`
  - `argmax_renoise @ 0.5`: exact `0.16015625`, label `0.1875`, token `0.3796369376479874`, t2i `0.97265625`, uncond `0.8125`
- Decision: keep `candidate_renoise @ 0.5` as the active default. The `0.4` and `0.6` candidate projections improve token accuracy in one direction or another but lose exact/label accuracy. `argmax_renoise @ 0.5` is close on label accuracy but lower on exact/token, so the gain is mostly from canonical label projection rather than re-noising alone.
- Environment note: all sweep logs still show Emu3.5 VisionTokenizer remote-code download messages despite offline env vars.

## Recent gamma sweep

- Run timestamp: `2026-04-23T06:50:19Z` (`2026-04-23 14:50:19 CST`)
- Local record timestamp: `2026-04-23T07:14:00Z` (`2026-04-23 15:14:00 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/gamma_sweep/20260423T065018Z-gamma-sweep/summary.json`
- Code state: remote detached checkout `6d5d483`.
- Base config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Fixed sampler settings: `sampling.steps = 32`, `sampling.temperature = 0.7`, `sampling.integrator = legacy_progress_euler`, `sampling.final_model_progress = 0.95`, `sampling.image_to_text_projection = candidate_renoise`, `sampling.image_to_text_projection_progress = 0.5`.
- Interpretation note: `t_pos` is the clean weight used by `mix_flm_noise = (1 - t) * noise + t * x1`. Because i2t text time uses `progress^power`, higher `sampling.image_to_text_text_time_power` means lower text gamma and a noisier text state.
- Results:
  - `power = 3.0`: exact `0.16796875`, label `0.1875`, token `0.38910812943962114`, t2i `0.97265625`, uncond `0.8125`
  - `power = 4.0`: exact `0.16796875`, label `0.19140625`, token `0.3820047355958958`, t2i `0.97265625`, uncond `0.8125`
  - `power = 5.0`: exact `0.15234375`, label `0.1796875`, token `0.3804262036306235`, t2i `0.97265625`, uncond `0.8125`
  - `power = 6.0`: exact `0.140625`, label `0.1796875`, token `0.36621941594317287`, t2i `0.97265625`, uncond `0.8125`
  - `power = 8.0`: exact `0.109375`, label `0.1640625`, token `0.34964483030781374`, t2i `0.97265625`, uncond `0.8125`
- Decision: do not promote a new gamma. Keep `sampling.image_to_text_text_time_power = 4.0`. Lower-gamma settings (`5.0`, `6.0`, `8.0`) monotonically hurt exact/token accuracy and do not improve constrained label accuracy. Higher-gamma `3.0` only improves token accuracy while lowering constrained label accuracy, so the active bottleneck is not simply that text gamma is too high.
- Lesson: after midpoint `candidate_renoise`, the next useful sampler change should change how the projection is selected or repeated, not make the subsequent text trajectory noisier.
- Environment note: all sweep logs still show Emu3.5 VisionTokenizer remote-code download messages despite offline env vars.

## Active image-dependence diagnostic

- Run timestamp: `2026-04-22T05:10:29Z` (`2026-04-22 13:10:29 CST`)
- Local record timestamp: `2026-04-22T05:14:54Z` (`2026-04-22 13:14:54 CST`)
- Remote path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/fullvocab_long_tsw075_i2tr06/20260422T051028Z-diagnose-i2t-image-dependence/i2t_image_dependence.json`
- Code state: remote detached checkout `16edd16`.
- Config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Direct denoiser control metrics:
  - `progress = 0.50`: true label `0.41796875`, shuffled label `0.1640625`, random label `0.15234375`; true-minus-shuffled label margin `0.25390625`
  - `progress = 0.75`: true exact `0.5546875`, shuffled exact `0.40234375`, random exact `0.44140625`; true-minus-shuffled exact margin `0.15234375`
  - `progress = 0.90`: true exact `0.9296875`, shuffled exact `0.859375`, random exact `0.828125`; true-minus-shuffled label margin `0.01953125`
  - `progress = 0.95`: true exact `0.98046875`, shuffled exact `0.90234375`, random exact `0.94921875`; true-minus-random label margin `0.0078125`
- Interpretation: the active denoiser has real image signal at low and mid text times, but high-progress predictions are dominated by the text prior enough that shuffled/random controls nearly catch up. The free sampler's weak i2t result is therefore more likely a trajectory/time-conditioning alignment problem than a total lack of image-conditioned denoising.
- Decision: keep the active baseline unchanged. The next sampler experiment should force the free trajectory to query the denoiser where the true-vs-control margin is largest, instead of only changing the final projection near `progress = 0.95`.

## Active sampler-trajectory diagnostic

- Run timestamp: `2026-04-22T05:15:13Z` (`2026-04-22 13:15:13 CST`)
- Remote path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/fullvocab_long_tsw075_i2tr06/20260422T051513Z-diagnose-i2t-sampler-trajectory/i2t_sampler_trajectory.json`
- Code state: remote detached checkout `95ee738`.
- Config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Free sampler trajectory metrics:
  - `progress = 0.50`: true label `0.33984375`, true exact `0.0703125`; true-minus-shuffled label margin `0.234375`
  - `progress = 0.75`: true label `0.17578125`, true exact `0.0703125`; true-minus-shuffled label margin `0.05859375`
  - `progress = 0.90`: true label `0.15625`, true exact `0.10546875`; true-minus-shuffled label margin `0.015625`
  - sampler-step `progress = 0.95`: true label `0.14453125`, true exact `0.1171875`; true-minus-shuffled label margin `0.0078125`
  - final model call at `progress = 0.95`: true exact `0.12109375`, shuffled exact `0.1171875`, random exact `0.125`; true/shuffled/random label all stay near `0.14453125` to `0.1484375`
- Interpretation: direct denoiser diagnostics show high image-conditioned accuracy at clean-noised text states, but the free sampler's text state has already drifted by `progress = 0.75`. The remaining final decoder call is effectively image-insensitive. This rules out another final-model-progress-only sweep as a likely fix.
- Decision: next sampler change should address state distribution drift, for example a midpoint reprojection/re-noising check or candidate-label projection around `progress = 0.5`, then resume the trajectory with the image condition.

## Recent low-t image-dependence check

- Run timestamp: `2026-04-22T04:22:35Z` (`2026-04-22 12:22:35 CST`)
- Local resume timestamp: `2026-04-22T04:52:52Z` (`2026-04-22 12:52:52 CST`)
- Remote run root: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/lowt_ft/20260422T042235Z-recovery`
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/lowt_ft/20260422T042235Z-recovery/summary.json`
- Code state: remote detached checkout `86bdfca`; local resume branch `codex/resume-schedule-fix-layout-integ` is the equivalent local tree at `7e15d46`.
- Config: `configs/flm_joint_work_fullvocab_short_i2tp40_tsw075_lowt.yaml`
- Eval metrics:
  - `image_to_text_exact_match = 0.0`
  - `image_to_text_token_accuracy = 0.2809786898184688`
  - `image_to_text_label_accuracy_constrained = 0.08984375`
  - `text_to_image_accuracy = 0.07421875`
  - `unconditional_consistency = 0.0`
- Direct i2t denoiser at `progress = 0.95`:
  - true image: `exact = 0.0`, `label = 0.33984375`, `token = 0.4222573007103394`
  - shuffled image: `exact = 0.0`, `label = 0.328125`, `token = 0.4151539068666141`
  - random image tokens: `exact = 0.0`, `label = 0.3203125`, `token = 0.4151539068666141`
- Decision: do not promote the low-t/noise-only i2t strategy. It collapses free sampling, damages text-to-image and unconditional metrics, and does not create a meaningful true-image advantage over shuffled or random image controls.
- Environment note: the run reached completion, but the eval log showed Hugging Face remote-code activity for the Emu3.5 tokenizer path despite offline env vars. Before any longer run, pin or vendor/cache the tokenizer code so `HF_HUB_OFFLINE=1` is actually sufficient.
- Lesson: low text time alone makes the text channel less useful, but it does not force the model to bind labels to the image condition. The next i2t change should use an explicit image-dependence check or sampler/time-conditioning alignment target, not just more task-mix pressure.

## Next experiment

Do not continue increasing `text_sequence_weight`, the low-t/noise-only i2t strategy, or lower-gamma sampling without a new reason. The next low-cost check should either pin/vendor the Emu3.5 tokenizer remote code for reliable offline runs, or run a narrower sampler-only check around canonical projection quality, such as candidate projection with a second projection point or candidate-score reranking after the midpoint reset.

## Archived local trees

- `visualize-joint-work`
- `schedule-fix-analysis`
- `uniindex-formal-run`

Treat them as read-only references unless they are explicitly reactivated.
