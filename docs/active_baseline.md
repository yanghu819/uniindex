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
- Use `./run.sh probe-i2t-llm-decoder --config configs/flm_joint_work_fullvocab_tsw075_i2t_llm_decoder.yaml --steps 200 --eval-every 50` to test the pretrained small-LLM i2t decoder path.
- Use `./down.sh --config configs/flm_joint_work_fullvocab_tsw075_i2t_llm_decoder.yaml --i2t-llm` to download `distilgpt2` into the repo-local Hugging Face cache before remote runs.
- Use `./run.sh probe-label-features --config configs/flm_joint_work_fullvocab_tsw075_label_feature_probe.yaml --steps 200 --eval-every 50` to test whether VQ image tokens and frozen FLM image hidden features contain linearly usable label information.
- Use `./run.sh sweep-i2t-power` for the 2/4/6 short sweep over `image_to_text_text_time_power`
- Use `./run.sh sweep-i2t-repeats` for the 4/6/8 short sweep over `stage2_image_to_text_repeats`
- `run.sh` exports `PYTHONPATH=$ROOT/src`, so every worktree resolves the local code instead of an unrelated editable install
- `configs_generated/` is runtime-only and should stay untracked

## Current i2t LLM decoder probe

- Local implementation timestamp: `2026-04-25`.
- Latest code state: `ec9230183e4ef4ba64e6349577d41a83b113f35f`.
- Config: `configs/flm_joint_work_fullvocab_tsw075_i2t_llm_decoder.yaml`.
- Purpose: test a paradigm shift for image-to-text understanding by freezing the active UniIndex checkpoint and `distilgpt2`, then training only an image-to-LLM soft prefix adapter.
- This does not alter the active FLM t2i/unconditional path, the active sampler, or the active checkpoint.
- Metrics to trust first:
  - true-image candidate accuracy over the ten canonical label strings
  - shuffled-image candidate accuracy as the binding control
  - free-generation exact match as a secondary check
- Local validation:
  - `ruff check .` passes
  - `.venv/bin/python -m pytest -q` passes with `83 passed`
  - `distilgpt2` downloads to `.cache/huggingface`
  - local LLM asset smoke returns prefix shape `(2, 8, 768)`, candidate score shape `(2, 10)`, and finite scores
- Lesson: treat free text generation as a downstream display problem, not the primary understanding metric. The first useful signal is whether a frozen LLM plus a trained image prefix can rank the correct label above shuffled-image controls.

### Remote i2t LLM decoder result

- Run timestamp: `2026-04-26T05:23:20Z` (`2026-04-26 13:23:20 CST`).
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/i2t_llm_decoder/20260426T052320Z-probe-i2t-llm-decoder/summary.json`.
- Code state: remote detached checkout `ec9230183e4ef4ba64e6349577d41a83b113f35f`.
- Setup fallback: remote Hugging Face access timed out for `distilgpt2`, so the local model cache was copied into `/fangxueji/Projects/PG/uniindex/.cache/huggingface` and the run used `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- Bug found and fixed before the final run: `torch.inference_mode()` feature extraction produced inference tensors that could not be used for adapter backward. The fix uses `torch.no_grad()` for frozen feature extraction and adds a regression test.
- Smoke result: 2-step remote smoke completed with `metadata.exit_status = ok`.
- 200-step results:
  - step `0`: candidate `0.07421875`, shuffled `0.07421875`, margin `0.0`, free generation exact `0.0`
  - step `50`: candidate `0.26171875`, shuffled `0.1171875`, margin `0.14453125`, free generation exact `0.0`
  - step `100`: candidate `0.28125`, shuffled `0.140625`, margin `0.140625`, free generation exact `0.0`
  - step `150`: candidate `0.31640625`, shuffled `0.1015625`, margin `0.21484375`, free generation exact `0.0`
  - step `200`: candidate `0.28125`, shuffled `0.12109375`, margin `0.16015625`, free generation exact `0.0`
- Decision: do not promote this first LLM decoder probe because the final and peak candidate accuracies are below the pre-set `0.35` signal threshold. However, the true-vs-shuffled margin is real, peaking at `0.21484375`, so the image prefix is not being ignored.
- Next LLM-side iteration should not pursue free generation yet. It should make the candidate-ranking objective stronger: keep frozen `distilgpt2`, train the prefix adapter with explicit true-vs-shuffled candidate contrast, and save/select the best eval checkpoint instead of assuming the final step is best.

## Current label feature probe

- Run timestamp: `2026-04-26T05:41:46Z` (`2026-04-26 13:41:46 CST`).
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/label_feature_probe/20260426T054146Z-probe-label-features/summary.json`.
- Code state: remote detached checkout `6c52f8ba9e71505bc38e3564a27fa1a8fb886fa3`.
- Purpose: directly test the hypothesis that the VQ tokens or frozen FLM image features might be too weak for image-to-text understanding.
- Method:
  - train a small supervised probe from VQ image tokens to the ten MNIST labels
  - train a separate small supervised probe from frozen FLM pooled image hidden features to the ten labels
  - keep the active checkpoint frozen; no FLM training and no checkpoint writes
- Local validation before the run:
  - `.venv/bin/python -m pytest -q` -> `89 passed`
  - `ruff check src/uniindex/label_feature_probe.py src/uniindex/cli.py tests/test_label_feature_probe.py` -> passed
  - `bash -n run.sh` -> passed
- Remote smoke: 2-step `probe-label-features` completed with `metadata.exit_status = ok`.
- 200-step results:
  - step `0`: VQ token probe `0.08984375`, FLM hidden probe `0.08984375`
  - step `50`: VQ token probe `0.12890625`, FLM hidden probe `0.26171875`
  - step `100`: VQ token probe `0.203125`, FLM hidden probe `0.33203125`
  - step `150`: VQ token probe `0.21484375`, FLM hidden probe `0.34375`
  - step `200`: VQ token probe `0.2734375`, FLM hidden probe `0.359375`
- Interpretation: VQ features are not blank, and the frozen FLM hidden features are better than raw VQ-token probing, but both are weak. This makes the current bottleneck look like weak image semantics plus weak image-text binding, not just a bad text sampler.
- Decision: do not keep trying sampler-only fixes as the main line. The next meaningful iteration should either improve the image feature used by the i2t decoder or add an explicit image/text binding objective. A stronger text decoder can help format labels, but it cannot create label evidence if the image feature remains this weak.
- Lesson: keep this probe as a cheap gate. Before spending time on new i2t decoders, first check whether the candidate image feature crosses a useful supervised-probe threshold; below roughly `0.50`, expect free text exact to stay unstable.

### SigLIP-VQ follow-up

- Run timestamp: `2026-04-26T07:53:22Z` (`2026-04-26 15:53:22 CST`).
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_label_feature_probe_128px_n256/20260426T075322Z-probe-label-features/summary.json`.
- Code state: remote detached checkout `5749031b3d4b06e2492adce91e535639fd499a27`, which includes the vq-only probe fixes through `cab8710a05db6e377aaac272c306b742cb1bd786`.
- Method: use LLaDA2.0-Uni SigLIP-VQ encoder tokens only, image size `128`, train/test limits `256/256`, and run the VQ-token label probe for `200` steps with `--vq-only`.
- Asset fallback:
  - remote Hugging Face API timed out before repo metadata lookup
  - local download succeeded for the three required encoder files only: `config.json`, `preprocessor_config.json`, `image_tokenizer.safetensors`
  - remote then downloaded `image_tokenizer.safetensors` from `hf-mirror.com` into the repo-local snapshot path and ran with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  - `torchvision` is intentionally not added as a dependency; the SigLIP-VQ loader supplies a small import-compatible shim for the two preprocessing functions used by the external tokenizer source
- Smoke:
  - `16/16`, image size `256`, 2-step vq-only probe completed after the encoder-only path and vq-only probe bugs were fixed
  - batch encoding path also completed a fresh `16/16` smoke
- `128/128`, image size `128`, 200-step result:
  - step `0`: `0.1171875`
  - step `50`: `0.265625`
  - step `100`: `0.5`
  - step `150`: `0.6015625`
  - step `200`: `0.6640625`
- `256/256`, image size `128`, 200-step result:
  - step `0`: `0.09765625`
  - step `50`: `0.25`
  - step `100`: `0.4609375`
  - step `150`: `0.609375`
  - step `200`: `0.69921875`
- Comparison: the earlier generation-tokenizer VQ-token probe reached only `0.2734375` on a 256-sample test set after `200` steps. SigLIP-VQ reaches `0.69921875` on the same test size even at 128px.
- Decision: SigLIP-VQ materially improves class-level semantics and should be tested as the shared image-token space for the next unified FLM branch, not only as a separate understanding adapter.
- Lesson: the old sampler line should be treated as a historical baseline. The next real algorithmic experiment should use SigLIP-VQ tokens inside the FLM itself and measure both image-to-text understanding and text-to-token generation.

### SigLIP-VQ text decoder probe

- Run timestamp: `2026-04-26T08:13:25Z` (`2026-04-26 16:13:25 CST`).
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_text_decoder_probe/20260426T081325Z-probe-vq-text-decoder/summary.json`.
- Code state: remote detached checkout `1f6b4b31a61121c0304ebb75ee06c6c4bf148749`.
- Method: train a small Transformer decoder directly from LLaDA2.0-Uni SigLIP-VQ tokens to canonical char-level label text. This bypasses the old FLM sampler, `candidate_renoise`, and legacy text-state fallbacks.
- Config: `configs/flm_joint_work_siglipvq_text_decoder_probe.yaml`, image size `128`, train/test limits `256/256`, `d_model = 256`, `n_layers = 2`, `steps = 200`, CE-only.
- Smoke: 2-step run completed successfully at `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_text_decoder_probe/20260426T081229Z-probe-vq-text-decoder/summary.json`.
- Main result:
  - step `0`: free exact `0.0`, candidate `0.12890625`, shuffled candidate `0.12109375`, token `0.05288082083662194`
  - step `50`: free exact `0.19921875`, candidate `0.76953125`, shuffled candidate `0.09765625`, token `0.6906077348066298`
  - step `100`: free exact `0.7265625`, candidate `0.89453125`, shuffled candidate `0.09765625`, token `0.8831886345698501`
  - step `150`: free exact `0.859375`, candidate `0.92578125`, shuffled candidate `0.08984375`, token `0.925808997632202`
  - step `200`: free exact `0.86328125`, candidate `0.921875`, shuffled candidate `0.09375`, token `0.925808997632202`
- Interpretation: this is a paradigm-level result, not a sampler tweak. The same task that stalled around `0.17` exact with the old FLM sampler reaches `0.86` free exact when the i2t path is fed SigLIP-VQ semantic tokens and trained as a direct text decoder.
- Decision: stop treating the current i2t failure as primarily a gamma/sampler problem. The next architecture test should be a unified FLM over SigLIP-VQ image tokens, so generation and understanding share the same semantic token space.
- Lesson: no contrast term was needed for the first proof; shuffled-image candidate accuracy stayed near chance (`0.09375`) while true-image candidate accuracy reached `0.921875`. Use contrast only if a larger-scale SigLIP-VQ decoder starts to leak label priors.

### SigLIP-VQ unified FLM generation probe

- Run timestamp: `2026-04-26T10:29:16Z` to `2026-04-26T10:36:58Z` (`2026-04-26 18:29:16-18:36:58 CST`).
- Code state: remote detached checkout `7fc0eb0585f0a081b708605167c3c44bc538b7bc`.
- Config: `configs/flm_joint_work_siglipvq_generation_probe.yaml`.
- Purpose: test whether SigLIP-VQ can be the shared image-token space for a unified FLM, not only an image-to-text probe.
- Decoder assets:
  - assets are cached under `/fangxueji/Projects/PG/uniindex/.cache`
  - `decoder-turbo/decoder_model.safetensors` is about `12.3GB`
  - the decoder is an evaluation renderer for VQ tokens; it is not part of FLM training
- Implementation fixes before the final run:
  - added a diffusers attention-dispatch compatibility wrapper for the decoder source
  - narrowed reconstruction-probe `inference_mode` so first-time classifier training keeps gradients enabled
  - changed SigLIP-VQ reconstruction to the decoder's native `resolution_multiplier = 2`
  - changed the generation probe config to `image_size = 512`, grid `32x32`, train/test limits `64/64`, batch size `4`, stage steps `50/100`
  - reconstruction grids now show input/reconstruction pairs
- Reconstruction oracle:
  - committed-config summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_probe_img512/20260426T102916Z-probe-siglipvq-reconstruction/summary.json`
  - visual grid: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_probe_img512/20260426T102916Z-probe-siglipvq-reconstruction/reconstruction_grid.png`
  - result: `2/2` classifier accuracy; `seven -> seven`, `two -> two`
  - lesson: 128px / `resolution_multiplier = 1` produced unreadable artifacts, but native 512px / `resolution_multiplier = 2` reconstructs recognizable MNIST digits. Do not judge this token space from the non-native renderer setting.
- Unified FLM training smoke:
  - stage1 metadata: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_probe_img512/20260426T103459Z-stage1/metadata.json`
  - stage2 metadata: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_probe_img512/20260426T103536Z-stage2/metadata.json`
  - checkpoints:
    - `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/models/siglipvq_generation_probe_img512/checkpoints/stage1_latest.pt`
    - `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/models/siglipvq_generation_probe_img512/checkpoints/stage2_latest.pt`
  - result: 512px SigLIP-VQ token space trains through stage1 and stage2 successfully; sequence length `1024` is workable with batch size `4`
- Initial i2t diagnostic after the short unified-FLM train:
  - diagnostics: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_probe_img512/20260426T103658Z-diagnose-i2t/i2t_diagnostics.json`
  - `progress=0.5`: exact `0.0`, token `0.3184713375796178`, constrained label `0.09375`
  - `progress=0.9`: exact `0.0`, token `0.31528662420382164`, constrained label `0.09375`
  - generated text collapsed mostly to `oio`
- Decision:
  - Promote the unified SigLIP-VQ FLM branch as the next real architecture experiment because the token space reconstructs and the FLM can train on it.
  - Do not treat the 100-step i2t result as a failure of the paradigm; it is a minimal connectivity smoke with only `64` training samples.
  - Do not run full image eval yet: the current renderer loads a large decoder per image, so full t2i/unconditional eval would be too slow. First add a batched/cached renderer or run tiny decode-only eval.
- Next iteration:
  - add a cached/batched SigLIP-VQ renderer so t2i/unconditional eval is practical
  - run a longer unified-FLM overfit probe on `16` samples, then `64/64`
  - add a supervised label/text auxiliary loss for image-to-text during stage2 if the longer run still collapses to short bogus strings

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

## Recent candidate-score projection sweep

- Run timestamp: `2026-04-23T11:54:14Z` (`2026-04-23 19:54:14 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/candidate_score_projection/20260423T115414Z-candidate-score-projection/summary.json`
- Code state: remote detached checkout `bc7c0cb`.
- Base config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Guard: `eval.isolate_sampling_rng = true`, continuous per-branch RNG streams, `eval.sampling_seed = 420700`.
- Fixed sampler settings: `sampling.steps = 32`, `sampling.temperature = 0.7`, `sampling.integrator = legacy_progress_euler`, `sampling.final_model_progress = 0.95`, `sampling.image_to_text_projection_progress = 0.5`.
- Change under test: `sampling.image_to_text_projection = candidate_score_renoise`, which scores canonical text candidates with the image-conditioned denoiser and then re-noises the selected canonical candidate at the midpoint.
- Results:
  - active `candidate_renoise @ 0.5`: exact `0.171875`, label `0.1953125`, token `0.39779005524861877`, t2i `0.95703125`, uncond `0.859375`
  - `candidate_score_renoise`, score `[0.5]`: exact `0.15234375`, label `0.16015625`, token `0.3701657458563536`, t2i `0.95703125`, uncond `0.859375`
  - `candidate_score_renoise`, score `[0.5, 0.75]`: exact `0.140625`, label `0.1796875`, token `0.36306235201262826`, t2i `0.95703125`, uncond `0.859375`
  - `candidate_score_renoise`, score `[0.5, 0.75, 0.9]`: exact `0.16015625`, label `0.1875`, token `0.3701657458563536`, t2i `0.95703125`, uncond `0.859375`
- Decision: do not promote candidate-score projection. All candidate-score variants lose exact match, constrained label accuracy, and token accuracy versus the active midpoint candidate projection. The t2i and unconditional metrics stay identical under RNG isolation, so the result is an i2t-only regression rather than cross-branch noise.
- Interpretation: selecting the midpoint canonical label from a separate denoiser-score pass is less reliable than selecting from the sampler's current text logits. The current trajectory state contains useful text-coherence information that the independent candidate score throws away.
- Lesson: do not keep sweeping pure denoiser-score projection progress or noise count. If candidate scoring is used again, use it only as a weak tie-breaker blended with the current sampler logits, not as a replacement for the current projection score.
- Environment note: the eval logs still show Emu3.5 VisionTokenizer remote-code download messages despite offline env vars; pin or vendor/cache this before longer unattended runs.

## Recent candidate-score blend mini sweep

- Run timestamp: `2026-04-23T13:33:28Z` (`2026-04-23 21:33:28 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/candidate_score_blend/20260423T123842Z-candidate-score-blend-mini/summary.json`
- Code state: remote detached checkout `c53f15a`.
- Base config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Guard: `eval.isolate_sampling_rng = true`, continuous per-branch RNG streams, `eval.sampling_seed = 420700`.
- Active reference: prior RNG-isolated `candidate_renoise @ 0.5`, exact `0.171875`, label `0.1953125`, token `0.39779005524861877`, t2i `0.95703125`, uncond `0.859375`.
- Change under test: `sampling.image_to_text_projection = candidate_score_blend_renoise`, using current sampler candidate score plus `blend_weight * denoiser_candidate_score`.
- Results:
  - `blend_weight = 0.10`, score `[0.5]`: exact `0.15625`, label `0.171875`, token `0.38595106550907654`, t2i `0.95703125`, uncond `0.859375`
  - `blend_weight = 0.25`, score `[0.5]`: exact `0.16015625`, label `0.17578125`, token `0.3867403314917127`, t2i `0.95703125`, uncond `0.859375`
- Decision: do not promote candidate-score blend. Both weak blend cases lose exact match, constrained label accuracy, and token accuracy versus active. The small gain from `0.10` to `0.25` is not enough to justify a larger `0.50` probe.
- Interpretation: even as a weak tie-breaker, the denoiser candidate score moves midpoint projection away from the free-text trajectory that the sampler can complete. Candidate-score information is not the missing ingredient for the current sampler failure.
- Lesson: stop sampler-side candidate-score iterations for now. The next meaningful step should change training pressure toward image-text binding, such as a lightweight mismatched-image contrastive or consistency loss, while preserving the active sampler for evaluation.
- Environment note: the eval logs still show Emu3.5 VisionTokenizer remote-code download messages despite offline env vars.

## Recent logit-normal gamma sweep

- Run timestamp: `2026-04-23T10:57:17Z` (`2026-04-23 18:57:17 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/logit_normal_gamma/20260423T105717Z-logit-normal-gamma/summary.json`
- Code state: remote detached checkout `2bf6a43`.
- Base config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Guard: `eval.isolate_sampling_rng = true`, continuous per-branch RNG streams, `eval.sampling_seed = 420700`.
- Fixed sampler settings: `sampling.steps = 32`, `sampling.temperature = 0.7`, `sampling.integrator = legacy_progress_euler`, `sampling.final_model_progress = 0.95`, `sampling.image_to_text_projection = candidate_renoise`, `sampling.image_to_text_projection_progresses = [0.5]`.
- Change under test: `sampling.image_to_text_text_time_schedule = logit_normal`, which directly sets i2t text gamma to `sigmoid(loc + scale * normal_icdf(progress))` instead of routing it through `progress^power`.
- Results:
  - active `power`: exact `0.171875`, label `0.1953125`, token `0.39779005524861877`, t2i `0.95703125`, uncond `0.859375`
  - `loc = -1.5, scale = 1.0`: exact `0.08203125`, label `0.24609375`, token `0.40173638516179955`, t2i `0.95703125`, uncond `0.859375`
  - `loc = -2.0, scale = 1.0`: exact `0.04296875`, label `0.26171875`, token `0.40568271507498027`, t2i `0.95703125`, uncond `0.859375`
  - `loc = -2.5, scale = 1.0`: exact `0.0390625`, label `0.32421875`, token `0.4238358326756117`, t2i `0.95703125`, uncond `0.859375`
  - `loc = -3.0, scale = 1.0`: exact `0.0546875`, label `0.33984375`, token `0.43646408839779005`, t2i `0.95703125`, uncond `0.859375`
  - `loc = -2.5, scale = 1.5`: exact `0.03515625`, label `0.23828125`, token `0.3898973954222573`, t2i `0.95703125`, uncond `0.859375`
- Decision: do not promote logit-normal low-gamma sampling. No case beats active on exact match, and the best exact among logit-normal cases is less than half the active exact.
- Interpretation: directly lowering i2t text gamma makes the model better at constrained label discrimination but worse at free text generation. This explains why label accuracy can rise while exact match collapses: the sampler is moving toward a classifier-like signal rather than a coherent text trajectory.
- Lesson: do not spend more runs on plain lower-gamma schedules unless the objective is explicitly constrained label classification. For the actual free-text i2t target, the next sampler change should preserve text coherence and change candidate selection or reranking, not simply reduce gamma.
- Environment note: all sweep logs still show Emu3.5 VisionTokenizer remote-code download messages despite offline env vars.

## Recent second-projection sweep

- Run timestamp: `2026-04-23T09:17:11Z` (`2026-04-23 17:17:11 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/second_projection_sweep/20260423T091711Z-second-projection-sweep/summary.json`
- Code state: remote detached checkout `823ef89`.
- Base config: `configs/flm_joint_work_fullvocab_tsw075.yaml`
- Fixed sampler settings: `sampling.steps = 32`, `sampling.temperature = 0.7`, `sampling.integrator = legacy_progress_euler`, `sampling.final_model_progress = 0.95`, `sampling.image_to_text_projection = candidate_renoise`, `sampling.image_to_text_text_time_power = 4.0`.
- Results:
  - `[0.5]`: exact `0.16796875`, label `0.19140625`, token `0.3820047355958958`, t2i `0.97265625`, uncond `0.8125`
  - `[0.5, 0.625]`: exact `0.21484375`, label `0.2421875`, token `0.4277821625887924`, t2i `0.9609375`, uncond `0.828125`
  - `[0.5, 0.7]`: exact `0.23828125`, label `0.2578125`, token `0.43725335438042623`, t2i `0.9609375`, uncond `0.828125`
  - `[0.5, 0.75]`: exact `0.23828125`, label `0.24609375`, token `0.4325177584846093`, t2i `0.9609375`, uncond `0.828125`
  - `[0.5, 0.8]`: exact `0.22265625`, label `0.23046875`, token `0.4230465666929755`, t2i `0.9609375`, uncond `0.828125`
- Decision: do not promote. The unisolated sweep made `[0.5, 0.7]` look strong, but the later RNG-isolated guard shows this was not a stable sampler improvement.
- Interpretation: second projection exposed a real measurement problem. In the unisolated eval loop, i2t and t2i sampling share one RNG stream; adding an extra i2t re-noise changes the RNG state seen by later batches and can create apparent i2t gains. Use RNG-isolated evals before promoting sampler changes that alter random draw counts.
- Lesson: midpoint projection remains useful, but repeating the same projection at `0.7` is not currently justified. Do not spend more runs on lower gamma or second-projection promotion unless a new candidate-selection rule changes the failure mode.
- Environment note: all sweep logs still show Emu3.5 VisionTokenizer remote-code download messages despite offline env vars.

## Recent RNG-isolated guard

- Code timestamp: `2026-04-23T10:41:00Z` (`2026-04-23 18:41:00 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/rng_guard/20260423T103656Z-rng-guard-continuous/summary.json`
- Code state: remote detached checkout `84f7598`.
- Guard config: generated from `configs/flm_joint_work_fullvocab_tsw075.yaml` with `eval.isolate_sampling_rng = true` and continuous per-branch RNG streams. The i2t stream is kept continuous, while t2i and unconditional sampling use independent streams so i2t projection changes cannot advance their random state.
- Results:
  - active `[0.5]`: exact `0.171875`, label `0.1953125`, token `0.39779005524861877`, t2i `0.95703125`, uncond `0.859375`
  - candidate `[0.5, 0.7]`: exact `0.15625`, label `0.17578125`, token `0.3820047355958958`, t2i `0.95703125`, uncond `0.859375`
  - candidate-minus-active: exact `-0.015625`, label `-0.01953125`, token `-0.01578531965272295`, t2i `0.0`, uncond `0.0`
- Decision: keep active `[0.5]`. Under the fair guard, `[0.5, 0.7]` is worse on i2t and has no t2i/unconditional difference.
- Lesson: future sampler sweeps should enable `eval.isolate_sampling_rng` when comparing algorithms that consume different numbers of random draws. Otherwise, apparent wins can come from RNG-stream coupling rather than the algorithm itself.

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

## Recent mismatch-binding short probe

- Run timestamp: `2026-04-23T13:59:30Z` (`2026-04-23 21:59:30 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/mismatch_binding/20260423T135930Z-mismatch-w010-short/summary.json`
- Code state: remote detached checkout `1779f77`.
- Base config: generated from `configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml`.
- Change under test: `train.image_to_text_mismatch_weight = 0.10`, `train.image_to_text_mismatch_margin = 1.0`, evaluated with the active `candidate_renoise @ 0.5` sampler and RNG isolation.
- Metrics:
  - `image_to_text_exact_match = 0.0`
  - `image_to_text_token_accuracy = 0.3362273086029992`
  - `image_to_text_label_accuracy_constrained = 0.0703125`
  - `text_to_image_accuracy = 0.13671875`
  - `unconditional_consistency = 0.0`
- Diagnostics: generated text collapsed to `"tie"` for 235/256 samples and `"tiee"` for 21/256 samples.
- Decision: do not promote the first mismatch-binding implementation. Applying the mismatch margin loss during `joint` training is too destructive for the shared image/text manifold.
- Lesson: the binding objective should be scoped to `image_to_text` updates only. The next quick probe keeps the same mismatch loss idea but removes it from `joint` tasks so stage1 remains a clean joint denoising warmup.
- Environment note: the initial eval spent several minutes in Hugging Face remote-code HEAD retries. Re-running eval with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` completed, but Transformers still emitted dynamic-module "downloaded" warnings from the cached Emu3.5 files. Pin or vendor the tokenizer code before long unattended runs.

## Recent image-to-text-only mismatch probe

- Run timestamp: `2026-04-23T14:25:46Z` (`2026-04-23 22:25:46 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/mismatch_binding/20260423T142546Z-mismatch-i2tonly-w010-short/summary.json`
- Code state: remote detached checkout `26a7ae9`.
- Base config: generated from `configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml`.
- Change under test: same `train.image_to_text_mismatch_weight = 0.10`, but mismatch loss is applied only when `task == "image_to_text"`, not during `joint` warmup.
- Eval metrics:
  - `image_to_text_exact_match = 0.0`
  - `image_to_text_token_accuracy = 0.3378058405682715`
  - `image_to_text_label_accuracy_constrained = 0.08984375`
  - `text_to_image_accuracy = 0.13671875`
  - `unconditional_consistency = 0.0`
- Short-run context: the old short i2tp40 baseline also had exact `0.0`, t2i `0.13671875`, and uncond `0.0`, so short free-sampling metrics alone are not a reliable promotion signal.
- Direct image-dependence diagnostic:
  - `progress = 0.50`: true label `0.140625`, shuffled label `0.14453125`, random label `0.125`
  - `progress = 0.75`: true label `0.23828125`, shuffled label `0.2421875`, random label `0.234375`
  - `progress = 0.90`: true label `0.30078125`, shuffled label `0.31640625`, random label `0.3203125`
  - `progress = 0.95`: true label `0.3359375`, shuffled label `0.3359375`, random label `0.328125`
- Decision: do not promote image-to-text-only mismatch binding at weight `0.10`. It avoids corrupting stage1 but still does not create a true-image advantage over shuffled/random controls.
- Lesson: pairwise mismatched-image margin loss is not yet the right binding objective. It can alter token-position statistics, but it does not reliably make the denoiser prefer the true image. The next training-side attempt should use a more direct image-conditioned label objective or a fine-tune from the active long checkpoint, not another short from-scratch mismatch-weight sweep.
- Environment note: running with offline env avoids repeated HF HEAD retries, but Transformers still emits dynamic-module "downloaded" warnings while loading cached Emu3.5 remote-code files.

## Recent image-conditioned label fine-tune probe

- Run timestamp: `2026-04-23T15:05:04Z` (`2026-04-23 23:05:04 CST`)
- Retry completion timestamp: `2026-04-23T15:16:57Z` (`2026-04-23 23:16:57 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/label_ft/20260423T150710Z-label-ft-w005/summary.json`
- Code state: remote detached checkout `1f31ba3`.
- Base checkpoint: `models/fullvocab_long_tsw075_i2tr06/checkpoints/stage2_latest.pt`.
- Generated config: `configs_generated/flm_joint_work_fullvocab_tsw075_label_ft_w005.yaml`.
- Change under test: stage2-only fine-tune for 300 steps from the active long checkpoint with `train.image_to_text_label_weight = 0.05`, `train.image_to_text_label_text_time = 0.0`, `lr = 0.0001`, active `candidate_renoise @ 0.5`, and RNG-isolated eval seed `420700`.
- Eval metrics:
  - `image_to_text_exact_match = 0.25390625`
  - `image_to_text_token_accuracy = 0.4262036306235201`
  - `image_to_text_label_accuracy_constrained = 0.26953125`
  - `text_to_image_accuracy = 0.9296875`
  - `unconditional_consistency = 0.796875`
- Active reference for the same RNG guard: exact `0.171875`, label `0.1953125`, token `0.39779005524861877`, t2i `0.95703125`, uncond `0.859375`.
- Direct image-dependence diagnostic:
  - `progress = 0.50`: true label `0.4609375`, shuffled label `0.14453125`, random label `0.13671875`; true-minus-shuffled label margin `0.31640625`
  - `progress = 0.75`: true exact `0.57421875`, shuffled exact `0.37109375`, random exact `0.3828125`; true-minus-shuffled exact margin `0.203125`
  - `progress = 0.90`: true exact `0.94140625`, shuffled exact `0.90625`, random exact `0.91015625`; high-progress margins remain small
  - `progress = 0.95`: true exact `0.98828125`, shuffled exact `0.97265625`, random exact `0.9765625`; high-progress text prior still dominates
- Decision: do not replace the active full baseline with this checkpoint because t2i and unconditional fall below the active reference. Do promote the algorithmic direction: direct image-conditioned label supervision from the active checkpoint is the first training-side probe that materially improves free i2t exact, token accuracy, constrained label accuracy, and low/mid-progress true-image margins.
- Lesson: the binding bottleneck is not solved by lower sampling gamma or pairwise mismatch margins. A small supervised label auxiliary can pull the i2t trajectory toward image evidence, but `weight = 0.05` for 300 steps is too much for preserving the shared generative manifold. The next quick run should sweep smaller label pressure from the same active checkpoint, for example `weight = 0.01` and `0.02`, before changing the objective again.
- Environment note: the first eval failed because the new `models_dir` did not contain `eval/mnist_classifier.pt`; copying the active classifier fixed the retry. Generated fine-tune configs should either reuse the active classifier path or copy it into the new `models_dir/eval`. The run still emitted Emu3.5 remote-code "downloaded" warnings despite offline env vars.

## Recent smaller label-weight fine-tune sweep

- Run timestamp: `2026-04-24T01:29:38Z` (`2026-04-24 09:29:38 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/label_ft_sweep/20260424T012938Z-label-ft-mini-sweep/summary.json`
- Code state: remote detached checkout `2a9b3c6`.
- Base checkpoint: `models/fullvocab_long_tsw075_i2tr06/checkpoints/stage2_latest.pt`.
- Change under test: stage2-only fine-tune for 300 steps from the active checkpoint with `train.image_to_text_label_text_time = 0.0`, active `candidate_renoise @ 0.5`, and RNG-isolated eval seed `420700`.
- Results:
  - `label_weight = 0.01`: exact `0.25`, label `0.26953125`, token `0.4262036306235201`, t2i `0.93359375`, uncond `0.8125`
  - `label_weight = 0.02`: exact `0.25`, label `0.26953125`, token `0.425414364640884`, t2i `0.93359375`, uncond `0.8125`
- Decision: do not replace the active baseline. Both smaller weights preserve the strong i2t gain but still hurt text-to-image and unconditional metrics versus the active reference.
- Interpretation: the issue is no longer that `label_weight` is simply too large. At 300 stage2 steps, even `0.01` is enough to push the shared model away from the active generative balance.
- Artifact hygiene note: the original wrapper left `metadata.json` with a stale nonzero exit status after both cases completed; the case artifacts and `summary.json` are valid.
- Lesson: direct label binding is still the most useful i2t training signal, but the next fine-tune should reduce total pressure, for example shorter step counts, rather than keep shrinking the label weight.

## Recent checkpoint interpolation sweep

- Run timestamp: `2026-04-24T03:23:19Z` (`2026-04-24 11:23:19 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/checkpoint_interp/20260424T032319Z-checkpoint-interp/summary.json`
- Code state: remote detached checkout `2a9b3c6`.
- Active checkpoint: `models/fullvocab_long_tsw075_i2tr06/checkpoints/stage2_latest.pt`.
- Source checkpoint: `models/fullvocab_long_tsw075_i2tr06_label_ft_w001/checkpoints/stage2_latest.pt`.
- Change under test: eval-only model weight interpolation, `model = (1 - alpha) * active + alpha * label_ft`, with active `candidate_renoise @ 0.5` and RNG-isolated eval seed `420700`.
- Results:
  - `alpha = 0.125`: exact `0.17578125`, label `0.19921875`, token `0.39857932123125495`, t2i `0.9609375`, uncond `0.765625`
  - `alpha = 0.25`: exact `0.1953125`, label `0.21875`, token `0.40568271507498027`, t2i `0.97265625`, uncond `0.875`
  - `alpha = 0.375`: exact `0.203125`, label `0.24609375`, token `0.4159431728492502`, t2i `0.9609375`, uncond `0.859375`
  - `alpha = 0.50`: exact `0.2265625`, label `0.25390625`, token `0.4262036306235201`, t2i `0.94921875`, uncond `0.765625`
- Active reference for the same RNG guard: exact `0.171875`, label `0.1953125`, token `0.39779005524861877`, t2i `0.95703125`, uncond `0.859375`.
- Decision: do not promote an interpolated checkpoint yet. `alpha = 0.375` is the best balanced point and improves i2t while preserving t2i/uncond, but it misses the promotion exact threshold `0.21875`. `alpha = 0.50` clears the i2t threshold but fails the t2i/uncond guards.
- Best-case image-dependence diagnostic for `alpha = 0.375`:
  - `progress = 0.50`: true-minus-shuffled label margin `0.33203125`, exact margin `0.0546875`
  - `progress = 0.75`: true-minus-shuffled label margin `0.12890625`, exact margin `0.1640625`
  - `progress = 0.90`: true-minus-shuffled label margin `-0.015625`, exact margin `0.01953125`
  - `progress = 0.95`: true-minus-shuffled label margin `-0.00390625`, exact margin `0.015625`
- Interpretation: weight interpolation is useful. It shows the label-ft binding direction is partially compatible with the active model: `alpha = 0.25` and `0.375` improve i2t and keep the guard metrics. The remaining bottleneck is finding a better route to the `0.375-0.50` region without the unconditional collapse seen at `0.50`.
- Environment note: the first interpolation runner generated configs one directory too deep under `configs_generated/checkpoint_interp/...`, which made `load_config()` infer the wrong repo root. The fixed configs live directly under `configs_generated/`. Eval logs still show Emu3.5 remote-code warnings despite offline env vars.

## Recent i2t understanding visual diagnostics

- Run timestamp: `2026-04-24T04:17:46Z` (`2026-04-24 12:17:46 CST`)
- Remote comparison summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/i2t_understanding_compare/20260424T041933Z-i2t-understanding-compare/summary.json`
- Code state: remote detached checkout `bcbaa35`.
- New command: `./run.sh diagnose-i2t-understanding --config ... --sample-count 32 --progress 0.5 --progress 0.75 --progress 0.9 --progress 0.95`
- Outputs per config:
  - `i2t_understanding/sample_cards.png`: original image, target text, free text, constrained top label, direct-vs-sampler anchor labels
  - `i2t_understanding/confusion_matrix.png`: true label versus final free i2t constrained label
  - `i2t_understanding/train_vs_sampler_margins.png`: true-minus-shuffled margins for train-style and sampler-style states
  - `i2t_understanding/samples.json`: per-sample top-3 label candidates, margins, shuffled/random controls, and failure type
- Active 32-sample final free i2t:
  - true-image exact `0.09375`, label `0.125`, token `0.3525641025641026`, mean true-label margin `-17.757659912109375`
  - shuffled-image label `0.0625`, random-token label `0.15625`
  - failure counts: `label_wrong = 22`, `image_insensitive = 6`, `exact_correct = 3`, `label_correct_text_wrong = 1`
- `alpha = 0.375` 32-sample final free i2t:
  - true-image exact `0.15625`, label `0.1875`, token `0.41025641025641024`, mean true-label margin `-16.626012802124023`
  - shuffled-image label `0.0625`, random-token label `0.125`
  - failure counts: `label_wrong = 21`, `image_insensitive = 5`, `exact_correct = 5`, `label_correct_text_wrong = 1`
- Train-vs-sampler margins:
  - active at progress `0.5`: direct label margin `0.46875`, sampler label margin `0.34375`; final exact remains weak
  - `alpha = 0.375` at progress `0.5`: direct label margin `0.46875`, sampler label margin `0.34375`
  - `alpha = 0.375` improves sampler exact margin at progress `0.5/0.75/0.9/0.95` from active `0.0/0.03125/0.0625/0.0625` to `0.09375/0.09375/0.125/0.09375`
- Interpretation: this supports a training-inference trajectory mismatch, not just a missing classifier signal. Direct train-style image evidence is already visible at low/mid progress, but the final free sampler still has negative true-label margins and many label-wrong samples. The interpolation checkpoint appears to make the sampler trajectory retain image evidence better, even though it is not promotable on full metrics.
- Lesson: future i2t experiments should report sample cards, confusion matrices, and train-vs-sampler margins alongside aggregate metrics. The next training change should make stage2 see sampler-like text states or reduce fine-tune pressure, rather than only increasing direct label classification strength.

## Recent i2t fixed-batch overfit probe

- Run timestamp: `2026-04-24T06:56:37Z` (`2026-04-24 14:56:37 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/fullvocab_long_tsw075_i2tr06_candidate_proj_p050/20260424T065637Z-probe-i2t-overfit/overfit_summary.json`
- Launcher metadata: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/i2t_overfit_launcher/20260424T065612Z-base/metadata.json`
- Code state: remote detached checkout `946ed6c`.
- New command: `./run.sh probe-i2t-overfit --config configs/flm_joint_work_fullvocab_tsw075_candidate_proj_p050.yaml --steps 120 --sample-count 16 --test-sample-count 16 --eval-every 30 --lr 0.0001`
- Change under test: no new loss and no long-term checkpoint. The active stage2 checkpoint is fine-tuned in memory on a fixed 16-example i2t mini-batch, then compared against a fixed 16-example held-out test slice with true-image and shuffled-image controls.
- Fixed train sampler metrics:
  - initial true-image: exact `0.3125`, label `0.3125`, token `0.5`, mean true-label margin `-10.483437538146973`
  - final true-image: exact `0.875`, label `0.875`, token `0.9102564102564102`, mean true-label margin `15.650797843933105`
  - final shuffled-image: exact `0.0`, label `0.0625`, token `0.21794871794871795`, mean true-label margin `-30.93375015258789`
- Fixed held-out sampler metrics:
  - initial true-image: exact `0.125`, label `0.125`, token `0.34210526315789475`, mean true-label margin `-18.82723617553711`
  - final true-image: exact `0.3125`, label `0.3125`, token `0.42105263157894735`, mean true-label margin `-15.3848295211792`
  - final shuffled-image: exact `0.25`, label `0.3125`, token `0.39473684210526316`, mean true-label margin `-20.782306671142578`
- Direct denoiser behavior after overfit:
  - fixed train at progress `0.50`: true-image label `1.0`, shuffled-image label `0.0`; this confirms the model can learn true image-to-text binding on the fixed batch.
  - fixed train at progress `0.90/0.95`: shuffled-image label rises to `1.0`; high-gamma direct states are dominated by the clean text target and are not a reliable image-dependence measure.
  - held-out at progress `0.95`: true-image exact `1.0`, shuffled-image exact `0.875`; again, high-gamma direct denoising is mostly text self-prompting.
- Interpretation: the current denoiser has enough capacity to memorize image-to-text binding. The main bottleneck is not "can the transformer represent the mapping?" It is generalizing the binding and keeping it on the free sampler trajectory. The fixed-batch gain does not transfer cleanly to held-out sampler metrics, so simply adding a tiny LLM or more text modeling capacity is not the next highest-leverage move.
- Lesson: use `probe-i2t-overfit` before changing architecture. If a future objective cannot quickly overfit the fixed train batch while keeping shuffled-image sampler low, reject it early. If it overfits fixed train but not held-out, the next change should target sampler-state binding or regularized image-text contrast, not a larger text prior.

## Recent sampler-state binding FT

- Run timestamp: `2026-04-24T07:45:08Z` (`2026-04-24 15:45:08 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampler_state_binding/20260424T074508Z-sampler-state-binding/summary.json`
- Code state: remote detached checkout `179c25f`.
- New command: `./run.sh probe-i2t-sampler-state-ft --config ... --steps ... --progress 0.5 --progress 0.75 --progress 0.9`
- Change under test: freeze an active-checkpoint teacher, use the active `candidate_renoise @ 0.5` sampler to generate sampler-like i2t text states, then fine-tune a student on text CE plus sequence loss. The contrast case adds shuffled-image margin loss with `weight = 0.02`.
- Implementation note: the first remote smoke exposed a backward failure because teacher traces were created under inference mode. The fix clones `z_t` and `t_pos` before the student forward, and the regression test now covers backward through inference-mode sampler traces.
- Validation:
  - `.venv/bin/python -m pytest -q` -> `73 passed`
  - `.venv/bin/ruff check src/uniindex/i2t_sampler_state_ft.py tests/test_i2t_sampler_state_ft.py` -> passed
- `ssb_ce` eval metrics:
  - `image_to_text_exact_match = 0.10546875`
  - `image_to_text_token_accuracy = 0.4956590370955012`
  - `image_to_text_label_accuracy_constrained = 0.44921875`
  - `text_to_image_accuracy = 0.62890625`
  - `unconditional_consistency = 0.0`
- `ssb_ce_contrast_w002` eval metrics:
  - `image_to_text_exact_match = 0.10546875`
  - `image_to_text_token_accuracy = 0.494869771112865`
  - `image_to_text_label_accuracy_constrained = 0.44921875`
  - `text_to_image_accuracy = 0.65625`
  - `unconditional_consistency = 0.0`
- `ssb_ce` 32-sample understanding diagnostic:
  - true-image final: exact `0.125`, label `0.5625`, token `0.5897435897435898`, mean true-label margin `0.2681337893009186`
  - shuffled-image final: exact `0.0`, label `0.03125`, token `0.2692307692307692`, mean true-label margin `-4.192084312438965`
  - random-image-token final: exact `0.0`, label `0.0625`, token `0.3141025641025641`, mean true-label margin `-3.532057762145996`
  - failure counts: `exact_correct = 4`, `label_correct_text_wrong = 14`, `label_wrong = 12`, `image_insensitive = 2`
- Decision: do not promote sampler-state FT. It proves the model can be pushed to use image evidence on sampler-like states, because true-image label accuracy separates sharply from shuffled/random controls. But full-model FT is too destructive: both CE-only and contrast variants collapse t2i and unconditional consistency.
- Lesson: the next training-side idea should preserve the shared generative model while adding binding. Use a smaller write surface such as freezing most layers, adapter-only binding, or active-checkpoint KL/anchor regularization on t2i/unconditional logits. Do not add a tiny LLM for MNIST label text; the failure is not language modeling capacity.
- Run control note: the contrast case was stopped after eval metrics because the decision guards had already failed and the runner was spending non-informative time in Hugging Face remote-code retries before contrast diagnostics.
- Environment note: eval and diagnostics still attempted Hugging Face remote-code HEAD/download for `BAAI/Emu3.5-VisionTokenizer`. Pinning or vendoring that tokenizer code and forcing offline mode remains required before longer unattended experiments.

## Recent fallback-free FLM ablation

- Run timestamp: `2026-04-25T06:16:20Z` (`2026-04-25 14:16:20 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/clean_flm_ablation/20260425T061620Z-clean-flm-ablation/summary.json`
- Code state: remote detached checkout `b951633`.
- New committed configs:
  - `configs/flm_joint_work_fullvocab_tsw075_minflm.yaml`
  - `configs/flm_joint_work_fullvocab_tsw075_no_i2t_projection.yaml`
- `minflm` removes the active sampler-side fallback stack: no midpoint projection, `integrator = scheduled_euler`, and `final_decode = last_endpoint`.
- `no_i2t_projection` keeps the active sampler shell but removes `candidate_renoise`, isolating the projection effect without changing t2i/unconditional random streams.
- Validation:
  - `.venv/bin/python -m pytest tests/test_config.py -q` -> `14 passed`
  - `.venv/bin/python -m pytest -q` -> `75 passed`
- `minflm` metrics:
  - `image_to_text_exact_match = 0.12109375`
  - `image_to_text_token_accuracy = 0.3362273086029992`
  - `image_to_text_label_accuracy_constrained = 0.1484375`
  - `text_to_image_accuracy = 0.875`
  - `unconditional_consistency = 0.640625`
- `no_i2t_projection` metrics:
  - `image_to_text_exact_match = 0.1328125`
  - `image_to_text_token_accuracy = 0.345698500394633`
  - `image_to_text_label_accuracy_constrained = 0.1484375`
  - `text_to_image_accuracy = 0.95703125`
  - `unconditional_consistency = 0.859375`
- Active RNG-isolated reference with `candidate_renoise @ 0.5`: exact `0.171875`, label `0.1953125`, token `0.39779005524861877`, t2i `0.95703125`, uncond `0.859375`.
- 32-sample understanding diagnostic for `no_i2t_projection`:
  - true-image final: exact `0.125`, label `0.1875`, token `0.391025641025641`, mean true-label margin `-17.700584411621094`
  - shuffled-image final: exact `0.09375`, label `0.125`, token `0.28846153846153844`, mean true-label margin `-25.122175216674805`
  - random-image-token final: exact `0.125`, label `0.15625`, token `0.3525641025641026`, mean true-label margin `-18.01748275756836`
  - failure counts: `exact_correct = 4`, `label_correct_text_wrong = 2`, `label_wrong = 20`, `image_insensitive = 6`
- Decision: do not promote fallback-free configs. Use them as clean measurement baselines. Removing `candidate_renoise` exposes the raw i2t trajectory: t2i/uncond stay healthy, but i2t exact/label/token fall well below the active reference. Pure `minflm` is worse and also harms t2i/unconditional, so the paper-style sampler is not the immediate rescue path for this checkpoint.
- Interpretation: the active fallback is not the root cause; it is a small crutch over a weak raw i2t trajectory. The core failure remains image-conditioned text generation on the free sampler path: the final true-image margin is still strongly negative, and random/shuffled controls are too close to true-image outcomes.
- Lesson: future training changes should be evaluated first on `no_i2t_projection` to measure raw understanding. Only after raw i2t improves should `candidate_renoise @ 0.5` be re-enabled as a convenience sampler, not as proof of understanding.
- Environment note: even with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, the logs still print cached Emu3.5 dynamic-module "downloaded" warnings. Pin/vendor remains necessary.

## Recent anchored sampler-state binding raw guard

- Run timestamp: `2026-04-25T09:43:44Z` (`2026-04-25 17:43:44 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampler_state_anchor/20260425T094344Z-ssb-anchor-raw/summary.json`
- Code state: remote detached checkout `70ad6f6`.
- Method: start from the active checkpoint, train only the final transformer block plus norm/head on sampler-state i2t states, and add teacher KL anchors on `joint` and `text_to_image` states. Runtime config used the raw `no_i2t_projection` guard so the result measures real i2t trajectory understanding, not the `candidate_renoise` sampler crutch.
- Local validation before the run:
  - `.venv/bin/python -m pytest tests/test_i2t_sampler_state_ft.py -q` -> `7 passed`
  - `.venv/bin/ruff check src/uniindex/i2t_sampler_state_ft.py src/uniindex/cli.py tests/test_i2t_sampler_state_ft.py` -> passed
  - `.venv/bin/python -m pytest -q` -> `78 passed`
- `last_block_anchor_w100` metrics:
  - `image_to_text_exact_match = 0.0078125`
  - `image_to_text_token_accuracy = 0.35753749013417524`
  - `image_to_text_label_accuracy_constrained = 0.27734375`
  - `text_to_image_accuracy = 0.796875`
  - `unconditional_consistency = 0.0`
- Raw guard reference (`no_i2t_projection`): exact `0.1328125`, label `0.1484375`, token `0.345698500394633`, t2i `0.95703125`, uncond `0.859375`.
- 32-sample understanding diagnostic for `last_block_anchor_w100`:
  - final true-image sampler: exact `0.0`, label `0.375`, token `0.3782051282051282`, mean true-label margin `-0.5351239442825317`
  - final shuffled-image sampler: exact `0.0`, label `0.0625`, token `0.28205128205128205`, mean true-label margin `-2.1434545516967773`
  - final random-image-token sampler: exact `0.0`, label `0.0625`, token `0.3076923076923077`, mean true-label margin `-1.712289810180664`
  - failure counts: `label_correct_text_wrong = 13`, `label_wrong = 16`, `image_insensitive = 3`
- Decision: do not promote. The first case increased constrained label separation but destroyed free text exact and shared generation, so the planned `last_two_blocks_anchor_w100` case was stopped early before spending another training window.
- Lesson: this result is more informative than a plain failure. The model can be pushed toward image-conditioned class evidence, but the current objective does it by corrupting the sequence decoder/generator. `anchor_weight = 1.0` on final-block sampler-state FT is too destructive; future binding probes need an explicit early-abort gate on raw exact/t2i/uncond and should either use a smaller/lower-rank write surface or change the target so label binding cannot win while text sequence quality collapses.
- Environment note: eval and diagnostics again printed Emu3.5 dynamic-module "downloaded" warnings. Pin/vendor remains required before long unattended runs.

## Recent head-only sampler-state sequence guard

- Run timestamp: `2026-04-25T10:40:32Z` (`2026-04-25 18:40:32 CST`)
- Remote summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampler_state_seq_guard/20260425T104032Z-seq-guard-head-mini-offline/summary.json`
- Follow-up summary: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/sampler_state_seq_guard/20260425T105524Z-seq-guard-head-s50-offline/summary.json`
- Code state: remote detached checkout `907ff94f9b5f4725c7bd537fba6ad5157f2c2088`.
- Method: use the raw `no_i2t_projection` guard, train only `norm` + `head` on sampler-state i2t sequence CE, and keep the active checkpoint as the source. This tests whether a very small write surface can improve real free i2t without corrupting t2i/unconditional.
- Run control:
  - First attempt failed before training because the generated runner had a Python quoting bug.
  - Second attempt completed the first 25-step train but eval hit Hugging Face online HEAD retries; it was stopped and marked `stopped_eval_hf_online_retry`.
  - Final run exported `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and all cache roots under the repo path before train/eval.
- Raw guard reference (`no_i2t_projection`): exact `0.1328125`, token `0.345698500394633`, label `0.1484375`, t2i `0.95703125`, uncond `0.859375`.
- `head_seq_w150_a000_s25` metrics:
  - `image_to_text_exact_match = 0.140625`
  - `image_to_text_token_accuracy = 0.3425414364640884`
  - `image_to_text_label_accuracy_constrained = 0.15234375`
  - `text_to_image_accuracy = 0.96875`
  - `unconditional_consistency = 0.875`
- `head_seq_w150_a010_s25` metrics:
  - `image_to_text_exact_match = 0.13671875`
  - `image_to_text_token_accuracy = 0.3393843725335438`
  - `image_to_text_label_accuracy_constrained = 0.15625`
  - `text_to_image_accuracy = 0.96484375`
  - `unconditional_consistency = 0.890625`
- Decision after 25 steps: weak positive signal, but small. Pure head sequence CE is better for exact; weak anchor helps label/uncond slightly but hurts token. Extend only the pure head sequence case to 50 steps.
- `head_seq_w150_a000_s50` metrics:
  - `image_to_text_exact_match = 0.12890625`
  - `image_to_text_token_accuracy = 0.3322809786898185`
  - `image_to_text_label_accuracy_constrained = 0.14453125`
  - `text_to_image_accuracy = 0.95703125`
  - `unconditional_consistency = 0.84375`
- 50-step understanding diagnostic:
  - final true-image sampler: exact `0.1875`, label `0.1875`, token `0.3974358974358974`, mean true-label margin `-14.1028413772583`
  - final shuffled-image sampler: exact `0.09375`, label `0.09375`, token `0.23717948717948717`, mean true-label margin `-22.013517379760742`
  - final random-image-token sampler: exact `0.15625`, label `0.15625`, token `0.358974358974359`, mean true-label margin `-14.624062538146973`
  - failure counts: `label_wrong = 18`, `exact_correct = 6`, `image_insensitive = 8`
  - visualization paths: `sample_cards.png`, `confusion_matrix.png`, and `train_vs_sampler_margins.png` under `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/fullvocab_long_tsw075_i2tr06_ssb_seq_guard_offline_head_seq_w150_a000_s50/20260425T110212Z-diagnose-i2t-understanding/i2t_understanding/`
- Decision: do not promote and do not keep increasing steps. The 25-step result is a real but tiny raw free-i2t signal; 50 steps loses that gain while preserving t2i/uncond only barely.
- Lesson: the useful window is short and head-local. This supports "small write surface + early stop" over full-model or final-block FT, but the signal is too small and unstable to be an algorithmic win. Next probes should search around the early window (`10/20/30` or `15/25/35` steps) and optimize exact/token directly; do not add more anchor or longer training until the raw exact gain repeats.
- Environment note: even with offline flags, Transformers still prints cached dynamic-module "downloaded" warnings. It no longer blocks on network in this run, but pin/vendor remains required.

## Next experiment

Do not continue increasing `text_sequence_weight`, the low-t/noise-only i2t strategy, lower-gamma/logit-normal sampling, plain second projection, pure candidate-score projection, candidate-score blend, joint-task mismatch loss, pairwise mismatch-margin weight, unregularized full-model sampler-state FT, high-anchor final-block sampler-state FT, weak-anchor head-only FT, longer head-only FT, or tiny-LLM text priors without a new reason. Checkpoint interpolation remains a useful low-cost probe, but the `alpha = 0.375` result is not strong enough to replace the active baseline.

The label-feature probe and direct text-decoder probe shift the next priority away from sampler-only work. SigLIP-VQ clears both cheap semantic gates: VQ-label accuracy `0.69921875` on the earlier 256px gate, and direct text-decoder free exact `0.86328125` with shuffled-image candidate accuracy near chance. The current failure is not "VQ has no understanding signal"; it is that the unified FLM hidden path is not preserving that signal for text generation. Keep `probe-label-features`, `probe-vq-text-decoder`, `probe-i2t-overfit`, and `diagnose-i2t-understanding` in the acceptance loop.

## Recent SigLIP-VQ label-token FLM iteration

- Run window: `2026-04-27T03:18:09Z` to `2026-04-27T03:51:38Z` (`2026-04-27 11:18:09-11:51:38 CST`).
- Code states:
  - `73a67e6`: atomic label text mode (`text.kind = label`).
  - `a4cec98`: train-only pooled-image-hidden semantic auxiliary (`image_to_text_semantic_weight`).
- Local validation:
  - `.venv/bin/python -m pytest tests/test_config.py tests/test_train_schedule.py -q` -> `30 passed`.
  - `.venv/bin/ruff check src/uniindex/train.py src/uniindex/config.py tests/test_config.py tests/test_train_schedule.py` -> passed.
  - `.venv/bin/python -m pytest -q` -> `103 passed, 1 warning`.
- Atomic label-token baseline:
  - Config: `configs/flm_joint_work_siglipvq_generation_labeltoken_probe.yaml`.
  - Diagnose path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_probe_img512/20260427T031809Z-diagnose-i2t/i2t_diagnostics.json`.
  - Result: all progress values produced only `nine`; exact/label `0.109375`, token `0.5546875`.
- Atomic label-token overfit probes:
  - CE-only path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_probe_img512/20260427T031814Z-probe-i2t-overfit/overfit_summary.json`.
  - CE-only final train true-image exact/label `0.625`, shuffled `0.0`; held-out true-image exact/label `0.0625`.
  - Label-weight path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_probe_img512/20260427T032229Z-probe-i2t-overfit/overfit_summary.json`.
  - `label_weight=1.0` final train true-image exact/label `0.8125`, shuffled `0.0`; held-out true-image exact/label `0.1875`, shuffled `0.0625`.
  - Lesson: fixed-sample image binding is learnable; the blocker is held-out generalization, not token spelling.
- Larger label-token short train:
  - Generated config: `configs_generated/flm_joint_work_siglipvq_generation_labeltoken_probe_train512_labelw1.yaml`.
  - Diagnose path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_probe_img512_train512_labelw1/20260427T033050Z-diagnose-i2t/i2t_diagnostics.json`.
  - Result: all progress values produced only `one`; exact/label `0.1171875`, token `0.55859375`.
  - Feature probe path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_probe_img512_train512_labelw1/20260427T033155Z-probe-label-features/summary.json`.
  - Same split probe: direct VQ-token head reached `0.90625` test accuracy, while frozen FLM hidden reached only `0.1484375` best and `0.1171875` final.
  - Lesson: the SigLIP-VQ token sequence contains strong label information, but the trained unified FLM hidden path is nearly class-prior level.
- Pooled semantic auxiliary:
  - Config: `configs/flm_joint_work_siglipvq_generation_labeltoken_semantic_probe.yaml`.
  - Commit: `a4cec98`.
  - Diagnose path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_semantic_probe_img512/20260427T034526Z-diagnose-i2t/i2t_diagnostics.json`.
  - Result: unchanged single-label collapse to `one`; exact/label `0.1171875`, token `0.55859375`.
  - Feature probe path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_semantic_probe_img512/20260427T034532Z-probe-label-features/summary.json`.
  - Direct VQ-token head `0.9140625`; FLM hidden best `0.1484375`, final `0.1171875`.
  - Train log: `semantic_label_loss` stayed near random-class CE (`~2.18-2.45`).
- Strong semantic/i2t-only control:
  - Generated config: `configs_generated/flm_joint_work_siglipvq_generation_labeltoken_semantic_w10_i2tonly.yaml`.
  - Diagnose path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_semantic_w10_i2tonly_img512/20260427T035133Z-diagnose-i2t/i2t_diagnostics.json`.
  - Result: unchanged single-label collapse to `one`; exact/label `0.1171875`, token `0.55859375`.
  - Feature probe path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/siglipvq_generation_labeltoken_semantic_w10_i2tonly_img512/20260427T035138Z-probe-label-features/summary.json`.
  - Direct VQ-token head `0.90625`; FLM hidden best `0.1484375`, final `0.1171875`.
  - Train log: even `semantic_weight=10.0` with i2t-only stage2 left `semantic_label_loss` around `2.13-2.44`.
- Decision: do not promote label-token, plain label loss, or the pooled semantic auxiliary. They prove the current hidden path is not a good semantic bottleneck. The next algorithmic iteration should change the representation path, not add another sampler fallback: add an explicit global image summary route such as a CLS/summary token or dedicated image-to-text conditioning prefix inside the unified FLM, then require the same VQ-token-vs-FLM-hidden probe to show FLM hidden label accuracy moves well above chance before running longer generation experiments.

## Recent FLM image-summary-to-text route probe

- Run window: `2026-04-27T04:05:43Z` to `2026-04-27T04:28:22Z` (`2026-04-27 12:05:43-12:28:22 CST`).
- Code state: GitHub SHA `bad8a2ba8a918fd24d96e034150779dfdd579bdb`.
- Config: `configs/flm_joint_work_siglipvq_generation_labeltoken_summary_probe.yaml`.
- Code change: add optional `model.image_summary_to_text`, which projects the clean image hidden summary into text positions only when image is clean and text is noisy. Also add text/all pooling controls for the semantic auxiliary and label-feature probe.
- Local validation:
  - `.venv/bin/python -m pytest tests/test_config.py tests/test_model.py tests/test_train_schedule.py tests/test_i2t_llm_decoder.py -q` -> `42 passed`.
  - `.venv/bin/python -m pytest -q` -> `107 passed, 1 warning`.
  - `ruff` was unavailable in the local shell for this run.
- Full run control:
  - Initial path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/summary_route/20260427T040543Z-image-summary-text/`.
  - `prepare`, `stage1`, and `stage2` completed.
  - `eval` was stopped after it became dominated by slow image decoding and repeated model reloads (`error:143`); this was intentionally cut because the question for this iteration was text-side semantic binding, not generated image quality.
- Quick diagnostic path: `/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/runs/summary_route/20260427T042718Z-image-summary-text-quick/summary.json`.
- `diagnose-i2t` result:
  - At progress `0.5`, `0.9`, and `0.95`, all generated labels were `one`.
  - Exact/label accuracy stayed `0.1171875`.
  - Token accuracy stayed `0.55859375`.
- Label-feature probe result:
  - Direct VQ-token probe reached `0.921875` test accuracy.
  - FLM text-hidden probe stayed at `0.1171875`; best was also `0.1171875`.
- Stage2 train log:
  - At step 500, `semantic_label_loss = 2.1282246112823486`, `label_loss = 2.118138074874878`, and `sequence_loss = 2.117321014404297`.
  - These values remain close to random ten-class CE, so the model is not learning image-to-label binding during training.
- Decision: do not promote `image_summary_to_text`. A single global summary injection is too weak; it neither changes free i2t output nor makes text hidden linearly/MLP-decodable for labels.
- Lesson: the useful semantic signal is still in the VQ token sequence, but the unified FLM is failing to build a supervised semantic bottleneck from it. The next iteration should stop adding weak residual hints and instead force an explicit bottleneck: a dedicated image semantic token/prefix with direct label supervision that text positions must attend to, or a two-head unified FLM objective where the semantic token is part of the denoising state rather than a train-only auxiliary head.

## Archived local trees

- `visualize-joint-work`
- `schedule-fix-analysis`
- `uniindex-formal-run`

Treat them as read-only references unless they are explicitly reactivated.
