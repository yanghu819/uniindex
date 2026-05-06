# 2026-05-06 text-to-image collapse decoded sweep

Timestamp:

- Local: 2026-05-06 12:45:47 CST
- UTC: 2026-05-06 04:45:47 UTC

## Goal

Focus the next iteration on `text -> image` generation collapse while keeping the unified FLM architecture. The decoder path is now treated as working, so generation quality is judged with true decoded images plus token-space NN metrics, not token heatmaps.

## Baseline

Base checkpoint:

- `runs/20260505T012638Z-stage2-continue/checkpoints/stage2_step004500.pt`

Base config:

- `configs/flm_words_joint_long_rope_p8_i2theavy_gpu80.yaml`

Standard decoded baseline:

- NFE: 256
- temperature: 0.7
- samples: 16 per digit, 160 total
- decoded token-NN accuracy: 0.2375
- nearest-label counts: `9:81, 1:60, 7:19`
- unique nearest labels: 3

Sampling sweep over NFE `64/128/256/512` and temperature `0.5/0.7/1.0` did not fix collapse. Best decoded token-NN point was:

- NFE: 512
- temperature: 0.5
- decoded token-NN accuracy: 0.2750
- nearest-label counts: `9:83, 1:56, 7:19, 4:2`

Conclusion: sampling-only changes improve little and keep the same `9/1/7` concentration.

## 500-step training experiments

All experiments continue from the same base checkpoint and keep the same unified transformer/FLM setup. No special modality head was added.

| Experiment | Stage2 task ratio `joint:t2i:i2t` | i2t exact 512 | t2i token-NN 512 | decoded token-NN 160 | decoded nearest-label counts |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | previous i2t-heavy checkpoint | ~0.8320 | 0.2793 | 0.2375 | `9:81, 1:60, 7:19` |
| gen124 | `1:2:4` | 0.8047 | 0.3379 | 0.2938 | `9:65, 7:51, 1:44` |
| gen144 | `1:4:4` | 0.7813 | 0.2813 | 0.2813 | `9:115, 1:24, 7:21` |
| gen142 | `1:4:2` | 0.7969 | 0.3359 | 0.3000 | `9:98, 1:36, 7:19, 4:7` |

Run directories:

- gen124: `runs/20260506T033603Z-stage2-continue`
- gen144: `runs/20260506T040221Z-stage2-continue`
- gen142: `runs/20260506T042157Z-stage2-continue`

Local decoded grids copied under ignored `logs/visualizations/`:

- `logs/visualizations/t2i_decoded_baseline_step4500_20260506T0305Z/decoded_grid_steps256_temp0.70.png`
- `logs/visualizations/t2i_decoded_gen124_500_20260506T0345Z/decoded_grid_steps256_temp0.70.png`
- `logs/visualizations/t2i_decoded_gen144_500_20260506T0410Z/decoded_grid_steps256_temp0.70.png`
- `logs/visualizations/t2i_decoded_gen142_500_20260506T0430Z/decoded_grid_steps256_temp0.70.png`

## Findings

1. The true decoded generation is better than token-NN alone suggests. In decoded grids, the left generated digit often follows the prompt shape for many labels, while the nearest real token-space neighbor is still labeled `9/1/7`.
2. Token-NN remains useful as a collapse alarm, but it is not a complete visual-quality metric after decoding. It can mark a generated zero/two/four/five/eight as nearest to nine even when the decoded left image is visibly condition-shaped.
3. `gen124` is the best balanced candidate among the three small training probes: it gives the largest 512-sample t2i token-NN gain while losing less i2t than `gen144` and about the same decoded quality as `gen142`.
4. `gen144` is not worth scaling: it hurts i2t most and concentrates decoded NN labels further into `9`.
5. `gen142` has the best 160-sample decoded token-NN score, but its 512-sample t2i token-NN is not better than gen124 and its nearest-label distribution is more `9`-heavy.

## Decision

Do not scale `gen144`.

Use `gen124` as the current best candidate if scaling one branch now. It is not accepted as solved, because i2t exact drops from about 83% to 80.5% and token-NN collapse still exists, but it is the clearest positive direction among the 500-step probes.

## Longer training probe

Timestamp:

- Local: 2026-05-06 14:10:31 CST
- UTC: 2026-05-06 06:10:31 UTC

Because `gen124` was the best balanced 500-step probe, it was extended from step 5000 to step 6500.

Extended checkpoint:

- `runs/20260506T053542Z-stage2-continue/checkpoints/stage2_step006500.pt`

Longer-training results:

| Experiment | Step | i2t exact 512 | t2i token-NN 512 | decoded token-NN 160 | decoded nearest-label counts |
| --- | ---: | ---: | ---: | ---: | --- |
| gen124 | 5000 | 0.8047 | 0.3379 | 0.2938 | `9:65, 7:51, 1:44` |
| gen124 extended | 6500 | 0.8164 | 0.3164 | 0.2938 | `9:105, 7:32, 1:21, 6:2` |

Finding: training longer helped `image -> text` recover from 80.5% to 81.6%, but it did not improve `text -> image`. The 512-sample t2i token-NN score dropped, and the decoded nearest-label distribution became more concentrated on `9`.

Decision: do not blindly scale this training recipe. Step 5000 is still the better generation checkpoint for `gen124`; step 6500 is only better if preserving understanding is the priority.

## Next recommended experiments

1. Add a decoded-image classifier metric or OCR-like lightweight judge for generated images. The current nearest-neighbor token metric is too pessimistic for some decoded samples, but it still catches collapse.
2. Evaluate step 5500 and 6000 only if a finer early-stopping curve is needed. The final 6500 point already shows that longer training alone can over-concentrate generation.
3. Try a targeted anti-collapse change while keeping the model unified, for example task schedule/curriculum changes rather than modality-specific heads.
4. Keep NFE 256 and temperature 0.7 for comparison runs. Sampling sweep already showed that spending more NFE alone is not the main fix.

## Silent fallback audit

- No Qwen route was used.
- No Hugging Face download path was used during these runs; decoder uses the repo-local cache.
- No token heatmap was used as the generation image.
- No architecture fallback to separate heads was introduced.
- The only metric fallback retained is token-NN, and the new finding is that it should be paired with true decoded visual inspection or a decoded-image classifier.
