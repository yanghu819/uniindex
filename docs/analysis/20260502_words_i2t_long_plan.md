# Words Image-To-Text Long Run Plan

Timestamp: `2026-05-02T15:12:00+0800`

## Correction

The relevant language target is the English class word form:

```text
zero, one, two, three, four, five, six, seven, eight, nine
```

The earlier `0` to `9` single-character run is not the same task.

## Current Known Baseline

Old English-word image-to-text eval:

- `exact_match`: about `0.117`
- `token_accuracy`: about `0.344`
- `constrained_label_accuracy`: about `0.188`

This baseline used a much smaller setting than the later 30k single-character run, so it is not a fair final result.

## Hypothesis

The English-word failure may be caused by two practical issues:

- sampling with only `8` steps is too coarse for multi-character text;
- the English-word model was undertrained compared with later 30k GPU runs.

## Next Run

Config: `configs/flm_understanding_words_i2t_30k_long_gpu80.yaml`

Changes:

- 30k train / 5k test;
- explicit word targets;
- stage1 `1200` steps;
- stage2 `8000` image-to-text steps;
- eval sampling `32` steps;
- temperature `0.7`.

Readout:

- If exact rises sharply, the old failure was mostly undertraining plus too-few sampling steps.
- If exact remains low while constrained is high, the issue is free text sequence generation.
- If constrained also remains low, the image-to-word conditional signal is still not learned well enough.
