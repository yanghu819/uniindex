# Structured Robot Probe Results

Timestamp: `2026-05-02T13:32:31+0800`

Remote target: A100 on SSH port `30186`

## Goal

Use the current image-to-text path as pre-research for robot-scale multimodal understanding. MNIST remains only a cheap proxy for fast mechanism tests; the target is to learn which design choices survive before moving to larger robot data and bigger clusters.

## Completed Run

Config: `configs/flm_robot_probe_structured_i2t_30k_gpu80.yaml`

Checkpoint commit: `b459618`

Evaluation commit: `c2dd0ea`

Train data: 30k proxy images with long structured text targets such as:

```text
class=digit_8; loops=2; parity=even; pose=double_loop
```

Training:

- Stage1: 800 steps
- Stage2: 4000 image-to-text steps
- GPU during training: roughly 99-100% utilization, 69-75GB memory

Final observed losses:

- Stage1 step 800: `loss=3.019`, `text_loss=0.121`, `sequence_loss=0.082`
- Stage2 step 4000: `loss=0.4516`, `text_loss=0.2213`, `sequence_loss=0.1152`

## Field Evaluation

Metrics on 5k test samples:

- `exact_match`: `0.0526`
- `token_accuracy`: `0.84342934402086`
- `constrained_label_accuracy`: `0.4618`
- `parseable_rate`: `1.0`

Field accuracy:

- `class`: `0.3894`
- `loops`: `0.542`
- `parity`: `0.627`
- `pose`: `0.0782`

## Insight

The model is not blank. It learned the syntax well enough that outputs are parseable and token accuracy is high. But long structured strings are brittle: small character errors and field-tail collapse destroy exact match.

The `pose` field is the clearest failure. It appears late in the string and contains long values like `double_loop`, `loop_tail`, and `loop_head`. The generated outputs include many near-valid spellings, which points to a surface-form and sequence-reliability problem, not only an image-understanding problem.

Constrained full-string scoring gives `0.4618`, much better than exact generation but far below the earlier single-character digit probe. This suggests that evaluation and output representation are now part of the research object.

## Lesson

For robot-scale pre-research, do not treat free-form generated text exact match as the only success metric. Track:

- token accuracy,
- constrained candidate accuracy,
- per-field accuracy,
- parseability,
- output distribution skew,
- position-sensitive field failures.

## Next Experiment

Run a compact structured schema with shorter field/value symbols. This tests whether the current limitation is long character-level serialization.

Proposed target strings:

```text
c=0;l=1;p=E;q=R0
c=8;l=2;p=E;q=D2
```

Expected readout:

- If compact schema improves exact and field accuracy sharply, the main bottleneck is output serialization.
- If compact schema still fails similarly, the bottleneck is deeper image-to-state grounding or training dynamics.
