# MNIST Compact Baseline Analysis

Run:
- Eval: `20260415T044225Z-eval`
- Stage1: `20260415T043457Z-stage1`
- Stage2: `20260415T043721Z-stage2`

Metrics:
- `tokenizer_ceiling = 0.9961`
- `image_to_label_accuracy = 0.1523`
- `label_to_image_accuracy = 0.8047`
- `unconditional_consistency = 0.6406`

Stage1:
- `loss`: 7.503 -> 2.741
- `image_loss`: 10.400 -> 4.697
- `label_loss`: 2.303 -> 0.393

Stage2 averages:
- `label_to_image loss avg = 4.582`
- `image_to_label loss avg = 2.123`
- `image_to_label label_t_mean avg = 0.190`

Interpretation:
- 生成侧已经 work。`label_to_image_accuracy` 到了 `0.8047`，不是随机碰运气。
- 理解侧没有对称起来。`image_to_label_accuracy` 只有 `0.1523`，明显落后。
- `stage1` 里 `label_loss` 很快掉到接近 0，而 `image_loss` 长时间还在 `4.5~5.5` 区间。这说明统一模型先学会了标签位，不是先学会了图像位。
- `stage2` 里两类任务继续分化：`label_to_image` 的条件标签始终是 clean，loss 稳定在 `4.1~4.8` 的图像恢复；`image_to_label` 的 `label_t_mean` 只在 `0.1~0.3` 左右，说明标签位恢复仍然处在高噪声、小容量、脆弱判别的 regime。
- 当前瓶颈不是 tokenizer ceiling。`0.9961` 说明上限几乎满了，问题在 unified one-hot FLM 的优化分配。

Current full-vocab run:
- `configs/flm_joint_work_fullvocab.yaml` 正在远端 `prepare`，还没进入可比较的 `stage1/stage2/eval`。
