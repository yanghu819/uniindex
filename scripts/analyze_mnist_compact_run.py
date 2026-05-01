from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/Users/torusmini/Downloads/uniindex")
DUMP_PATH = Path("/tmp/compact_remote_dump.txt")
OUT_DIR = ROOT / "docs" / "analysis" / "mnist_compact_20260415"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_sections(text: str) -> dict[str, str]:
    markers = ["__METRICS__", "__STAGE1__", "__STAGE2__"]
    sections: dict[str, str] = {}
    for i, marker in enumerate(markers):
        start = text.find(marker)
        if start == -1:
            raise RuntimeError(f"missing marker {marker}")
        start += len(marker)
        end = len(text)
        for next_marker in markers[i + 1 :]:
            pos = text.find(next_marker, start)
            if pos != -1:
                end = pos
                break
        sections[marker.strip("_").lower()] = text[start:end]
    return sections


def parse_metrics(section: str) -> dict[str, float]:
    start = section.find("{")
    end = section.find("}", start)
    if start != -1 and end != -1:
        return json.loads(section[start : end + 1])
    raise RuntimeError("metrics json not found")


def parse_jsonl(section: str) -> list[dict]:
    rows: list[dict] = []
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("no jsonl rows found")
    return rows


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        ".title { font: 700 20px sans-serif; fill: #111827; }",
        ".subtitle { font: 12px sans-serif; fill: #4b5563; }",
        ".axis { font: 11px monospace; fill: #374151; }",
        ".label { font: 12px sans-serif; fill: #111827; }",
        ".small { font: 10px monospace; fill: #374151; }",
        '</style>',
    ]


def build_metrics_svg(metrics: dict[str, float], out_path: Path) -> None:
    width, height = 880, 340
    left, top, chart_w, chart_h = 180, 70, 620, 210
    labels = [
        ("Tokenizer Ceiling", metrics["tokenizer_ceiling"], "#9ca3af"),
        ("Image -> Label", metrics["image_to_label_accuracy"], "#ef4444"),
        ("Label -> Image", metrics["label_to_image_accuracy"], "#10b981"),
        ("Unconditional", metrics["unconditional_consistency"], "#3b82f6"),
    ]
    lines = _svg_header(width, height)
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append('<text x="24" y="34" class="title">MNIST Compact Baseline Metrics</text>')
    lines.append('<text x="24" y="54" class="subtitle">run 20260415T044225Z-eval, compact_vocab=true, unified-index + one-hot FLM</text>')
    for i in range(6):
        y = top + chart_h - i * (chart_h / 5)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{i/5:.1f}</text>')
    bar_h = 34
    gap = 18
    for idx, (label, value, color) in enumerate(labels):
        y = top + idx * (bar_h + gap)
        bar_w = chart_w * value
        lines.append(f'<text x="24" y="{y + 22}" class="label">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" fill="#f3f4f6" rx="6"/>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="{color}" rx="6"/>')
        lines.append(f'<text x="{left + bar_w + 10:.1f}" y="{y + 22}" class="axis">{value:.4f}</text>')
    lines.append('</svg>')
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_loss_svg(stage1_rows: list[dict], stage2_rows: list[dict], out_path: Path) -> None:
    width, height = 980, 520
    lines = _svg_header(width, height)
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append('<text x="24" y="32" class="title">Loss Decomposition</text>')
    lines.append('<text x="24" y="52" class="subtitle">Stage1 joint training keeps image loss high while label loss collapses; stage2 remains asymmetric.</text>')

    def panel(origin_x: int, origin_y: int, panel_w: int, panel_h: int, rows: list[dict], series: list[tuple[str, str]], title: str) -> None:
        max_step = max(r["step"] for r in rows)
        max_val = max(max(r[name] for name, _ in series) for r in rows)
        left = origin_x + 58
        top = origin_y + 24
        chart_w = panel_w - 82
        chart_h = panel_h - 60
        lines.append(f'<text x="{origin_x}" y="{origin_y}" class="label">{title}</text>')
        for i in range(6):
            y = top + chart_h - i * (chart_h / 5)
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{(max_val * i / 5):.1f}</text>')
        for i in range(6):
            x = left + i * (chart_w / 5)
            lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}" stroke="#f3f4f6" stroke-width="1"/>')
            lines.append(f'<text x="{x:.1f}" y="{top + chart_h + 18}" text-anchor="middle" class="axis">{int(max_step * i / 5)}</text>')
        for name, color in series:
            points = []
            for row in rows:
                x = left + chart_w * (row["step"] / max_step)
                y = top + chart_h * (1 - row[name] / max_val)
                points.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        legend_x = origin_x + panel_w - 140
        legend_y = origin_y + 6
        for idx, (name, color) in enumerate(series):
            lines.append(f'<line x1="{legend_x}" y1="{legend_y + idx*18}" x2="{legend_x + 18}" y2="{legend_y + idx*18}" stroke="{color}" stroke-width="3"/>')
            lines.append(f'<text x="{legend_x + 24}" y="{legend_y + idx*18 + 4}" class="small">{name}</text>')

    panel(24, 86, 440, 390, stage1_rows, [("loss", "#111827"), ("image_loss", "#2563eb"), ("label_loss", "#dc2626")], "Stage1 Joint")

    stage2_image_to_label = [row for row in stage2_rows if row["task"] == "image_to_label"]
    stage2_label_to_image = [row for row in stage2_rows if row["task"] == "label_to_image"]
    merged_rows = []
    for idx, row in enumerate(stage2_image_to_label):
        merged_rows.append(
            {
                "step": idx + 1,
                "image_to_label_loss": row["loss"],
                "image_to_label_label_t": row["label_t_mean"],
                "label_to_image_loss": stage2_label_to_image[idx]["loss"],
            }
        )
    panel(
        504,
        86,
        452,
        390,
        merged_rows,
        [("image_to_label_loss", "#dc2626"), ("label_to_image_loss", "#10b981"), ("image_to_label_label_t", "#7c3aed")],
        "Stage2 Task Split",
    )
    lines.append('</svg>')
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_summary_md(metrics: dict[str, float], stage1_rows: list[dict], stage2_rows: list[dict], out_path: Path) -> None:
    stage1_first = stage1_rows[0]
    stage1_last = stage1_rows[-1]
    image_to_label_rows = [row for row in stage2_rows if row["task"] == "image_to_label"]
    label_to_image_rows = [row for row in stage2_rows if row["task"] == "label_to_image"]

    def avg(rows: list[dict], key: str) -> float:
        return sum(float(row[key]) for row in rows) / max(len(rows), 1)

    text = f"""# MNIST Compact Baseline Analysis

Run:
- Eval: `20260415T044225Z-eval`
- Stage1: `20260415T043457Z-stage1`
- Stage2: `20260415T043721Z-stage2`

Metrics:
- `tokenizer_ceiling = {metrics['tokenizer_ceiling']:.4f}`
- `image_to_label_accuracy = {metrics['image_to_label_accuracy']:.4f}`
- `label_to_image_accuracy = {metrics['label_to_image_accuracy']:.4f}`
- `unconditional_consistency = {metrics['unconditional_consistency']:.4f}`

Stage1:
- `loss`: {stage1_first['loss']:.3f} -> {stage1_last['loss']:.3f}
- `image_loss`: {stage1_first['image_loss']:.3f} -> {stage1_last['image_loss']:.3f}
- `label_loss`: {stage1_first['label_loss']:.3f} -> {stage1_last['label_loss']:.3f}

Stage2 averages:
- `label_to_image loss avg = {avg(label_to_image_rows, 'loss'):.3f}`
- `image_to_label loss avg = {avg(image_to_label_rows, 'loss'):.3f}`
- `image_to_label label_t_mean avg = {avg(image_to_label_rows, 'label_t_mean'):.3f}`

Interpretation:
- 生成侧已经 work。`label_to_image_accuracy` 到了 `0.8047`，不是随机碰运气。
- 理解侧没有对称起来。`image_to_label_accuracy` 只有 `0.1523`，明显落后。
- `stage1` 里 `label_loss` 很快掉到接近 0，而 `image_loss` 长时间还在 `4.5~5.5` 区间。这说明统一模型先学会了标签位，不是先学会了图像位。
- `stage2` 里两类任务继续分化：`label_to_image` 的条件标签始终是 clean，loss 稳定在 `4.1~4.8` 的图像恢复；`image_to_label` 的 `label_t_mean` 只在 `0.1~0.3` 左右，说明标签位恢复仍然处在高噪声、小容量、脆弱判别的 regime。
- 当前瓶颈不是 tokenizer ceiling。`0.9961` 说明上限几乎满了，问题在 unified one-hot FLM 的优化分配。

Current full-vocab run:
- `configs/flm_joint_work_fullvocab.yaml` 正在远端 `prepare`，还没进入可比较的 `stage1/stage2/eval`。
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    text = DUMP_PATH.read_text(encoding="utf-8", errors="ignore")
    sections = parse_sections(text)
    metrics = parse_metrics(sections["metrics"])
    stage1_rows = parse_jsonl(sections["stage1"])
    stage2_rows = parse_jsonl(sections["stage2"])

    write_json(OUT_DIR / "metrics.json", metrics)
    write_json(OUT_DIR / "stage1_rows.json", stage1_rows)
    write_json(OUT_DIR / "stage2_rows.json", stage2_rows)
    build_metrics_svg(metrics, OUT_DIR / "metrics.svg")
    build_loss_svg(stage1_rows, stage2_rows, OUT_DIR / "losses.svg")
    build_summary_md(metrics, stage1_rows, stage2_rows, OUT_DIR / "analysis.md")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
