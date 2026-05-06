from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

from uniindex.config import load_config
from uniindex.data import build_loader, load_tokenizer_state, split_path
from uniindex.layout import TaskLayout, unified_targets
from uniindex.model import UnifiedDenoiser
from uniindex.runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from uniindex.schedule import build_schedule_tables
from uniindex.state import build_flm_clean_state, condition_clean_timesteps
from uniindex.task_schedule import task_for_step
from uniindex.text import metadata_from_state, shifted_label_text_tokens, text_scoring_mask
from uniindex.train import (
    _build_zt,
    _loss_for_task,
    _save_checkpoint,
    _task_time_schedule,
    latest_checkpoint_path,
)


def _infinite(loader):
    while True:
        yield from loader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-config")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--train-image-to-text-text-time-power", type=float)
    parser.add_argument("--extra-steps", type=int, required=True)
    parser.add_argument("--out-dir", default="logs/experiments/stage2_continue")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.train_image_to_text_text_time_power is not None and args.checkpoint_config is None:
        raise ValueError(
            "--train-image-to-text-text-time-power changes the output checkpoint identity; "
            "pass --checkpoint-config explicitly or use a checked-in config for the target experiment."
        )
    if args.train_image_to_text_text_time_power is not None:
        config = replace(
            config,
            train=replace(
                config.train,
                image_to_text_text_time_power=args.train_image_to_text_text_time_power,
            ),
            sampling=replace(
                config.sampling,
                image_to_text_text_time_power=args.train_image_to_text_text_time_power,
            ),
        )
    if args.checkpoint_path is not None and args.checkpoint_config is not None:
        raise ValueError("pass only one of --checkpoint-path or --checkpoint-config")
    checkpoint_config = load_config(args.checkpoint_config or args.config)
    ensure_project_dirs(config)
    set_seed(config.train.seed + 1009)
    device = resolve_device(config.train.device, config.train.gpu_index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_context = RunContext(config, "stage2-continue")
    run_context.set_device(device)

    tokenizer_state = load_tokenizer_state(config)
    text_metadata = metadata_from_state(tokenizer_state)
    layout = TaskLayout(
        image_seq_len=int(tokenizer_state["image_seq_len"]),
        text_seq_len=int(text_metadata.seq_len),
        codebook_size=int(tokenizer_state["codebook_size"]),
        text_vocab_size=int(text_metadata.vocab_size),
    )
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    shifted_label_tokens = shifted_label_text_tokens(text_metadata, token_offset=layout.codebook_size)
    label_text_mask = text_scoring_mask(
        text_metadata.label_text_tokens,
        text_metadata,
        include_bos=False,
        include_eos=True,
    )

    loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(loader)

    model = UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        position_encoding=config.model.position_encoding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    stage2_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else latest_checkpoint_path(checkpoint_config, "stage2")
    )
    if not stage2_path.exists():
        raise FileNotFoundError(f"missing stage2 checkpoint at {stage2_path}")
    payload = torch.load(stage2_path, map_location=device)
    model.load_state_dict(payload["model"])
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    start_step = int(payload.get("step", 0))

    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    train_log = out_dir / "train_continue.jsonl"
    last_checkpoint = None

    append_jsonl(
        out_dir / "metadata.jsonl",
        {
            "config": args.config,
            "checkpoint_config": args.checkpoint_config or args.config,
            "checkpoint_path": args.checkpoint_path,
            "stage2_checkpoint": str(stage2_path),
            "start_step": start_step,
            "extra_steps": args.extra_steps,
            "train_image_to_text_text_time_power": config.train.image_to_text_text_time_power,
        },
    )

    model.train()
    for offset in tqdm(range(1, args.extra_steps + 1), desc="stage2-continue"):
        step = start_step + offset
        batch = next(train_iter)
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
        x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
        progress = torch.rand(image_tokens.shape[0], device=device)
        task = task_for_step(config, "stage2", step - 1)
        t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, task)
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=task == "image_to_text",
            condition_text=task == "text_to_image",
        )
        z_t = _build_zt(x1, t_pos, layout, task, valid_token_mask)
        logits = model(z_t, t_pos, modality_ids)
        loss, parts = _loss_for_task(
            logits=logits,
            targets=targets,
            layout=layout,
            joint_weight=config.train.joint_weight,
            text_weight=config.train.text_weight,
            text_pad_id=text_metadata.pad_id,
            label_text_tokens=shifted_label_tokens,
            label_text_mask=label_text_mask,
            text_sequence_weight=config.train.text_sequence_weight,
            task=task,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
        optimizer.step()

        if offset == 1 or offset % config.train.log_every == 0 or offset == args.extra_steps:
            append_jsonl(
                train_log,
                {
                    "step": step,
                    "offset": offset,
                    "task": task,
                    "loss": float(loss.item()),
                    "image_t_mean": float(t_pos[:, layout.image_slice].mean().item()),
                    "text_t_mean": float(t_pos[:, layout.text_slice].mean().item()),
                    **parts,
                },
            )

        if offset % config.train.save_every == 0 or offset == args.extra_steps:
            last_checkpoint = _save_checkpoint(config, "stage2", model, optimizer, step, tokenizer_state, run_context)

    run_context.update_status("ok")
    print(
        json.dumps(
            {
                "run_dir": str(run_context.run_dir),
                "log_dir": str(out_dir),
                "last_checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
                "final_step": start_step + args.extra_steps,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
