from __future__ import annotations

import math

import torch

from .config import ProjectConfig


def _estimate_decode_error_curve(
    vocab_size: int,
    num_points: int,
    num_samples: int,
    min_t: float,
) -> dict[str, torch.Tensor]:
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be at least 2, got {vocab_size}")

    t_grid = torch.linspace(float(min_t), 1.0, steps=num_points, dtype=torch.float32)
    if num_points == 1:
        return {
            "t_grid": t_grid,
            "error_grid": torch.zeros_like(t_grid),
            "progress_grid": torch.ones_like(t_grid),
        }

    active_t = t_grid[:-1].clamp(max=1.0 - 1e-4).to(torch.float64)
    mu = active_t / (1.0 - active_t)
    samples = torch.randn(num_samples, active_t.numel(), dtype=torch.float64) + mu.unsqueeze(0)
    phi = 0.5 * (1.0 + torch.erf(samples / math.sqrt(2.0)))
    success = torch.exp((vocab_size - 1) * torch.log(phi.clamp_min(1e-12))).mean(dim=0)
    errors = torch.cat([1.0 - success, torch.zeros(1, dtype=torch.float64)], dim=0)

    monotonic = errors.clone()
    for index in range(1, monotonic.numel()):
        monotonic[index] = min(monotonic[index - 1], monotonic[index])

    base_error = float(monotonic[0].item())
    denom = max(base_error - float(monotonic[-1].item()), 1e-8)
    solved = (base_error - monotonic) / denom
    solved[0] = 0.0
    solved[-1] = 1.0
    monotonic_solved = solved.clone()
    for index in range(1, monotonic_solved.numel()):
        monotonic_solved[index] = max(monotonic_solved[index - 1], monotonic_solved[index])

    return {
        "t_grid": t_grid,
        "error_grid": monotonic.to(torch.float32),
        "progress_grid": monotonic_solved.to(torch.float32),
    }


def build_schedule_tables(config: ProjectConfig, image_vocab_size: int, text_vocab_size: int) -> dict[str, dict[str, torch.Tensor] | str]:
    if config.schedule.kind == "power":
        return {"kind": "power"}
    if config.schedule.kind != "empirical":
        raise ValueError(f"unsupported schedule kind: {config.schedule.kind}")
    return {
        "kind": "empirical",
        "image": _estimate_decode_error_curve(
            vocab_size=image_vocab_size,
            num_points=config.schedule.num_points,
            num_samples=config.schedule.num_samples,
            min_t=config.schedule.min_t,
        ),
        "text": _estimate_decode_error_curve(
            vocab_size=text_vocab_size,
            num_points=config.schedule.num_points,
            num_samples=config.schedule.num_samples,
            min_t=config.schedule.min_t,
        ),
    }


def _lookup_progress(progress: torch.Tensor, progress_grid: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    flat = progress.reshape(-1).clamp(0.0, 1.0)
    grid = progress_grid.to(progress.device)
    t_values = t_grid.to(progress.device)
    indices = torch.bucketize(flat, grid[1:-1], right=False)
    left_index = indices
    right_index = (indices + 1).clamp_max(grid.numel() - 1)
    left_progress = grid[left_index]
    right_progress = grid[right_index]
    left_t = t_values[left_index]
    right_t = t_values[right_index]
    denom = (right_progress - left_progress).clamp_min(1e-6)
    alpha = ((flat - left_progress) / denom).clamp(0.0, 1.0)
    return (left_t + alpha * (right_t - left_t)).reshape_as(progress)


def apply_schedule(
    progress: torch.Tensor,
    modality_ids: torch.Tensor,
    schedule_tables: dict[str, dict[str, torch.Tensor] | str],
    *,
    image_time_power: float,
    text_time_power: float,
) -> torch.Tensor:
    if progress.dim() != 1:
        raise ValueError(f"expected progress to have shape (batch,), got {tuple(progress.shape)}")
    if schedule_tables["kind"] == "power":
        powers = torch.where(
            modality_ids.long() == 0,
            torch.full_like(modality_ids, float(image_time_power), dtype=torch.float32),
            torch.full_like(modality_ids, float(text_time_power), dtype=torch.float32),
        ).to(progress.device)
        return progress[:, None].pow(powers.unsqueeze(0))

    image_schedule = schedule_tables["image"]
    text_schedule = schedule_tables["text"]
    image_progress = progress.pow(float(image_time_power))
    text_progress = progress.pow(float(text_time_power))
    image_t = _lookup_progress(image_progress, image_schedule["progress_grid"], image_schedule["t_grid"])
    text_t = _lookup_progress(text_progress, text_schedule["progress_grid"], text_schedule["t_grid"])
    return torch.where(
        modality_ids.unsqueeze(0).to(progress.device) == 0,
        image_t.unsqueeze(1),
        text_t.unsqueeze(1),
    )
