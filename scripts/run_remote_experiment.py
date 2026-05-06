from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AISTATION_HELPER = Path("/Users/torusmini/.codex/skills/aistation-skill/scripts/aistation_api.js")
DEFAULT_REMOTE_ROOT = "/fangxueji/Projects/PG/uniindex"
DEFAULT_SSH_HOST = "172.16.78.10"
DEFAULT_SSH_PORT = 30186
DEFAULT_SSH_USER = "root"
DEFAULT_AISTATION_NAME = "A100"
FORBIDDEN_COMMAND_SUBSTRINGS = (" scp ", "\tscp ", " rsync ", "\trsync ", "pip install", "uv pip install", "huggingface-cli")
OFFLINE_ENV = {
    "PYTHONPATH": "src",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HOME": f"{DEFAULT_REMOTE_ROOT}/.cache/huggingface",
    "TRANSFORMERS_CACHE": f"{DEFAULT_REMOTE_ROOT}/.cache/huggingface/transformers",
}


class RunnerError(RuntimeError):
    def __init__(self, message: str, *, stage: str | None = None, command: str | None = None, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.command = command
        self.exit_code = exit_code


@dataclass(frozen=True)
class Experiment:
    path: Path
    experiment_id: str
    hypothesis: str
    code_sha: str
    config: str
    base_checkpoint: str
    train_steps: int
    reset_optimizer: bool
    override_lr: float | None
    eval: dict[str, Any]
    acceptance: dict[str, Any]
    base_checkpoint_note: str | None
    next_step_basis: str | None


@dataclass
class CommandResult:
    exit_code: int
    output: str
    command: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def run_local(args: list[str], *, cwd: Path = REPO_ROOT, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RunnerError(
            f"local command failed: {' '.join(shlex.quote(part) for part in args)}",
            command=" ".join(args),
            exit_code=result.returncode,
        )
    return result


def current_git_sha() -> str:
    return run_local(["git", "rev-parse", "HEAD"]).stdout.strip()


def current_branch() -> str:
    return run_local(["git", "branch", "--show-current"]).stdout.strip()


def load_experiment(path: Path, *, code_sha_override: str | None) -> Experiment:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RunnerError(f"experiment YAML must be a mapping: {path}")
    required = ["experiment_id", "hypothesis", "code_sha", "config", "base_checkpoint", "train_steps"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise RunnerError(f"missing required experiment keys in {path}: {', '.join(missing)}")
    experiment_id = str(raw["experiment_id"])
    if not experiment_id or any(char.isspace() for char in experiment_id) or "/" in experiment_id:
        raise RunnerError(f"invalid experiment_id {experiment_id!r}; use a single path-safe id")
    code_sha = code_sha_override or str(raw["code_sha"])
    if code_sha == "AUTO":
        code_sha = current_git_sha()
    train_steps = int(raw["train_steps"])
    if train_steps <= 0:
        raise RunnerError("train_steps must be positive")
    eval_config = dict(raw.get("eval", {}))
    eval_config.setdefault("sampling_steps", 256)
    eval_config.setdefault("temperature", 0.7)
    eval_config.setdefault("max_samples", 512)
    eval_config.setdefault("bank_samples", 5000)
    eval_config.setdefault("decoded_repeats_per_label", 16)
    acceptance = dict(raw.get("acceptance", {}))
    acceptance.setdefault("i2t_exact_min", 0.80)
    acceptance.setdefault("collapse_labels", [9, 1, 7])
    return Experiment(
        path=path,
        experiment_id=experiment_id,
        hypothesis=str(raw["hypothesis"]),
        code_sha=code_sha,
        config=str(raw["config"]),
        base_checkpoint=str(raw["base_checkpoint"]),
        train_steps=train_steps,
        reset_optimizer=bool(raw.get("reset_optimizer", False)),
        override_lr=None if raw.get("override_lr") is None else float(raw["override_lr"]),
        eval=eval_config,
        acceptance=acceptance,
        base_checkpoint_note=None if raw.get("base_checkpoint_note") is None else str(raw["base_checkpoint_note"]),
        next_step_basis=None if raw.get("next_step_basis") is None else str(raw["next_step_basis"]),
    )


def q(value: str | Path | int | float) -> str:
    return shlex.quote(str(value))


def offline_prefix(remote_root: str) -> str:
    env = dict(OFFLINE_ENV)
    env["HF_HOME"] = f"{remote_root}/.cache/huggingface"
    env["TRANSFORMERS_CACHE"] = f"{remote_root}/.cache/huggingface/transformers"
    return " ".join(f"{key}={q(value)}" for key, value in env.items())


def reject_forbidden_command(command: str) -> None:
    padded = f" {command} "
    for forbidden in FORBIDDEN_COMMAND_SUBSTRINGS:
        if forbidden in padded:
            raise RunnerError(f"forbidden command fragment detected: {forbidden.strip()}")


def extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            best = value
    if best is None:
        raise RunnerError("could not parse JSON object from command output")
    return best


class Heartbeat:
    def __init__(self, runner: "RemoteExperimentRunner", phase: str, command: str, interval: int) -> None:
        self.runner = runner
        self.phase = phase
        self.command = command
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "Heartbeat":
        self.runner.write_heartbeat(self.phase, self.command)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.runner.write_heartbeat(self.phase, self.command)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.runner.write_heartbeat(self.phase, self.command)


class RemoteExperimentRunner:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        password: str | None,
    ) -> None:
        self.args = args
        self.password = password
        self.remote_root = args.remote_root.rstrip("/")
        self.ssh_target = f"{args.ssh_user}@{args.ssh_host}"
        self.helper = Path(args.aistation_helper)
        self.record_dir = REPO_ROOT / "docs" / "analysis" / "remote_experiments"
        self.runner_log_root = REPO_ROOT / "logs" / "runner"
        self.last_output_lines: list[str] = []
        self.max_tail_lines = 80

    def run_status_only(self) -> None:
        status = self.aistation_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))

    def run_experiment(self, experiment: Experiment) -> dict[str, Any]:
        self.last_output_lines = []
        self.runner_dir(experiment).mkdir(parents=True, exist_ok=True)
        self.write_heartbeat("start", "runner-start")
        summary: dict[str, Any] = {
            "experiment_id": experiment.experiment_id,
            "hypothesis": experiment.hypothesis,
            "config": experiment.config,
            "base_checkpoint": experiment.base_checkpoint,
            "base_checkpoint_note": experiment.base_checkpoint_note,
            "commit_sha": experiment.code_sha,
            "train_steps": experiment.train_steps,
            "reset_optimizer": experiment.reset_optimizer,
            "override_lr": experiment.override_lr,
            "started_at": utc_now(),
            "finished_at": None,
            "status": "running",
            "acceptance": {"visual_review": "pending", "visual_note": "pending manual decoded-grid review"},
            "next_step_basis": experiment.next_step_basis,
            "remote": self.remote_metadata(experiment),
        }
        try:
            self.ensure_a100_running()
            self.ssh_smoke()
            self.remote_checkout(experiment)
            self.remote_preflight(experiment)
            train_result = self.remote_train(experiment)
            checkpoint_path = str(train_result["last_checkpoint"])
            summary["train"] = train_result
            i2t = self.remote_i2t_eval(experiment, checkpoint_path)
            t2i = self.remote_t2i_eval(experiment, checkpoint_path)
            decoded = self.remote_decoded_eval(experiment, checkpoint_path)
            summary["i2t_eval"] = i2t
            summary["t2i_eval"] = t2i
            summary["decoded_eval"] = decoded
            summary["acceptance"].update(self.assess_acceptance(experiment, i2t, t2i, decoded))
            summary["status"] = "ok"
        except Exception as exc:
            summary["status"] = "failed"
            summary["failure"] = self.failure_payload(exc)
        finally:
            summary["finished_at"] = utc_now()
            paths = self.write_records(experiment, summary)
            if not self.args.no_commit and not self.args.dry_run:
                try:
                    self.commit_records(experiment, paths, summary)
                except Exception as exc:
                    summary["record_commit_error"] = str(exc)
                    paths = self.write_records(experiment, summary)
            self.write_heartbeat("finish", f"runner-finish:{summary['status']}")
        return summary

    def runner_dir(self, experiment: Experiment) -> Path:
        return self.runner_log_root / experiment.experiment_id

    def remote_metadata(self, experiment: Experiment) -> dict[str, str | int]:
        return {
            "aistation_name": self.args.aistation_name,
            "ssh_host": self.args.ssh_host,
            "ssh_port": self.args.ssh_port,
            "remote_root": self.remote_root,
            "worktree": self.remote_worktree(experiment),
        }

    def remote_worktree(self, experiment: Experiment) -> str:
        return f"{self.remote_root}/worktrees/{experiment.experiment_id}"

    def remote_project_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self.remote_root}/{path}"

    def remote_log_rel(self, experiment: Experiment, *parts: str) -> str:
        return "/".join(("logs", "remote_experiments", experiment.experiment_id, *parts))

    def remote_log_abs(self, experiment: Experiment, *parts: str) -> str:
        return f"{self.remote_worktree(experiment)}/{self.remote_log_rel(experiment, *parts)}"

    def aistation_status(self) -> dict[str, Any]:
        if not self.helper.exists():
            raise RunnerError(f"AIStation helper missing: {self.helper}", stage="aistation_status")
        result = run_local(
            ["node", str(self.helper), "status", self.args.aistation_name],
            check=False,
            timeout=self.args.aistation_timeout,
        )
        if result.returncode != 0:
            raise RunnerError(
                f"AIStation status failed: {result.stderr.strip() or result.stdout.strip()}",
                stage="aistation_status",
                exit_code=result.returncode,
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"AIStation status returned non-JSON output: {exc}", stage="aistation_status") from exc
        if not payload.get("ok"):
            raise RunnerError(f"AIStation status returned ok=false: {payload}", stage="aistation_status")
        return payload

    def aistation_target_status(self) -> str:
        payload = self.aistation_status()
        targets = payload.get("targets") or []
        if not targets:
            raise RunnerError(f"AIStation target not found: {self.args.aistation_name}", stage="aistation_status")
        return str(targets[0].get("wpStatus") or "unknown")

    def ensure_a100_running(self) -> None:
        status = self.aistation_target_status()
        if status == "Running":
            return
        if status == "Halt":
            result = run_local(
                ["node", str(self.helper), "open", self.args.aistation_name],
                check=False,
                timeout=self.args.aistation_timeout,
            )
            if result.returncode != 0:
                raise RunnerError(
                    f"AIStation open failed: {result.stderr.strip() or result.stdout.strip()}",
                    stage="aistation_open",
                    exit_code=result.returncode,
                )
        elif status not in {"Pending", "Queuing"}:
            raise RunnerError(f"AIStation status is not runnable: {status}", stage="aistation_open")

        deadline = time.time() + self.args.aistation_wait_sec
        while time.time() < deadline:
            status = self.aistation_target_status()
            if status == "Running":
                return
            self.write_heartbeat("aistation_wait", f"waiting-for-{self.args.aistation_name}:{status}")
            time.sleep(self.args.aistation_poll_sec)
        raise RunnerError(f"AIStation did not reach Running before timeout, last status={status}", stage="aistation_open")

    def expect_command(self, remote_command: str) -> list[str]:
        remote_shell_command = f"bash -lc {q(remote_command)}"
        ssh_args = [
            "ssh",
            "-p",
            str(self.args.ssh_port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            self.ssh_target,
            remote_shell_command,
        ]
        if not self.password:
            return ssh_args
        script = """
set timeout -1
set password $env(A100_SSH_PASSWORD)
log_user 1
spawn {*}$argv
expect {
    -re "(?i)are you sure you want to continue connecting" {
        send "yes\\r"
        exp_continue
    }
    -re "(?i)password:" {
        send "$password\\r"
        exp_continue
    }
    eof
}
catch wait result
exit [lindex $result 3]
"""
        return ["expect", "-f", "-", "--", *ssh_args, script]

    def run_remote_capture(self, remote_command: str, *, stage: str, timeout: int | None = None) -> CommandResult:
        reject_forbidden_command(remote_command)
        command_for_log = f"ssh {self.ssh_target}:{self.args.ssh_port} -- {remote_command}"
        if self.args.dry_run:
            print(f"[dry-run:{stage}] {remote_command}")
            return CommandResult(0, "", command_for_log)
        if self.password:
            script = self.expect_command(remote_command)
            args = script[:-1]
            stdin = script[-1]
            env = dict(os.environ)
            env["A100_SSH_PASSWORD"] = self.password
            result = subprocess.run(
                args,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        else:
            result = subprocess.run(
                self.expect_command(remote_command),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        output = (result.stdout or "") + (result.stderr or "")
        self.remember_output(output)
        if result.returncode != 0:
            raise RunnerError(output.strip() or f"remote command failed at {stage}", stage=stage, command=command_for_log, exit_code=result.returncode)
        return CommandResult(result.returncode, output, command_for_log)

    def run_remote_stream(self, experiment: Experiment, remote_command: str, *, stage: str, timeout: int | None = None) -> CommandResult:
        reject_forbidden_command(remote_command)
        command_for_log = f"ssh {self.ssh_target}:{self.args.ssh_port} -- {remote_command}"
        if self.args.dry_run:
            print(f"[dry-run:{stage}] {remote_command}")
            return CommandResult(0, "", command_for_log)

        log_path = self.runner_dir(experiment) / f"{stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.password:
            script = self.expect_command(remote_command)
            popen_args = script[:-1]
            stdin_text = script[-1]
            env = dict(os.environ)
            env["A100_SSH_PASSWORD"] = self.password
        else:
            popen_args = self.expect_command(remote_command)
            stdin_text = None
            env = None
        started = time.time()
        process = subprocess.Popen(
            popen_args,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if stdin_text is not None and process.stdin is not None:
            process.stdin.write(stdin_text)
            process.stdin.close()

        lines: list[str] = []
        def read_output() -> None:
            with log_path.open("w", encoding="utf-8") as handle:
                if process.stdout is None:
                    return
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    lines.append(line)
                    self.remember_output(line)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        with Heartbeat(self, stage, command_for_log, self.args.heartbeat_sec):
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                reader.join(timeout=5)
                raise RunnerError(f"remote command timed out after {timeout}s", stage=stage, command=command_for_log) from exc
        reader.join(timeout=5)
        output = "".join(lines)
        append_jsonl(
            log_path.with_suffix(".metadata.jsonl"),
            {
                "stage": stage,
                "command": command_for_log,
                "exit_code": process.returncode,
                "elapsed_sec": round(time.time() - started, 3),
                "finished_at": utc_now(),
            },
        )
        if exit_code != 0:
            raise RunnerError(output.strip() or f"remote command failed at {stage}", stage=stage, command=command_for_log, exit_code=exit_code)
        return CommandResult(exit_code, output, command_for_log)

    def remember_output(self, output: str) -> None:
        for line in output.splitlines():
            if line.strip():
                self.last_output_lines.append(line[-1000:])
        if len(self.last_output_lines) > self.max_tail_lines:
            self.last_output_lines = self.last_output_lines[-self.max_tail_lines :]

    def ssh_smoke(self) -> None:
        command = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        self.run_remote_capture(command, stage="ssh_gpu_smoke", timeout=30)

    def remote_checkout(self, experiment: Experiment) -> None:
        worktree = self.remote_worktree(experiment)
        fetch_ref = self.args.remote_ref or current_branch()
        script = "\n".join(
            [
                f"cd {q(self.remote_root)}",
                f"GIT_TERMINAL_PROMPT=0 timeout {int(self.args.git_fetch_timeout_sec)} git fetch origin {q(fetch_ref)}",
                f"if [ -e {q(worktree)} ]; then echo {q('worktree already exists: ' + worktree)}; exit 7; fi",
                f"git worktree add --detach {q(worktree)} {q(experiment.code_sha)}",
                f"cd {q(worktree)}",
                "if [ ! -e artifacts ]; then ln -s ../../artifacts artifacts; fi",
                "if [ ! -e .cache ] && [ -d ../../.cache ]; then ln -s ../../.cache .cache; fi",
                f'test "$(git rev-parse HEAD)" = {q(experiment.code_sha)}',
            ]
        )
        self.run_remote_stream(experiment, script, stage="checkout", timeout=600)

    def remote_preflight(self, experiment: Experiment) -> None:
        checkpoint = self.remote_project_path(experiment.base_checkpoint)
        script = "\n".join(
            [
                f"cd {q(self.remote_worktree(experiment))}",
                "test -x ../../.venv/bin/python",
                f"test -f {q(checkpoint)}",
                "test -d ../../data",
                "test -d artifacts/tokenized",
                f"{offline_prefix(self.remote_root)} ../../.venv/bin/python - <<'PY'",
                "import os",
                "for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):",
                "    assert os.environ.get(key) == '1', key",
                "import torch, yaml",
                "print('preflight-ok', torch.__version__)",
                "PY",
            ]
        )
        self.run_remote_stream(experiment, script, stage="preflight", timeout=120)

    def remote_train(self, experiment: Experiment) -> dict[str, Any]:
        checkpoint = self.remote_project_path(experiment.base_checkpoint)
        out_dir = self.remote_log_rel(experiment, "train")
        command = [
            "../../.venv/bin/python",
            "scripts/continue_stage2.py",
            "--config",
            experiment.config,
            "--checkpoint-path",
            checkpoint,
            "--extra-steps",
            str(experiment.train_steps),
            "--out-dir",
            out_dir,
        ]
        if experiment.reset_optimizer:
            command.append("--reset-optimizer")
        if experiment.override_lr is not None:
            command.extend(["--override-lr", str(experiment.override_lr)])
        script = "\n".join(
            [
                f"cd {q(self.remote_worktree(experiment))}",
                f"mkdir -p {q(out_dir)}",
                f"{offline_prefix(self.remote_root)} {' '.join(q(part) for part in command)}",
            ]
        )
        result = self.run_remote_stream(experiment, script, stage="train", timeout=self.args.train_timeout_sec)
        if self.args.dry_run:
            return {
                "run_dir": f"{self.remote_worktree(experiment)}/runs/<dry-run>",
                "log_dir": f"{self.remote_worktree(experiment)}/{out_dir}",
                "last_checkpoint": f"{self.remote_worktree(experiment)}/runs/<dry-run>/checkpoints/stage2_step<dry-run>.pt",
                "final_step": "dry-run",
                "command": result.command,
            }
        payload = extract_last_json_object(result.output)
        payload["command"] = result.command
        return payload

    def remote_i2t_eval(self, experiment: Experiment, checkpoint_path: str) -> dict[str, Any]:
        out_rel = self.remote_log_rel(experiment, "eval", "i2t_512.json")
        command = [
            "../../.venv/bin/python",
            "scripts/eval_image_to_text_tokens.py",
            "--config",
            experiment.config,
            "--checkpoint-path",
            checkpoint_path,
            "--out",
            out_rel,
            "--sampling-steps",
            str(experiment.eval["sampling_steps"]),
            "--temperature",
            str(experiment.eval["temperature"]),
            "--max-samples",
            str(experiment.eval["max_samples"]),
        ]
        script = "\n".join(
            [
                f"cd {q(self.remote_worktree(experiment))}",
                f"mkdir -p {q(self.remote_log_rel(experiment, 'eval'))}",
                f"{offline_prefix(self.remote_root)} {' '.join(q(part) for part in command)}",
            ]
        )
        self.run_remote_stream(experiment, script, stage="eval_i2t", timeout=self.args.eval_timeout_sec)
        if self.args.dry_run:
            return {"remote_path": f"{self.remote_worktree(experiment)}/{out_rel}", "dry_run": True}
        payload = self.remote_read_json(f"{self.remote_worktree(experiment)}/{out_rel}", stage="read_i2t")
        payload["remote_path"] = f"{self.remote_worktree(experiment)}/{out_rel}"
        return payload

    def remote_t2i_eval(self, experiment: Experiment, checkpoint_path: str) -> dict[str, Any]:
        out_rel = self.remote_log_rel(experiment, "eval", "t2i_512.json")
        command = [
            "../../.venv/bin/python",
            "scripts/eval_text_to_image_tokens.py",
            "--config",
            experiment.config,
            "--checkpoint-path",
            checkpoint_path,
            "--out",
            out_rel,
            "--sampling-steps",
            str(experiment.eval["sampling_steps"]),
            "--temperature",
            str(experiment.eval["temperature"]),
            "--max-samples",
            str(experiment.eval["max_samples"]),
            "--bank-samples",
            str(experiment.eval["bank_samples"]),
        ]
        script = "\n".join(
            [
                f"cd {q(self.remote_worktree(experiment))}",
                f"mkdir -p {q(self.remote_log_rel(experiment, 'eval'))}",
                f"{offline_prefix(self.remote_root)} {' '.join(q(part) for part in command)}",
            ]
        )
        self.run_remote_stream(experiment, script, stage="eval_t2i", timeout=self.args.eval_timeout_sec)
        if self.args.dry_run:
            return {"remote_path": f"{self.remote_worktree(experiment)}/{out_rel}", "dry_run": True}
        payload = self.remote_read_json(f"{self.remote_worktree(experiment)}/{out_rel}", stage="read_t2i")
        payload["remote_path"] = f"{self.remote_worktree(experiment)}/{out_rel}"
        return payload

    def remote_decoded_eval(self, experiment: Experiment, checkpoint_path: str) -> dict[str, Any]:
        out_rel = self.remote_log_rel(experiment, "decoded")
        command = [
            "../../.venv/bin/python",
            "scripts/sweep_text_to_image_decoded.py",
            "--config",
            experiment.config,
            "--checkpoint-path",
            checkpoint_path,
            "--out-dir",
            out_rel,
            "--steps",
            str(experiment.eval["sampling_steps"]),
            "--temperatures",
            str(experiment.eval["temperature"]),
            "--repeats-per-label",
            str(experiment.eval["decoded_repeats_per_label"]),
            "--bank-samples",
            str(experiment.eval["bank_samples"]),
        ]
        script = "\n".join(
            [
                f"cd {q(self.remote_worktree(experiment))}",
                f"mkdir -p {q(out_rel)}",
                f"{offline_prefix(self.remote_root)} {' '.join(q(part) for part in command)}",
            ]
        )
        self.run_remote_stream(experiment, script, stage="eval_decoded", timeout=self.args.decoded_timeout_sec)
        if self.args.dry_run:
            return {"remote_path": f"{self.remote_worktree(experiment)}/{out_rel}/summary.json", "dry_run": True}
        payload = self.remote_read_json(f"{self.remote_worktree(experiment)}/{out_rel}/summary.json", stage="read_decoded")
        payload["remote_path"] = f"{self.remote_worktree(experiment)}/{out_rel}/summary.json"
        return payload

    def remote_read_json(self, remote_path: str, *, stage: str) -> dict[str, Any]:
        result = self.run_remote_capture(f"cat {q(remote_path)}", stage=stage, timeout=60)
        try:
            return json.loads(result.output)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"failed to parse remote JSON {remote_path}: {exc}", stage=stage) from exc

    def write_heartbeat(self, phase: str, command: str) -> None:
        if self.args.dry_run:
            return
        record = {
            "timestamp": utc_now(),
            "local_timestamp": local_now(),
            "phase": phase,
            "aistation_status": "unknown",
            "gpu": "unknown",
            "command": command,
            "recent_output": self.last_output_lines[-20:],
        }
        try:
            record["aistation_status"] = self.aistation_target_status()
        except Exception as exc:
            record["aistation_status"] = f"error: {exc}"
        try:
            result = self.run_remote_capture(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
                stage="heartbeat_gpu",
                timeout=15,
            )
            record["gpu"] = result.output.strip()
        except Exception as exc:
            record["gpu"] = f"error: {exc}"
        heartbeat_path = self.runner_log_root / "heartbeat.jsonl"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(heartbeat_path, record)

    def failure_payload(self, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, RunnerError):
            return {
                "stage": exc.stage or "unknown",
                "reason": str(exc),
                "command": exc.command,
                "exit_code": exc.exit_code,
                "last_output": self.last_output_lines[-40:],
            }
        return {
            "stage": "unknown",
            "reason": repr(exc),
            "command": None,
            "exit_code": None,
            "last_output": self.last_output_lines[-40:],
        }

    def assess_acceptance(self, experiment: Experiment, i2t: dict[str, Any], t2i: dict[str, Any], decoded: dict[str, Any]) -> dict[str, Any]:
        i2t_exact = float(i2t.get("image_to_text_exact_match", 0.0))
        decoded_combo = (decoded.get("combos") or [{}])[0]
        t2i_counts = label_counts_dict(t2i.get("generated_label_top20", []))
        decoded_counts = label_counts_dict(decoded_combo.get("nearest_label_counts", []))
        collapse_labels = [int(value) for value in experiment.acceptance.get("collapse_labels", [9, 1, 7])]
        t2i_collapse_mass = sum(t2i_counts.get(label, 0) for label in collapse_labels)
        decoded_collapse_mass = sum(decoded_counts.get(label, 0) for label in collapse_labels)
        t2i_total = int(t2i.get("total", 0) or 0)
        decoded_total = int(decoded_combo.get("total", 0) or 0)
        auto_pass = i2t_exact >= float(experiment.acceptance["i2t_exact_min"])
        return {
            "i2t_exact_min": experiment.acceptance["i2t_exact_min"],
            "i2t_exact_pass": auto_pass,
            "t2i_nearest_label_distribution": t2i_counts,
            "decoded_nearest_label_distribution": decoded_counts,
            "t2i_collapse_labels": collapse_labels,
            "t2i_collapse_mass": t2i_collapse_mass,
            "t2i_collapse_fraction": None if t2i_total == 0 else t2i_collapse_mass / t2i_total,
            "decoded_collapse_mass": decoded_collapse_mass,
            "decoded_collapse_fraction": None if decoded_total == 0 else decoded_collapse_mass / decoded_total,
            "decoded_grid_path": decoded_combo.get("decoded_grid"),
            "final_acceptance": "pending_visual_review",
            "acceptance_note": "Token metrics are only a first pass; final pass requires manual decoded-grid visual note.",
        }

    def write_records(self, experiment: Experiment, summary: dict[str, Any]) -> list[Path]:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.record_dir / f"{experiment.experiment_id}.json"
        md_path = self.record_dir / f"{experiment.experiment_id}.md"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        md_path.write_text(self.render_markdown(summary), encoding="utf-8")
        return [json_path, md_path]

    def render_markdown(self, summary: dict[str, Any]) -> str:
        i2t = summary.get("i2t_eval") or {}
        t2i = summary.get("t2i_eval") or {}
        decoded = summary.get("decoded_eval") or {}
        decoded_combo = (decoded.get("combos") or [{}])[0]
        acceptance = summary.get("acceptance") or {}
        failure = summary.get("failure") or {}
        lines = [
            f"# {summary['experiment_id']}",
            "",
            f"- Status: `{summary['status']}`",
            f"- Started: `{summary['started_at']}`",
            f"- Finished: `{summary.get('finished_at')}`",
            f"- Commit SHA: `{summary['commit_sha']}`",
            f"- Config: `{summary['config']}`",
            f"- Base checkpoint: `{summary['base_checkpoint']}`",
            f"- Train steps: `{summary['train_steps']}`",
            f"- Reset optimizer: `{summary['reset_optimizer']}`",
            f"- Override LR: `{summary['override_lr']}`",
            "",
            "## Hypothesis",
            "",
            summary["hypothesis"],
            "",
            "## Metrics",
            "",
            f"- i2t 512 exact: `{i2t.get('image_to_text_exact_match', 'n/a')}`",
            f"- i2t 512 token acc: `{i2t.get('image_to_text_token_accuracy', 'n/a')}`",
            f"- t2i 512 token-NN acc: `{t2i.get('text_to_image_token_nn_accuracy', 'n/a')}`",
            f"- t2i nearest-label distribution: `{acceptance.get('t2i_nearest_label_distribution', 'n/a')}`",
            f"- decoded 160 token-NN acc: `{decoded_combo.get('token_nn_accuracy', 'n/a')}`",
            f"- decoded nearest-label distribution: `{acceptance.get('decoded_nearest_label_distribution', 'n/a')}`",
            f"- decoded grid: `{acceptance.get('decoded_grid_path', 'n/a')}`",
            "",
            "## Acceptance",
            "",
            f"- i2t exact pass: `{acceptance.get('i2t_exact_pass', 'n/a')}`",
            f"- visual review: `{acceptance.get('visual_review', 'pending')}`",
            f"- final acceptance: `{acceptance.get('final_acceptance', 'n/a')}`",
            f"- note: {acceptance.get('acceptance_note', 'n/a')}",
            "",
            "## Next Step Basis",
            "",
            summary.get("next_step_basis") or "n/a",
            "",
        ]
        if summary.get("status") == "failed":
            lines.extend(
                [
                    "## Failure",
                    "",
                    f"- Stage: `{failure.get('stage')}`",
                    f"- Exit code: `{failure.get('exit_code')}`",
                    f"- Reason: {failure.get('reason')}",
                    f"- Command: `{failure.get('command')}`",
                    "",
                    "Recent output:",
                    "",
                    "```text",
                    "\n".join(str(line) for line in failure.get("last_output", [])[-40:]),
                    "```",
                    "",
                ]
            )
        return "\n".join(lines)

    def commit_records(self, experiment: Experiment, paths: list[Path], summary: dict[str, Any]) -> None:
        rel_paths = [str(path.relative_to(REPO_ROOT)) for path in paths]
        run_local(["git", "add", *rel_paths])
        diff = run_local(["git", "diff", "--cached", "--quiet"], check=False)
        if diff.returncode == 0:
            return
        message = f"Record remote experiment {experiment.experiment_id}"
        body = "\n".join(
            [
                f"Timestamp UTC: {utc_now()}",
                f"Experiment: {experiment.experiment_id}",
                f"Status: {summary.get('status')}",
                f"Hypothesis: {experiment.hypothesis}",
                "Lesson: keep experiment orchestration commit-SHA based and record failures instead of changing paradigm.",
            ]
        )
        run_local(["git", "commit", "-m", message, "-m", body])
        branch = current_branch()
        run_local(["git", "push", "origin", branch])


def label_counts_dict(raw_counts: Any) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in raw_counts or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            counts[int(item[0])] = int(item[1])
    return counts


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dry_run(experiments: list[Experiment], args: argparse.Namespace) -> None:
    runner = RemoteExperimentRunner(args=args, password=None)
    for experiment in experiments:
        print(f"experiment_id: {experiment.experiment_id}")
        print(f"resolved_code_sha: {experiment.code_sha}")
        runner.remote_checkout(experiment)
        runner.remote_preflight(experiment)
        runner.remote_train(experiment)
        runner.remote_i2t_eval(experiment, "<trained_checkpoint>")
        runner.remote_t2i_eval(experiment, "<trained_checkpoint>")
        runner.remote_decoded_eval(experiment, "<trained_checkpoint>")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run commit-SHA based uniindex experiments on AIStation A100.")
    parser.add_argument("experiments", nargs="*", help="Experiment YAML files.")
    parser.add_argument("--code-sha", default=None, help="Override YAML code_sha for all experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Validate YAML and print remote commands only.")
    parser.add_argument("--status-only", action="store_true", help="Read-only AIStation status check.")
    parser.add_argument("--no-commit", action="store_true", help="Write records but do not commit or push them.")
    parser.add_argument("--aistation-helper", default=str(DEFAULT_AISTATION_HELPER))
    parser.add_argument("--aistation-name", default=DEFAULT_AISTATION_NAME)
    parser.add_argument("--aistation-timeout", type=int, default=60)
    parser.add_argument("--aistation-wait-sec", type=int, default=1800)
    parser.add_argument("--aistation-poll-sec", type=int, default=20)
    parser.add_argument("--remote-ref", default=None, help="Remote git ref to fetch before detached checkout. Defaults to the local branch.")
    parser.add_argument("--git-fetch-timeout-sec", type=int, default=300)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    parser.add_argument("--train-timeout-sec", type=int, default=7200)
    parser.add_argument("--eval-timeout-sec", type=int, default=7200)
    parser.add_argument("--decoded-timeout-sec", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    runner = RemoteExperimentRunner(args=args, password=None)
    if args.status_only:
        runner.run_status_only()
        return 0

    if not args.experiments:
        raise RunnerError("pass at least one experiment YAML, or use --status-only")
    experiments = [load_experiment(Path(path), code_sha_override=args.code_sha) for path in args.experiments]
    if args.dry_run:
        dry_run(experiments, args)
        return 0

    password = os.environ.get("A100_SSH_PASSWORD")
    if password is None and sys.stdin.isatty():
        password = getpass.getpass(f"SSH password for {args.ssh_user}@{args.ssh_host}:{args.ssh_port} (empty to try key auth): ")
        if password == "":
            password = None
    runner = RemoteExperimentRunner(args=args, password=password)

    any_failed = False
    for experiment in experiments:
        summary = runner.run_experiment(experiment)
        print(json.dumps({"experiment_id": experiment.experiment_id, "status": summary["status"]}, indent=2, ensure_ascii=False))
        any_failed = any_failed or summary["status"] != "ok"
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
