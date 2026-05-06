# ac_t2i_noisy_image_power2_500

- Status: `failed`
- Started: `2026-05-06T07:23:47Z`
- Finished: `2026-05-06T07:24:00Z`
- Commit SHA: `0f3d9d5cad84869afee8a26b6f45f37cfae542cc`
- Config: `configs/flm_words_joint_long_rope_p8_ac_t2i_noisy_image_power2_gpu80.yaml`
- Base checkpoint: `runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `True`
- Override LR: `0.0001`

## Hypothesis

Collapse may start in early denoising from high-noise image states; increasing the t2i image time-power during training may make text conditioning matter earlier.

## Metrics

- i2t 512 exact: `n/a`
- i2t 512 token acc: `n/a`
- t2i 512 token-NN acc: `n/a`
- t2i nearest-label distribution: `n/a`
- decoded 160 token-NN acc: `n/a`
- decoded nearest-label distribution: `n/a`
- decoded grid: `n/a`

## Acceptance

- i2t exact pass: `n/a`
- visual review: `pending`
- final acceptance: `n/a`
- note: n/a

## Next Step Basis

If decoded images follow prompts better and nearest labels spread beyond 9/1/7, run a small power sweep next.

## Failure

- Stage: `train`
- Exit code: `1`
- Reason: spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc 'cd /fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500
mkdir -p logs/remote_experiments/ac_t2i_noisy_image_power2_500/train
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/fangxueji/Projects/PG/uniindex/.cache/huggingface TRANSFORMERS_CACHE=/fangxueji/Projects/PG/uniindex/.cache/huggingface/transformers ../../.venv/bin/python scripts/continue_stage2.py --config configs/flm_words_joint_long_rope_p8_ac_t2i_noisy_image_power2_gpu80.yaml --checkpoint-path /fangxueji/Projects/PG/uniindex/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt --extra-steps 500 --out-dir logs/remote_experiments/ac_t2i_noisy_image_power2_500/train --reset-optimizer --override-lr 0.0001'
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.


root@172.16.78.10's password: 
Traceback (most recent call last):
  File "/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/scripts/continue_stage2.py", line 227, in <module>
    raise SystemExit(main())
  File "/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/scripts/continue_stage2.py", line 77, in main
    tokenizer_state = load_tokenizer_state(config)
  File "/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/src/uniindex/data.py", line 230, in load_tokenizer_state
    return torch.load(tokenizer_state_path(config), map_location="cpu")
  File "/fangxueji/Projects/PG/uniindex/.venv/lib/python3.10/site-packages/torch/serialization.py", line 1425, in load
    with _open_file_like(f, "rb") as opened_file:
  File "/fangxueji/Projects/PG/uniindex/.venv/lib/python3.10/site-packages/torch/serialization.py", line 751, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/fangxueji/Projects/PG/uniindex/.venv/lib/python3.10/site-packages/torch/serialization.py", line 732, in __init__
    super().__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: '/fangxueji/Projects/PG/uniindex/artifacts/tokenized/mnist-emu3p5-baai-emu3-5-visiontokenizer-img256-train30000-test5000-compact1-text483fcede53/tokenizer_state.pt'
- Command: `ssh root@172.16.78.10:30186 -- cd /fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500
mkdir -p logs/remote_experiments/ac_t2i_noisy_image_power2_500/train
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/fangxueji/Projects/PG/uniindex/.cache/huggingface TRANSFORMERS_CACHE=/fangxueji/Projects/PG/uniindex/.cache/huggingface/transformers ../../.venv/bin/python scripts/continue_stage2.py --config configs/flm_words_joint_long_rope_p8_ac_t2i_noisy_image_power2_gpu80.yaml --checkpoint-path /fangxueji/Projects/PG/uniindex/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt --extra-steps 500 --out-dir logs/remote_experiments/ac_t2i_noisy_image_power2_500/train --reset-optimizer --override-lr 0.0001`

Recent output:

```text
print('"'"'preflight-ok'"'"', torch.__version__)
PY'
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.
root@172.16.78.10's password: 
spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.
root@172.16.78.10's password: 
0, 0, 81920
preflight-ok 2.6.0+cu124
spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.
root@172.16.78.10's password: 
0, 0, 81920
spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc 'cd /fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500
mkdir -p logs/remote_experiments/ac_t2i_noisy_image_power2_500/train
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/fangxueji/Projects/PG/uniindex/.cache/huggingface TRANSFORMERS_CACHE=/fangxueji/Projects/PG/uniindex/.cache/huggingface/transformers ../../.venv/bin/python scripts/continue_stage2.py --config configs/flm_words_joint_long_rope_p8_ac_t2i_noisy_image_power2_gpu80.yaml --checkpoint-path /fangxueji/Projects/PG/uniindex/runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt --extra-steps 500 --out-dir logs/remote_experiments/ac_t2i_noisy_image_power2_500/train --reset-optimizer --override-lr 0.0001'
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.
root@172.16.78.10's password: 
spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.
root@172.16.78.10's password: 
0, 0, 81920
Traceback (most recent call last):
  File "/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/scripts/continue_stage2.py", line 227, in <module>
    raise SystemExit(main())
  File "/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/scripts/continue_stage2.py", line 77, in main
    tokenizer_state = load_tokenizer_state(config)
  File "/fangxueji/Projects/PG/uniindex/worktrees/ac_t2i_noisy_image_power2_500/src/uniindex/data.py", line 230, in load_tokenizer_state
    return torch.load(tokenizer_state_path(config), map_location="cpu")
  File "/fangxueji/Projects/PG/uniindex/.venv/lib/python3.10/site-packages/torch/serialization.py", line 1425, in load
    with _open_file_like(f, "rb") as opened_file:
  File "/fangxueji/Projects/PG/uniindex/.venv/lib/python3.10/site-packages/torch/serialization.py", line 751, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/fangxueji/Projects/PG/uniindex/.venv/lib/python3.10/site-packages/torch/serialization.py", line 732, in __init__
    super().__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: '/fangxueji/Projects/PG/uniindex/artifacts/tokenized/mnist-emu3p5-baai-emu3-5-visiontokenizer-img256-train30000-test5000-compact1-text483fcede53/tokenizer_state.pt'
spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.
root@172.16.78.10's password: 
0, 0, 81920
```
