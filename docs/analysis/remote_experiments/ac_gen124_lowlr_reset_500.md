# ac_gen124_lowlr_reset_500

- Status: `failed`
- Started: `2026-05-06T07:02:01Z`
- Finished: `2026-05-06T07:02:05Z`
- Commit SHA: `c6e9fe16a6df03aea38c5748eefbf94947b349d9`
- Config: `configs/flm_words_joint_long_rope_p8_ac_gen124_lowlr_reset_gpu80.yaml`
- Base checkpoint: `runs/20260506T033603Z-stage2-continue/checkpoints/stage2_step005000.pt`
- Train steps: `500`
- Reset optimizer: `True`
- Override LR: `0.0001`

## Hypothesis

step 6500 generation regression may come from high LR or inherited optimizer momentum pushing the model toward a 9 attractor; low LR plus reset optimizer should preserve gen124 prompt-following while reducing drift.

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

If i2t stays >=80% and t2i does not regress, extend to 1000 steps before broader scale-up.

## Failure

- Stage: `checkout`
- Exit code: `1`
- Reason: spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc cd /fangxueji/Projects/PG/uniindex
git fetch origin
if [ -e /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500 ]; then echo 'worktree already exists: /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500'; exit 7; fi
git worktree add --detach /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500 c6e9fe16a6df03aea38c5748eefbf94947b349d9
cd /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500
test "$(git rev-parse HEAD)" = c6e9fe16a6df03aea38c5748eefbf94947b349d9
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.


root@172.16.78.10's password: 
fatal: not a git repository (or any of the parent directories): .git
fatal: not a git repository (or any of the parent directories): .git
bash: line 4: cd: /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500: No such file or directory
fatal: not a git repository (or any of the parent directories): .git
- Command: `ssh root@172.16.78.10:30186 -- cd /fangxueji/Projects/PG/uniindex
git fetch origin
if [ -e /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500 ]; then echo 'worktree already exists: /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500'; exit 7; fi
git worktree add --detach /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500 c6e9fe16a6df03aea38c5748eefbf94947b349d9
cd /fangxueji/Projects/PG/uniindex/worktrees/ac_gen124_lowlr_reset_500
test "$(git rev-parse HEAD)" = c6e9fe16a6df03aea38c5748eefbf94947b349d9`

Recent output:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          Off |   00000000:8A:00.0 Off |                    0 |
| N/A   27C    P0             89W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
spawn ssh -p 30186 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@172.16.78.10 bash -lc nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
Warning: Permanently added '[172.16.78.10]:30186' (ED25519) to the list of known hosts.
root@172.16.78.10's password: 
Wed May  6 14:58:16 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          Off |   00000000:8A:00.0 Off |                    0 |
| N/A   28C    P0             89W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
