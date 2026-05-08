# 2026-05-08 LLaDA-Uni Official Oracle Download

Timestamp: 2026-05-08T07:36:00Z

Code SHAs:

- `59a3027a055e4e3d70e851f521e712f3deaba0dc`: initial official LLaDA-Uni oracle runner.
- `a3e81051b69100441d75d509835aa73cff17fb23`: force official HF downloads into repo-local cache.
- `ffd1677ac57748733ceedbd9824ee38217fafade`: pre-download via resumable `snapshot_download(max_workers=1)`.

Goal:

- Run the official LLaDA-Uni oracle path on A100:
  `official generate_image -> SigLIP-VQ image tokens -> decoder-turbo -> VAE -> image`.
- Use it as a reference for whether our SigLIP-VQ failure is a sampler/token-manifold issue rather than a decoder issue.

Remote:

- A100 `root@172.16.78.10 -p 30186`
- Worktree: `/fangxueji/Projects/PG/uniindex/worktrees/llada-oracle-ffd1677`
- Cache root: `/fangxueji/Projects/PG/uniindex/.cache`

What worked:

- A100 was opened through AIStation API and SSH/GPU probe passed.
- Official oracle code was added and unit-tested locally.
- Repo-local cache routing was verified; downloads went under `/fangxueji/Projects/PG/uniindex/.cache/huggingface`, not `/root/.cache`.
- Official remote-code files and part of the HF snapshot downloaded.
- The model index shows total LLaDA-Uni model weight size is `63,247,146,496` bytes across 13 shards.

What failed:

- Direct `huggingface.co` repeatedly failed on SSL/connect timeout.
- `hf-mirror.com` succeeded for metadata and small files but large shards were too slow/unstable.
- Xet/CAS path repeatedly hit `ReadTimeout` and remote disconnects.
- No-Xet mirror path progressed, but measured range download speed was roughly `735,664 bytes / 12s`, too slow for a 63GB model inside the A100 session.

Current evidence:

- Completed model shards in snapshot at the last check: `00007`, `00009`, `00010`, `00011`, `00012`.
- Missing large shards: `00001`-`00006`, `00008`, `00013`.
- Cache size reached about `20G`, including prior tokenizer/decoder assets and partial model shards.
- GPU stayed idle because execution never reached model load or generation.

Conclusion:

- The official oracle runner is ready, but the blocker is artifact acquisition, not FLM code or decoder code.
- To finish the oracle check, we need one of:
  - pre-stage the full `inclusionAI/LLaDA2.0-Uni` snapshot into repo-local cache from a faster network;
  - keep resuming downloads across multiple A100 sessions;
  - use a smaller official-compatible model if one exists and is approved.

Lesson:

- For large official oracle checks, separate artifact acquisition from inference and control download concurrency explicitly.
- Do not start model/algorithm changes before confirming the official oracle output.
