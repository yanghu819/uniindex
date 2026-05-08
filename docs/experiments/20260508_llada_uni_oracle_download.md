# 2026-05-08 LLaDA-Uni Official Oracle Download

Timestamp: 2026-05-08T07:36:00Z

Code SHAs:

- `59a3027a055e4e3d70e851f521e712f3deaba0dc`: initial official LLaDA-Uni oracle runner.
- `a3e81051b69100441d75d509835aa73cff17fb23`: force official HF downloads into repo-local cache.
- `ffd1677ac57748733ceedbd9824ee38217fafade`: pre-download via resumable `snapshot_download(max_workers=1)`.
- `7645f8ea17e15d0256eca9c630b0175f23f74b51`: record the first download blocker report.

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
- The official `inclusionAI/LLaDA2.0-Uni` snapshot metadata shows about `60.28GB` total:
  - `decoder-turbo/decoder_model.safetensors`: `12,321,673,696` bytes.
  - `image_tokenizer/image_tokenizer.safetensors`: `2,398,968,416` bytes.
  - `vae/diffusion_pytorch_model.safetensors`: `167,666,902` bytes.
  - main FLM shards include five `5.37GB` shards plus several smaller shards.
- The official `inclusionAI/LLaDA2.0-Uni-FP8` snapshot exists and is about `32.67GB`; it is a possible artifact-size fallback, but was not used as the baseline oracle without explicit approval.
- `decoder-turbo/decoder_model.safetensors` was already complete in the A100 repo-local cache.
- `image_tokenizer/image_tokenizer.safetensors` was staged through local ModelScope download plus parallel part upload to A100, then assembled and sha256-verified:
  - remote bytes: `2,398,968,416`
  - sha256: `e0a11a82ad221ac1f3b917abfce31ffaaec3571200ae7ee5318a223ff2eedc49`
- `vae/diffusion_pytorch_model.safetensors` was staged through local ModelScope download plus upload to A100 and sha256-verified:
  - remote bytes: `167,666,902`
  - sha256: `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`

What failed:

- Direct `huggingface.co` repeatedly failed on SSL/connect timeout.
- `hf-mirror.com` succeeded for metadata and small files but large shards were too slow/unstable.
- Xet/CAS path repeatedly hit `ReadTimeout` and remote disconnects.
- No-Xet mirror path progressed, but measured range download speed was roughly `735,664 bytes / 12s`, too slow for a 63GB model inside the A100 session.
- A100 direct ModelScope range download was also slow for large files, about hundreds of KB/s for the image tokenizer.
- Single-stream local-to-A100 upload was also slow; parallel part upload was materially better and completed the 2.4GB tokenizer transfer in about 25 minutes.

Current evidence:

- Completed model shards in snapshot at the last check: `00007`, `00009`, `00010`, `00011`, `00012`.
- Missing large shards: `00001`-`00006`, `00008`, `00013`.
- Cache size reached about `20G`, including prior tokenizer/decoder assets and partial model shards.
- Decoder-side official assets now present on A100:
  - `decoder-turbo/decoder_model.safetensors`
  - `image_tokenizer/image_tokenizer.safetensors`
  - `vae/diffusion_pytorch_model.safetensors`
- GPU stayed idle because execution never reached model load or generation.

Conclusion:

- The official oracle runner is ready, and the official decoder-side assets are now staged, but the blocker remains acquisition of the main LLaDA-Uni FLM shards.
- This supports the current research hypothesis: the next useful direction is a SigLIP-VQ semantic-token FLM paired with a learned diffusion decoder, but the full official oracle still needs main-model weights before we can compare official prompt-following.
- To finish the oracle check, we need one of:
  - pre-stage the full `inclusionAI/LLaDA2.0-Uni` snapshot into repo-local cache from a faster network;
  - keep resuming downloads across multiple A100 sessions;
  - use a smaller official-compatible model if one exists and is approved.

Lesson:

- For large official oracle checks, separate artifact acquisition from inference and control download concurrency explicitly.
- Do not start model/algorithm changes before confirming the official oracle output.
- When A100 outbound is slow, a practical artifact path is: local fast download from ModelScope, split into parts, parallel upload to `/fangxueji/Projects/PG/uniindex/.cache`, assemble remotely, then verify sha256.
- Keep model assets out of GitHub; only record paths, sizes, hashes, and blockers.
