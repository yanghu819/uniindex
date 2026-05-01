# Aistation Spec

## 1. Constants
- `GitHub` 是唯一真源。
- 仓库地址固定为 `${REPO_OWNER}/${REPO_NAME}`；不存在时先创建。
- 本项目固定：
  - `REPO_OWNER=yanghu819`
  - `REPO_NAME=uniindex`
- 持久化根目录固定为 `/fangxueji/Projects/PG/${REPO_NAME}`。
- 远端只用 `gpu0`。
- 所有敏感信息只允许通过环境变量注入，禁止写入仓库。
- `GITHUB_TOKEN` 只允许在当前 shell/session 环境变量中临时注入，禁止写入脚本、配置、日志、metadata 或提交说明。

## 2. Root Directory Rule
- 代码、数据、模型、缓存、日志、结果、checkpoint、临时文件，全部放在 `/fangxueji/Projects/PG/${REPO_NAME}` 下。
- 禁止使用 `~`、`/tmp`、系统默认 cache 目录保存任何需要跨重启保留的内容。
- 特别禁止把持久化内容写到 `/root/.cache`、`~/.cache`、系统临时目录或任何 4 小时重置路径。
- `.venv/`、`.cache/`、`artifacts/`、`models/`、`runs/`、`logs/` 必须全部位于 `/fangxueji/Projects/PG/${REPO_NAME}` 下。
- 必须显式设置并固定这些目录到仓库根目录内：
  - `UV_CACHE_DIR`
  - `UV_PYTHON_INSTALL_DIR`
  - `PIP_CACHE_DIR`
  - `HF_HOME`
  - `HF_HUB_CACHE`
  - `TRANSFORMERS_CACHE`
  - `TORCH_HOME`
  - `XDG_CACHE_HOME`

## 3. Git and Worktree Rule
- `main` 是冻结分支，禁止直接修改。
- 所有修改默认在新 worktree 中完成：
  - `git worktree add ../<task_name> -b <task_name>`
- 任何 GPU 运行前，本地改动必须先 `commit + push` 到 GitHub。
- 每次提交必须写详细描述，至少包含：
  - 本地时间戳和时区
  - 做了什么
  - 为什么这样做
  - 如何验证
  - 经验、教训、下次可复用做法
- 远端只允许按 commit SHA 同步代码：
  - `git fetch origin`
  - `git checkout --detach <commit_sha>`
- 禁止在远端执行 `git pull`。
- 禁止使用 `rsync`、`scp` 直接传代码或数据。

## 4. Dependency Rule
- 使用 `uv` 管理 Python 环境和依赖。
- Python 依赖必须固定版本，并提交 `uv.lock`。
- 远端只安装运行所需依赖；开发依赖只在需要时安装。
- 大型依赖不 vendor 到仓库正文；使用固定版本，并通过 `uv` / `uv pip install` 安装。
- 难下载的安装包必须先在本地下载为 `whl`，再上传到 GitHub Release 或等效稳定存储，并附带 `manifest.sha256`。
- 远端安装优先使用 wheelhouse：
  - `uv pip install --no-index --find-links <wheelhouse_dir> ...`
  - 不依赖不稳定公网源重复安装大包。

## 5. Data and Model Delivery Rule
- 难下载的数据、模型、安装包，先在本地下载，再上传到 GitHub Release 或等效稳定存储。
- 上传物必须带 `sha256` 或 `manifest.sha256`。
- 远端优先从 GitHub 上的稳定资产拉取，不依赖临时镜像。
- 大模型、数据、缓存、日志、checkpoint 默认不提交到 git 历史；默认保存在 `/fangxueji/Projects/PG/${REPO_NAME}`。
- 如必须放到 GitHub，优先使用 Release 资产或 Git LFS，并记录校验信息。

## 6. Script Contract
- `setup.sh`
  - 只负责系统依赖、`uv`、Python 环境和 Python 依赖。
  - 不下载大模型。
  - 优先复用已有环境；缺失时再修复。
- `down.sh`
  - 只负责下载数据、模型、额外仓库和缓存。
  - 额外实验仓库必须固定 commit。
- `run.sh`
  - 一键复现训练、评测、可视化等流程。
  - 必须支持 `smoke` 和正式运行模式。

## 7. Smoke Rule
- 每次新环境、新依赖方案或新脚本修改后，必须先跑最低成本 smoke。
- smoke 必须验证：
  - `setup.sh / down.sh / run.sh` 全链路可执行
  - 最小数据路径可用
  - 日志可写
  - 结果可落盘
  - 退出状态正确

## 8. Formal Run Rule
- 正式运行前必须确认：
  - 代码已 `commit + push`
  - 远端已按 commit SHA checkout
  - 只使用 `gpu0`
  - 所有 cache 和输出目录都在仓库根目录内
- 长跑训练必须默认开启 checkpoint 与 resume。
- 机器重启后，优先 resume，禁止默认从头重跑。

## 9. Artifact and Metadata Rule
- 每次运行都必须记录 metadata，至少包含：
  - `commit_sha`
  - `run_mode`
  - `config`
  - `started_at`
  - `finished_at`
  - `exit_status`
  - `cost_estimate`
  - `gpu_type`
  - `gpu_index`
- 日志、结果、checkpoint 必须保存在仓库根目录下的稳定路径中，并能映射回具体 `commit SHA`。
- 产物目录命名必须稳定、可解析、可审计。
- 任何长跑、远端会话或机器终止前，必须先把 metadata、关键日志摘要、经验教训落盘到仓库根目录内的稳定路径。
- 可提交的小型文本产物必须 `commit + push` 到 GitHub；大模型、数据、checkpoint 不进 git，只记录路径、校验和、生成方式。

## 10. Default Storage Layout
- 建议固定以下目录：
  - `.cache/`
  - `data/`
  - `artifacts/`
  - `models/`
  - `runs/`
  - `logs/`
  - `third_party/`
  - `wheelhouse/`

## 11. Execution Principle
- 优先做最稳、最可复用的方案，而不是每次现场修环境。
- 对于会反复重启的机器，默认策略是：
  - 本地先下载
  - GitHub 保存稳定资产
  - 远端只做拉取、校验、安装、运行
