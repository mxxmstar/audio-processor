# FlashSR 集成实施计划

> 状态：规划中（待评审）
> 编制日期：2026-09-03
> 目标：新增独立 Python 模块 `audio_ai_flashsr`，把 FlashSR 作为音质提升的可选后端接入现有 JSONL Worker 协议，解决 AudioSR 推理过慢的问题。

---

## 1. 背景与问题

### 1.1 AudioSR 为什么慢

现有链路为 `QualityView.vue` → `audio_quality_start` → `audio_quality/ai_worker.rs`（JSONL 子进程）→ `python/audio_ai/worker.py` → `audiosr` 后端。

以 3 分钟立体声、48 kHz 为例（`worker.py:1009-1054`、`worker.py:756-841`）：

| 环节 | 成本 |
|---|---|
| 分块 | `AUDIOSR_MAX_CHUNK_SECONDS = 10.24` s → 约 18 块 |
| 每块 | **逐声道**推理 → ×2 = 约 36 次模型调用 |
| 每次调用 | DDIM **50 步**扩散采样（`AUDIOSR_DDIM_STEPS = 50`，guidance 3.5） |
| I/O | 每块每声道都要 `encode_audio()` 写临时 WAV，再被 AudioSR 读回（`infer_audiosr`） |

合计约 **1800 步扩散 + 36 次 WAV 落盘往返**，是耗时的主要来源。

### 1.2 FlashSR 是什么

FlashSR（*One-step Versatile Audio Super-resolution via Diffusion Distillation*，arXiv:2501.10807）是 AudioSR 的**扩散蒸馏一步模型**：把 50 步 DDIM 采样蒸馏为 **1 步**求解，同时保留 latent diffusion + VAE + BigVGAN 系声码器的结构。

关键差异：

| 维度 | AudioSR | FlashSR |
|---|---|---|
| 采样步数 | 50（DDIM） | **1**（DPM-Solver Multistep） |
| 单次输入形状 | 单声道，需逐声道循环 | `[B, T]`，**立体声按 batch 一次通过** |
| 固定块长 | ≤10.24 s（软限制） | **硬限制 245760 样本 = 5.12 s @48 kHz** |
| I/O | 需要临时 WAV 往返 | 直接吃张量，无磁盘往返 |
| 权重体积 | 1 个文件 6.18 GB | 3 个文件合计约 3.3 GB |
| 采样率 | 固定 48 kHz | 固定 48 kHz |

### 1.3 定位说明（重要，避免产品口径错误）

FlashSR 是 AudioSR 的**蒸馏加速版**，音质目标是"接近 AudioSR"而非"超越 AudioSR"，且仍然是对已丢失高频的**估计**，不能声称"恢复无损"。

因此建议的产品口径：

| 档位 | 后端 | 说明 |
|---|---|---|
| 标准（推荐、默认） | `flashsr` | 快，适合批量与长音频 |
| 高保真（慢） | `audiosr-basic` | 保留原 AudioSR，供对结果不满意时对比 |

前端与历史记录中显示的是**实际后端名**，不显示"无损恢复"字样（沿用 `docs/低质量音频转高品质音频功能规划.md` 第 1 节的口径）。

### 1.4 目标与非目标

**目标**

1. 新增独立模块 `python/audio_ai_flashsr/`，作为独立子进程 Worker，复用现有一切协议与生命周期约定。
2. 前端模型下拉中可切换 `flashsr`，安装、推理、进度、取消、历史记录与 AudioSR 完全一致。
3. 同一输入下 FlashSR 的端到端耗时相比 AudioSR 有量级下降（目标 ≥10 倍，实测回填）。
4. 不破坏现有 AudioSR 链路；不改动 `models/manifest.json` 中已有条目的语义。

**非目标**

- 不在本次重构 `python/audio_ai/worker.py` 的推理逻辑（仅修复阻塞性缺陷，见 §5 阶段 0）。
- 不实现"自动选择后端"。
- 不引入模型自动下载（仍必须由用户显式点击"安装模型"触发）。
- 不做视频音轨处理。

---

## 2. 现状梳理

### 2.1 调用链

```
src/QualityView.vue
  invoke audio_quality_check_ai_runtime / _download_model / _start / _cancel / _list_tasks
  listen "audio-quality-progress"
        ↓
src-tauri/src/commands/audio_quality.rs  （任务状态、路径分配、参数校验、历史写入）
        ↓
src-tauri/src/audio_quality/ai_worker.rs （JSONL 子进程协议 v1：process / cancel / shutdown）
        ↓  spawn
python/audio_ai/worker.py                （backend: torchscript | deepfilternet | audiosr）
python/audio_ai/model_manager.py         （显式断点下载 + SHA-256 校验，独立进程）
```

### 2.2 关键位置

| 关注点 | 文件:行 |
|---|---|
| Worker 进程启动与协议 | `src-tauri/src/audio_quality/ai_worker.rs:305-478` |
| ready 握手探测 | `ai_worker.rs:481-564` |
| 模型安装器调用 | `ai_worker.rs:567-609`、`commands/audio_quality.rs:186-210` |
| Python 解释器查找 | `ai_worker.rs:740-757`（`AUDIO_AI_PYTHON` → `.venv/Scripts/python.exe` → `python`） |
| WorkerSpec 构造 | `ai_worker.rs:190-235`（`fake` / `production` / `model_manager`） |
| 任务编排与默认值 | `commands/audio_quality.rs:249-270`（`audiosr-basic` → 10.24 s / 1.28 s） |
| 后端选择 | `commands/audio_quality.rs:385-397`（`select_worker_spec`） |
| 命令注册 | `src-tauri/src/lib.rs:55-59` |
| Python 协议主循环 | `python/audio_ai/worker.py:1166-1236` |
| 后端分派 | `python/audio_ai/worker.py:988-1000`、`1041-1069` |
| 清单解析 | `python/audio_ai/worker.py:353-436`、`models/manifest.json` |
| 下载与校验 | `python/audio_ai/model_manager.py:206-272` |

### 2.3 在途改动（必须先处理）

工作区有未提交改动：`python/audio_ai/worker.py`（+151/−34）、`src-tauri/Cargo.toml`。

其中"分块写盘"重构**尚未完成**，存在必然触发的缺陷：

| 位置 | 问题 |
|---|---|
| `worker.py:1087`、`worker.py:1099` | 调用 `_flush_region(...)`，但仓库内**无该函数定义** → `NameError` |
| `worker.py:1112` | 调用 `encode_pcm_file(pcm_path, ...)`，但 `pcm_path` **从未赋值** → `NameError` |
| `worker.py:1025` | `peak = 0.0` 初始化后从未更新 → `gain` 恒为 1.0、`peak_db` 恒为 −160 dB |

即当前未提交状态下 AudioSR 后端**跑不通**。若不先修复，新模块就没有可对比的基线，也无法判断"FlashSR 更快"到底是模型快还是旧后端崩了。

---

## 3. FlashSR 上游技术事实（调研结论）

来源：`github.com/jakeoneijk/FlashSR_Inference`（`main` 分支）、`huggingface.co/datasets/jakeoneijk/FlashSR_weights`。

### 3.1 分发形态

- **PyPI 上没有官方包**，`setup.py` 声明的包名 `FlashSRInfer` 未发布。必须以源码方式引入。
- 仓库顶层含 `FlashSR/`（模型）与 `TorchJaekwon/`（工具与扩散框架）两个包；`diffusers` 的 DPM-Solver 调度器已被内联在 `TorchJaekwon/Model/Diffusion/External/diffusers/`，**不需要安装 diffusers**。

### 3.2 权重

| 文件 | 大小 | 下载 URL |
|---|---|---|
| `student_ldm.pth` | 1.03 GB | `https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/student_ldm.pth` |
| `sr_vocoder.pth` | 628 MB | `https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/sr_vocoder.pth` |
| `vae.pth` | 1.66 GB | `https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/vae.pth` |

合计约 3.3 GB（对比 AudioSR 单文件 6.18 GB）。HF 页面显示 pickle 导入项为 `torch.FloatStorage` / `torch._utils._rebuild_tensor_v2` / `collections.OrderedDict`。

> **注意**：`size_bytes` 与 `sha256` 必须在实际下载后**实测填写**，不能照抄上表的 1.03 GB 等近似值（那是 HF 的展示值，非精确字节数）。

### 3.3 推理 API（上游 `Example.py`）

```python
import torch
from FlashSR.FlashSR import FlashSR

flashsr = FlashSR(student_ldm_ckpt_path, sr_vocoder_ckpt_path, vae_ckpt_path)
flashsr = flashsr.to(device)

# audio.shape = [channel_size, time]，例如 [2, 245760]，48 kHz
# 目前模型只支持 245760 样本（5.12 秒）
audio = UtilData.fix_length(audio, 245760).to(device)

pred_hr_audio = flashsr(audio, lowpass_input=False)   # -> 同形状
```

`FlashSR.forward` 签名（`FlashSR/FlashSR.py`）：

```python
def forward(self,
            lr_audio: torch.Tensor,        # [batch, time]
            num_steps: int = 1,
            lowpass_input: bool = True,
            lowpass_cutoff_freq: int = None) -> torch.Tensor
```

要点：

- 第 0 维是 **batch**，不是 channel。上游把立体声当作 `batch=2` 传入，等价于"两声道独立推理、共享一次前向"。
- `lowpass_input=True` 时会先经 `UtilAudioSR.find_cutoff_freq` 估截止频率再做切比雪夫低通；上游示例用的是 `False`。**对低码率有损输入，`True` 通常更稳**（缩小训练/推理数据分布差异），建议作为默认并允许关闭。
- 输出已在内部裁剪到 `lr_audio.shape[-1]`，长度恒等于输入。
- 权重通过 `torch.load(path)` 加载，**未传 `weights_only`**：torch ≥2.6 默认 `weights_only=True`，可能抛 `UnpicklingError`，需兼容处理（见 §7 R3）。

### 3.4 依赖

上游 `setup.py` 的 `install_requires`：

```
numpy  tqdm  psutil  pyyaml  matplotlib  librosa  wandb  tensorboardX  einops
```

实际运行还隐含需要：`torch`、`soundfile`、`scipy`（`UtilAudio` 顶层 import）、`torchaudio`（`UtilAudio` 内 try/except 可选 import）。

其中 `wandb`、`tensorboardX`、`psutil`、`matplotlib` 属训练期工具，`librosa`/`einops` 是运行期真实依赖。

### 3.4.1 依赖实测结论（阶段 1，2026-09-03，Python 3.10 + torch 2.14.0+cpu）

| 结论 | 内容 |
|---|---|
| 推理期真实依赖 | `torch` `numpy` `scipy` `soundfile` `librosa` `einops` `pyyaml` `tqdm` |
| 上游声明但未加载 | `wandb` `tensorboardX` `psutil` —— 实测 `sys.modules` 中均无，**不需安装** |
| 可选导入、本项目不用 | `torchaudio` `pydub`（上游 `try/except`；本项目用 FFmpeg 解码，不调用 `UtilAudio.read`） |
| 绘图依赖 | `matplotlib` / `sklearn` / `librosa.display` —— **全部打桩屏蔽，不需安装** |

绘图依赖的打桩细节、两个踩过的坑（`__spec__` 与 `librosa.display`）以及
启动耗时数据，见 `python/audio_ai_flashsr/README.md` §3.2、§3.3。

### 3.5 声道限制

`TorchJaekwon/Util/UtilAudio.py::read` 末尾有断言：

```python
assert (len(audio_data.shape) == 1) or (audio_data.shape[0] in [1, 2])
```

即上游只接受单声道/立体声。新 Worker 自行用 FFmpeg 解码，不调用 `UtilAudio.read`，但模型本身按 batch 处理，>2 声道无定义 → **需要在解码阶段用 `ffmpeg -ac 2` 下混并给出提示**。

---

## 4. 方案设计

### 4.1 总体架构

```
┌──────────────────────────────────────────────────────────┐
│ QualityView.vue                                          │
│  模型下拉：flashsr（标准/快） / audiosr-basic（高保真/慢）  │
└───────────────────────┬──────────────────────────────────┘
                        ↓ invoke（命令名不变）
┌──────────────────────────────────────────────────────────┐
│ commands/audio_quality.rs                                │
│  + audio_quality_list_models（新增，汇总各后端可用模型）    │
│  select_worker_spec(model_id)  ← 改为按后端路由            │
└───────────────────────┬──────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ audio_quality/ai_worker.rs（协议 v1，完全复用）             │
│  WorkerSpec::production()  → python/audio_ai/worker.py    │
│  WorkerSpec::flashsr()     → python/audio_ai_flashsr/…    │
│  WorkerSpec::model_manager() → 不变（无需 torch）           │
└───────────┬──────────────────────────┬───────────────────┘
            ↓                          ↓
  python/audio_ai/           python/audio_ai_flashsr/
  （.venv，numpy 1.23.5）     （.venv-flashsr，独立依赖）
```

### 4.2 关键决策

| # | 决策 | 选择 | 理由 |
|---|---|---|---|
| D1 | 在 `audio_ai` 内加 `flashsr` backend，还是新模块 | **新模块**（用户要求） | 依赖冲突隔离（§D3），且两后端互不干扰 |
| D2 | 新模块是否复用 `audio_ai` 代码 | **阶段 2 自包含复制，阶段 7 收敛抽公共包** | 当前 `audio_ai/worker.py` 有未完成的在途重构（§2.3），此时抽公共包风险高；自包含可独立推进、独立验证 |
| D3 | Python 环境 | **独立虚拟环境 `.venv-flashsr`** | AudioSR 硬锁 `numpy==1.23.5`（`requirements-ai.txt:2`），而 FlashSR 侧 `torch`/`librosa` 生态倾向更新的 numpy，同环境极易冲突 |
| D4 | 模型清单表达 | manifest 条目新增**可选** `files[]` 数组 | 一条 entry = 一个逻辑模型（3 个权重文件）；对现有单文件条目零影响 |
| D5 | 进程模型 | 每任务一次 spawn，协议 v1 原样复用 | 与现有 `run_worker_with_callback` 完全一致，取消/超时/stderr 过滤逻辑全都复用 |
| D6 | Rust 后端路由 | 新增 `audio_quality/backend.rs` 解析 manifest 的 `backend` 字段，失败时回退到 model_id 前缀规则 | 不把后端知识硬编码散落在命令层 |

### 4.3 目录结构（新增部分）

```
python/
└── audio_ai_flashsr/
    ├── __init__.py                 # 包说明
    ├── __main__.py                 # python -m audio_ai_flashsr
    ├── worker.py                   # 协议主循环 + 任务编排（对齐 audio_ai/worker.py）
    ├── backend.py                  # FlashSR 加载/缓存/单块推理封装
    ├── fake_worker.py              # 可选：协议自检用的假后端（与 audio_ai 对齐）
    ├── requirements-flashsr.txt    # 独立依赖清单
    ├── vendor/
    │   └── FlashSR_Inference/      # 上游源码快照（README 中记录 commit）
    │       ├── FlashSR/
    │       └── TorchJaekwon/
    └── README.md                   # 环境搭建与自检步骤

requirements-flashsr.txt            # 项目根，便于一键安装（与 requirements-ai.txt 并列）
```

### 4.4 模块文件职责

**`worker.py`**（与 `audio_ai/worker.py` 同构，自包含）

| 组成 | 说明 |
|---|---|
| `_force_utf8_stdio()` | 原样复制。Windows 区域编码会破坏含日文/汉字的路径 |
| `emit` / `emit_progress` / `emit_error` / `_read_stdin_line` | 原样复制。stdout 独占 JSONL，日志走 stderr |
| `probe_audio` / `decode_audio` / `encode_pcm_file` / `run_ffmpeg_with_blocks` | 复用现有 FFmpeg 封装。解码时若 `channels > 2` 强制 `-ac 2` 并记录提示 |
| `read_manifest` / `find_model` / `available_models` / `sha256_file` | 复用，并**扩展支持 `files[]`**（§4.5） |
| `choose_device` | 原样复制（`auto` / `cpu` / `cuda`） |
| `ChunkAssembler` | 原样复制 OLA 累加器（定稿即落盘，长音频不驻留内存） |
| `enhance()` | 新编排：固定 245760 块长 + 尾块补零 + 立体声按 batch 推理 |
| `process_in_thread` / `main()` | 原样复制。**必须 `daemon=False`**（`audio_ai/worker.py:1220-1222` 的血泪注释） |

**`backend.py`**

```python
FLASHSR_CHUNK_SAMPLES = 245_760      # 5.12 s @ 48 kHz，模型硬限制
FLASHSR_SAMPLE_RATE = 48_000
DEFAULT_NUM_STEPS = 1
DEFAULT_LOWPASS_INPUT = True
DEFAULT_OVERLAP_SECONDS = 0.5

def bootstrap_vendor_path() -> None          # 把 vendor/FlashSR_Inference 加入 sys.path
def load_flashsr(paths: FlashSrWeights, device) -> Any   # 构造 + to(device) + eval()
def cached_flashsr(paths, device) -> Any                 # 线程安全缓存
def infer_flashsr(model, block: np.ndarray, device,
                  num_steps: int, lowpass_input: bool) -> np.ndarray
```

`infer_flashsr` 契约：

- 入参 `block`：`(channels, 245760)` float32，channels ∈ {1, 2}
- 转为 `(channels, 245760)` 张量直接送模型（第 0 维即 batch）
- 出参：同形状 float32；需校验 `isfinite` 且长度一致，否则 `WorkerFailure("OUTPUT_INVALID", …)`
- 前向必须包在 `contextlib.redirect_stdout(sys.stderr)` 中（stdout 属协议专用）
- `torch.inference_mode()` + 捕获 `RuntimeError` 中的 `out of memory` → `OUT_OF_MEMORY`

### 4.5 manifest 扩展：可选 `files[]`

现状（`models/manifest.json`）一条 entry = 一个文件，字段 `file` / `size_bytes` / `sha256` / `source`。

FlashSR 需要 3 个文件，方案是新增**可选** `files`：

```json
{
  "id": "flashsr",
  "backend": "flashsr",
  "version": "2501.10807+5701dea",
  "file": "cache/flashsr",
  "files": [
    {
      "name": "student_ldm",
      "file": "cache/flashsr/student_ldm.pth",
      "size_bytes": 0,
      "sha256": "",
      "source": "https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/student_ldm.pth"
    },
    { "name": "sr_vocoder", "file": "cache/flashsr/sr_vocoder.pth", "...": "..." },
    { "name": "vae",        "file": "cache/flashsr/vae.pth",        "...": "..." }
  ],
  "license": "unknown",
  "upstream": "https://github.com/jakeoneijk/FlashSR_Inference"
}
```

兼容性规则（**向后兼容，现有条目零改动**）：

- 无 `files` 字段 → 视为单文件模型，行为与今天完全一致。
- 有 `files` 字段 → 以 `files` 为准，**忽略**顶层 `size_bytes` / `sha256` / `source`。

需要改动的解析点：

| 位置 | 改动 |
|---|---|
| `python/audio_ai/model_manager.py:30-55` `read_model_entry` | 返回统一的 `artifacts: list[{file, size_bytes, sha256, source}]`；`files` 存在时逐项校验，否则用顶层字段构造单项 |
| `model_manager.py:206-272` `install_model` | 对 `artifacts` 逐个做"已存在则跳过校验 → 分片断点下载 → 合并 → SHA-256 → `os.replace`"，并汇报 `n/total` 与累计进度 |
| `python/audio_ai/worker.py:353-436` `read_manifest` | 允许 `backend` 取值增加 `flashsr`；`files` 存在时逐项校验字段 |
| `worker.py:399-436` `find_model` | 返回权重组（三路径）而非单路径 |
| `worker.py:439-480` `available_models` | 按 `artifacts` 完整校验；大文件延后 hash 的逻辑保持不变 |

> `model_manager.py` 由 `WorkerSpec::model_manager()` 用**基础 `.venv`** 的 Python 启动（只依赖标准库），不随 FlashSR 环境迁移。

### 4.6 分块与重叠相加

- 固定块长 **245760 样本**（5.12 s @48 kHz），`chunk_seconds` 请求值被忽略并给出 `progress` 提示（对齐 `worker.py:1009-1019` 的做法）。
- 默认 `overlap_seconds = 0.5`（24000 样本），可通过请求覆盖；必须满足 `0 ≤ overlap < chunk`。
- 尾块不足 245760 时**补零到满长**，推理后按有效长度裁剪，避免模型收到非法形状。
- OLA 权重与 `audio_ai` 完全一致（首块右端淡出、中间块两端、末块左端淡入），复用 `ChunkAssembler`，保证两后端拼接听感一致。
- 峰值统计在定稿落盘时增量计算：`peak = max(peak, float(np.abs(block).max()))` —— 新模块**不能**重蹈 `audio_ai/worker.py:1025` 的覆辙（`peak` 恒为 0 导致 `gain` 恒为 1.0、`peak_db` 恒为 −160 dB）。

### 4.7 进度阶段

与现有前端语义保持一致，避免改前端进度渲染：

| phase | 时机 | percent 区间 |
|---|---|---|
| `load_model` | 开始加载权重 | 5 |
| `prepare_input` | FFmpeg 解码完成、块长被限制时提示 | 8 |
| `inference` | 每块完成 | 10 → 90 |
| `write_output` | 编码落盘 | 95 |
| `completed` | `result` 事件前 | 100 |

### 4.8 Rust 侧改动清单

**`src-tauri/src/audio_quality/ai_worker.rs`**

```rust
pub fn flashsr() -> Option<Self>          // python/audio_ai_flashsr/worker.py
```

- 新增 `AUDIO_AI_FLASHSR_PYTHON` 环境变量，优先于 `.venv-flashsr/Scripts/python.exe`。
- 把 `find_python()`（L740-757）重构为 `find_python(candidates: &[PathBuf])`，供两个 Worker 复用。
- `is_progress_noise`（L699-713）补充 FlashSR 可能的输出特征（实测后回填）。

**`src-tauri/src/audio_quality/backend.rs`（新增）**

```rust
pub enum Backend { AudioAi, FlashSr }

/// 宽容解析 models/manifest.json 的 id/backend；解析失败时回退到
/// model_id 前缀规则（"flashsr*" -> FlashSr）。
pub fn resolve_backend(model_id: &str) -> Backend;
pub fn resolve_spec(model_id: &str) -> Result<(WorkerSpec, String), String>;
```

**`src-tauri/src/commands/audio_quality.rs`**

- `select_worker_spec()` → `select_worker_spec(model_id: &str)`（L385-397、L249）。
- 默认参数分支（L256-260）增加：`flashsr` → `(5.12, 0.5)`。
- `validate_request`（L399-413）增加：`flashsr` 要求 `output_sample_rate == 48000`，与 `audiosr` 同规则。
- 新增命令 `audio_quality_list_models`：对**每个**后端 Worker 各做一次 `probe_worker`，返回合并列表：

```rust
#[derive(Serialize)] #[serde(rename_all = "camelCase")]
pub struct BackendModel {
    pub model_id: String,
    pub backend: String,        // "flashsr" | "audiosr" | "deepfilternet"
    pub worker_kind: String,    // "python" | "fake" | "configured"
    pub available: bool,
    pub worker_version: Option<String>,
    pub error: Option<String>,
}
```

- `audio_quality_check_ai_runtime` 增加可选 `model_id` 参数：不传时行为与今天完全一致（探测 `audio_ai`），保持前端兼容。

**`src-tauri/src/lib.rs`**：注册 `audio_quality_list_models`（L55-59 附近）。

### 4.9 前端改动清单（`src/QualityView.vue`）

- `modelOptions`（L61-65）改为由 `audio_quality_list_models` 驱动，展示 `[后端] modelId · 可用/不可用`。
- 默认值 `modelId`（L46）由 `audiosr-basic` 改为 `flashsr`，并在不可用时回退到首个可用模型（沿用 L120-122 的逻辑）。
- 模型项增加说明文案：FlashSR = "标准（快）"，AudioSR = "高保真（慢）"。
- `installModel`（L196-210）不变：一次调用安装 `flashsr` 的 3 个权重文件（由 Python 侧 `files[]` 展开）。
- 输出采样率（L73-76）：选中 `flashsr` / `audiosr-basic` 时锁定 48000 并提示原因。
- 其余（进度、取消、历史、打开目录）零改动。

---

## 5. 实施步骤

### 阶段 0：修复在途阻塞缺陷（前置，必须最先做）

- [ ] 补齐 `python/audio_ai/worker.py` 的 `_flush_region()`：把 `ChunkAssembler` 定稿的块顺序写入会话级 f32le 临时文件。
- [ ] 补上 `pcm_path`：在解码后创建临时 PCM 文件，供 `encode_pcm_file`（L314-345）读取。
- [ ] 修复 `peak`：在定稿落盘时增量统计。
- [ ] 用一段 30 s 音频跑通 AudioSR 全链路，确认输出可播放、时长正确、`peak_db` 合理。
- [ ] 提交这批修复（`src-tauri/Cargo.toml` 的行尾改动一并确认是否保留）。

> 验收：AudioSR 端到端成功，`_flush_region` / `pcm_path` 未定义问题消失，`peak_db` 不再恒为 −160 dB。

### 阶段 1：环境与依赖

- [x] 创建 `requirements-flashsr.txt`：实测后确定为 `torch` `numpy` `scipy` `soundfile` `librosa` `einops` `pyyaml` `tqdm`。
- [x] 建立 `.venv-flashsr`（Python 3.10），安装 CPU 版 torch 2.14.0+cpu；CUDA 版安装方式已写入 `python/audio_ai_flashsr/README.md` §1.2。
- [x] 实测依赖触碰情况：`wandb` / `tensorboardX` / `psutil` 未进入 `sys.modules`，已从清单剔除；`matplotlib` / `sklearn` / `librosa.display` 改为打桩屏蔽（见 §3.4.1、README §3.2）。
- [x] 把上游仓库以固定 commit `2292814a7ef74f61a5479c8d96e653d2f90f369d` 快照到 `python/audio_ai_flashsr/vendor/FlashSR_Inference/`，来源、剔除范围与合规说明见 `vendor/VENDOR.md`。
- [x] 记录 `torch.load` 的 `weights_only` 行为（见 R3）：默认路径可加载纯张量 `OrderedDict`，已实现带回退的 `backend.load_state_dict()`。
- [ ] 单测脚本：加载 3 个权重 → 对一段 245760 样本随机张量做 1 步前向 → 打印输出形状/耗时/峰值内存。
      **阻塞**：脚本 `python/audio_ai_flashsr/selfcheck.py --weights` 已就绪，
      但 3.3 GB 权重尚未下载（属阶段 3「模型清单与安装器扩展」，需先扩展
      `manifest.json` 的 `files[]` 与 `model_manager.py`）。建议在阶段 3
      完成后立即回来补跑，并把实测数据填入阶段 6 的对比表。

> 已验收（不依赖权重部分）：`python/audio_ai_flashsr/selfcheck.py` 通过，
> 依赖齐备、上游导入成功、五个替身全部生效，冷启动 4.83 s。
>
> **遗留待验**：权重加载 + 一次前向（等价于上游 `Example.py` 流程），
> 待阶段 3 装好权重后补跑。

### 阶段 2：Python 模块骨架（自包含）

- [ ] 从 `python/audio_ai/worker.py` 复制协议/IO/OLA/清单相关代码到 `python/audio_ai_flashsr/worker.py`（§4.4）。
- [ ] 实现 `backend.py`（加载缓存 + 单块推理 + 三权重路径解析）。
- [ ] 实现 `enhance()`：固定块长、尾块补零、立体声 batch 推理、OLA 定稿落盘、增量 peak。
- [ ] `worker_version` 取 `"python-flashsr-0.1.0"`，便于 Rust 侧与前端区分来源。
- [ ] 先跑 `python -m audio_ai_flashsr` 手工喂一行 JSONL，验证 `ready` / `progress` / `result` / `error` 四种事件。
- [ ] 补 `python/audio_ai_flashsr/test_worker.py` 与 `test_backend.py`（对齐 `audio_ai` 现有测试）。

> 验收：命令行手工 JSONL 往返成功；短音频（20 s）产出可播放 FLAC，时长与输入一致。

### 阶段 3：模型清单与安装器扩展

- [ ] 按 §4.5 扩展 `model_manager.py`（`artifacts` 抽象 + 多文件逐个下载校验 + 进度汇报）。
- [ ] 扩展 `audio_ai/worker.py` 的 `read_manifest` / `find_model` / `available_models` 支持 `files[]` 与 `backend: flashsr`。
- [ ] 在 `models/manifest.json` 中新增 `flashsr` 条目，填入**实测**的 `size_bytes` 与 `sha256`。
- [ ] 补 `model_manager` 的多文件单元测试（含"部分文件已存在时跳过"的续传场景）。

> 验收：`python audio_ai/model_manager.py --model-id flashsr` 一次装完 3 个文件；重复执行秒退；人为破坏 1 个文件后能续传修复。

### 阶段 4：Rust 侧接入

- [ ] `ai_worker.rs`：新增 `WorkerSpec::flashsr()`，重构 `find_python`。
- [ ] 新增 `audio_quality/backend.rs`。
- [ ] `commands/audio_quality.rs`：`select_worker_spec(model_id)`、默认参数分支、48000 Hz 校验、新增 `audio_quality_list_models`、`audio_quality_check_ai_runtime` 增加可选参数。
- [ ] `lib.rs` 注册新命令。
- [ ] 补 Rust 单测：backend 路由（含 manifest 解析失败时的前缀回退）、默认分块参数。

> 验收：`audio_quality_list_models` 同时返回 `flashsr` 与 `audiosr-basic` 及各自可用性。

### 阶段 5：前端接入

- [ ] 按 §4.9 改造 `QualityView.vue`。
- [ ] 模型不可用时，"安装模型"按钮与提示正常工作；安装 3 文件期间给出明确进度文案。

> 验收：下拉可选两个后端，切换后重新检测；选中 flashsr 时采样率锁定 48 kHz。

### 阶段 6：联调与验收

- [ ] 同一段 3 分钟立体声，分别用 `flashsr` 与 `audiosr-basic` 跑通（CPU 一次、CUDA 一次）。
- [ ] 记录并回填实测数据：

| 指标 | audiosr-basic | flashsr |
|---|---|---|
| 端到端耗时 | | |
| 峰值内存 / 显存 | | |
| 输出规格（Hz / ch / 时长） | | |
| `peak_db` | | |

- [ ] 取消功能在两个后端上均生效（任务进入 `cancelled`，子进程退出）。
- [ ] 历史记录 `Enhance` 条目正确，成功项"打开目录"指向真实文件。
- [ ] 边界用例：5 s 以下短音频、单声道、5.1 声道（应下混为 2.0 并提示）、含日文/汉字路径的文件。

### 阶段 7：代码收敛（可选，在 FlashSR 稳定后）

- [ ] 抽 `python/audio_ai_common/`：协议、FFmpeg IO、清单校验、OLA、设备选择。
- [ ] `audio_ai/worker.py` 与 `audio_ai_flashsr/worker.py` 同时改为依赖它，消除重复实现。

---

## 6. 关键实现要点（易踩坑清单）

1. **非守护线程**：`process_in_thread` 必须 `daemon=False`（`audio_ai/worker.py:1220-1222`）。守护线程会让结果/错误事件随主线程退出而丢失，表现为"进程异常退出、退出码 1"，但文件其实已经生成。
2. **stdout 纯净**：上游及其依赖的 `print`、tqdm 必须重定向到 stderr（`contextlib.redirect_stdout(sys.stderr)`）。
3. **UTF-8 stdio**：`_force_utf8_stdio()` 必须在读任何请求前执行；Rust 侧已设 `PYTHONIOENCODING=utf-8`（`ai_worker.rs:327`），Python 侧仍需兜底。
4. **路径逃逸防护**：manifest 中的 `file` 解析后必须确认仍在 `model_dir` 内（`worker.py:405-407`、`model_manager.py:214-217`）。
5. **离线约束**：权重**只能**由 `model_manager.py` 显式下载；推理进程不得访问网络。
6. **模型校验时机**：大模型启动时不算 SHA-256（`STARTUP_HASH_MAX_BYTES = 128 MiB`），否则 Rust 侧 `READY_TIMEOUT = 3 s` 会超时；hash 在 `find_model` 内、加载前必做。
7. **块长为硬限制**：245760 不接受"稍微超一点"，尾块必须补零而非送入超长张量。
8. **声道下混**：>2 声道在 FFmpeg 解码阶段 `-ac 2`，并在 `progress.message` 中说明。
9. **延迟导入（R12，硬性要求）**：`torch` + `FlashSR` 导入实测约 4.8 s，
   超过 `READY_TIMEOUT = 3 s`。`worker.py` 必须在**模块顶层只导入标准库**，
   先完成 `available_models()` 并发出 `ready`，收到 `process` 命令后再导入
   `torch` 与 `FlashSR`。这与 `audio_ai/worker.py:54-55`（顶层即
   `import torch`）不同，是本模块为满足握手超时而必须的偏离。
10. **导入期 stdout 拦截**：上游导入会打印 `There is no Hparams`、
    `import error: torch`、`import error: pydub` 三行到 stdout。
    `backend.import_flashsr()` 已统一改道到 stderr，不要在它之外单独
    `import FlashSR.*`。

---

## 7. 风险与应对

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| R1 | `numpy==1.23.5`（AudioSR）与 FlashSR 侧 `torch`/`librosa` 生态冲突 | 同 venv 内两个后端不可共存 | 独立 `.venv-flashsr`（D3）；`requirements-flashsr.txt` 与 `requirements-ai.txt` 严格分离；`AUDIO_AI_FLASHSR_PYTHON` 支持自定义解释器 |
| R2 | 上游无 PyPI 包，需 vendor 源码 | 仓库体积增大、升级需手工 | 以固定 commit 快照 vendor，README 记录来源与 commit；上游 license 未明确，合规评估前**不要**随安装包分发权重（见 R8） |
| R3 | `torch.load` 在 torch ≥2.6 默认 `weights_only=True` | 权重加载失败 | 优先 `torch.load(p, map_location=device)`，捕获 `UnpicklingError` / `PickleError` 后回退 `weights_only=False`；**不**直接改全局默认值。已实现于 `backend.torch_load_fallback()`（构造期间临时包裹 `torch.load`，`finally` 中恢复 —— 上游 `FlashSR.__init__` 在自己内部调 `torch.load`，无法从外部替换）。实测（torch 2.14.0+cpu）：`OrderedDict[str, Tensor]` 在默认下即可加载；含 numpy 标量时才需要回退。HF 对三个权重的 pickle 扫描结果均在白名单内，**预期走默认路径**，待阶段 6 用真实权重确认 |
| R4 | 立体声按 batch=2 推理，显存翻倍 | 低端 GPU OOM | 捕获 OOM 后自动降级为逐声道推理（`chunk` 拆成 `[1, T]` 两次调用）并在 `progress.message` 中提示；仍失败则返回 `OUT_OF_MEMORY`（可重试） |
| R5 | 固定 5.12 s 块长导致块数翻倍 | 抵消部分加速收益 | 一次前向仅 1 步 + 无 WAV 往返，净收益仍显著；若实测不足，再评估 `num_steps` 与 batch 合并策略 |
| R6 | 3.3 GB 下载，耗时与失败率高 | 用户体验 | 复用现有 Range 分片 + 断点续传 + SHA-256；前端按 `n/3` 汇报文件级进度 |
| R7 | 未提交的 `worker.py` 重构尚未完成（§2.3） | 无可用基线，且抽公共包会放大风险 | 阶段 0 先修复并提交；阶段 2 采用自包含复制而非公共包抽取 |
| R8 | 上游仓库与权重**未声明 license** | 分发合规风险 | 权重不随应用分发，仅由用户显式下载；`manifest.json` 中 `license` 标注 `unknown`；在 README 与前端提示"第三方模型，许可未明确，仅个人使用" |
| R9 | 上游 `UtilAudio.read` 断言声道 ∈ {1,2} | >2 声道输入崩溃 | 新 Worker 不调用 `UtilAudio.read`，改用 FFmpeg 解码并 `-ac 2` 下混 |
| R10 | FlashSR 输出响度/电平与 AudioSR 不一致 | 两档位切换时听感差异 | 统一走增量 peak 归一化（`gain`），并在结果中回传 `peak_db` 便于比对 |
| R11 | ~~`wandb` / `tensorboardX` 在 import 时被触碰~~ | 依赖膨胀 | **阶段 1 已实测排除**：两者与 `psutil` 均未进入 `sys.modules`，无需安装，也无需打桩 |
| R12 | **导入 torch + FlashSR 约 4.8 s，超过 Rust 侧 `READY_TIMEOUT = 3 s`**（`ai_worker.rs:19`） | `audio_quality_check_ai_runtime` 恒返回 `ReadyTimeout`，前端显示"AI 运行时不可用" | **必须延迟导入**：Worker 先发 `ready`，收到 `process` 命令后才导入 `torch` 与 `FlashSR`（见 §6 要点 9）。阶段 4 另需评估是否上调 `READY_TIMEOUT`。已在阶段 1 通过打桩把耗时从约 5.5 s 降到 4.83 s，并消除了 matplotlib 首次构建字体缓存的约 30 s 尖峰 |
| R13 | 本机无 NVIDIA GPU（无 `nvidia-smi`），阶段 6 无法实测 CUDA | 无法验证 GPU 路径与显存降级（R4） | 阶段 1/2 只保障 CPU 正确；CUDA 路径在 README 中给出安装方式，并在有 GPU 的机器上补测；R4 的 OOM 降级逻辑照常实现，只是无法在本机触发 |

---

## 8. 验收标准

1. ✅ 前端模型下拉同时出现 `flashsr`（标准/快）与 `audiosr-basic`（高保真/慢），各自显示可用状态。
2. ✅ 点击"安装模型"能一次下载并校验 FlashSR 的 3 个权重文件；中断后可续传。
3. ✅ 选中 `flashsr` 后，处理长音频的进度、取消、失败提示与 AudioSR 完全一致。
4. ✅ 同一输入下 FlashSR 端到端耗时相比 AudioSR **≥10 倍**下降（以 §阶段 6 实测数据为准，未达标则在文档中如实标注实测倍数）。
5. ✅ 输出为 48 kHz 可播放 FLAC，时长与输入一致，不覆盖原文件。
6. ✅ 单声道、立体声、5.1（下混）、含非 ASCII 字符路径、5 s 以下短音频均能正常产出。
7. ✅ AudioSR 链路无任何回归（阶段 0 的修复已提交并通过原有测试）。
8. ✅ `cargo test` 与 Python 侧测试全部通过；新增代码覆盖 backend 路由、多文件清单解析、OLA 拼接。

---

## 9. 后续优化

- [ ] 抽 `python/audio_ai_common/` 消除两 Worker 的重复协议代码（阶段 7）。
- [ ] 后端自动选择：根据音频时长、设备显存推荐 `flashsr` / `audiosr-basic`。
- [ ] `lowpass_input` 与 `num_steps` 作为高级参数暴露给前端。
- [ ] 批量处理（当前一次仅一个任务，`BUSY` 保护见 `worker.py:1203-1205`）。
- [ ] 显存自适应 batch：长音频下按可用显存决定一次送几块。
- [ ] 与下载完成后的自动后处理流程串联（`docs/低质量音频转高品质音频功能规划.md` 的既定目标）。

---

## 10. 遗留与待办（进度快照：2026-09-04）

### 10.1 阶段进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| 阶段 0：修复在途阻塞缺陷 | **未开始** | 见 10.2 第 1 项，**当前最高优先级** |
| 阶段 1：环境与依赖 | 基本完成，1 项阻塞 | 不依赖权重的部分已验收；权重前向测试见 10.2 第 2 项 |
| 阶段 2：Python 模块骨架 | 未开始 | `backend.py` 已落地引导/兼容/加载，协议与编排待写 |
| 阶段 3：模型清单与安装器扩展 | 未开始 | 是阶段 1 遗留项的解锁前置 |
| 阶段 4：Rust 侧接入 | 未开始 | |
| 阶段 5：前端接入 | 未开始 | |
| 阶段 6：联调与验收 | 未开始 | |
| 阶段 7：代码收敛 | 未开始 | |

### 10.2 遗留项清单

**1. 阶段 0 未开始：`python/audio_ai/worker.py` 在途重构有 3 个阻塞缺陷**

已随提交 `439fdeb` 入库，属已知不可用状态：

| 位置 | 问题 | 后果 |
|---|---|---|
| `worker.py:1087`、`1099` | 调用 `_flush_region(...)`，全仓库无此函数定义 | `NameError` |
| `worker.py:1112` | 调用 `encode_pcm_file(pcm_path, ...)`，`pcm_path` 从未赋值 | `NameError` |
| `worker.py:1025` | `peak = 0.0` 初始化后从未更新 | `gain` 恒为 1.0、`peak_db` 恒为 −160 dB |

**影响**：AudioSR 后端当前跑不通，没有可对比的性能与音质基线，
"FlashSR 是否更快"无从验证。必须在阶段 6 之前完成。

**2. 阶段 1 阻塞：权重加载与一次前向尚未实测**

`python/audio_ai_flashsr/selfcheck.py --weights` 已就绪，但 3.3 GB 权重未下载。
下载依赖阶段 3（`manifest.json` 的 `files[]` 扩展 + `model_manager.py` 支持多文件）。

**解锁条件**：阶段 3 完成后立即补跑，把输出形状、耗时、实时倍率、峰值内存
填入阶段 6 的对比表，并确认 R3 的 `weights_only` 是否真需要回退。

**3. CUDA 路径无法在本机实测（R13）**

本机无 NVIDIA GPU（无 `nvidia-smi`），阶段 1/2 只保障 CPU 正确性。
CUDA 安装方式已写入模块 README §1.2；R4 的显存 OOM 降级逻辑照常实现，
但无法在本机触发验证。

**4. 上游 License 未声明（R8）**

上游仓库与权重均未提供 License。处置：权重不随应用分发，仅用户显式下载；
`manifest.json` 中 `license` 标注 `unknown`；`vendor/VENDOR.md` 与模块 README
均已写入提示。**正式分发前需完成合规评估。**

**5. 阶段 7 代码收敛未做**

阶段 2 采用"自包含复制"策略，`audio_ai` 与 `audio_ai_flashsr` 会各有一份
协议/IO/OLA 实现，存在漂移风险。需在 FlashSR 稳定后抽 `python/audio_ai_common/`。

### 10.3 已完成的关键决策与实测（供后续阶段参照）

- 上游快照 commit `2292814a7ef74f61a5479c8d96e653d2f90f369d`，177 文件 / 1.13 MB。
- 独立环境 `.venv-flashsr`：torch 2.14.0+cpu、numpy 2.2.6（与 AudioSR 的 1.23.5 冲突）。
- 依赖实测：真实只需 8 个包；`wandb`/`tensorboardX`/`psutil` 未触碰；
  `matplotlib`/`sklearn`/`librosa.display` 已打桩屏蔽。
- 冷启动 4.83 s，超过 `READY_TIMEOUT = 3 s` → **阶段 2 必须延迟导入**（R12、§6 要点 9）。

---

**文档版本**：v1.1
**创建日期**：2026-09-03
**最后更新**：2026-09-04
**上游参考**：
- 论文 https://arxiv.org/abs/2501.10807
- 代码 https://github.com/jakeoneijk/FlashSR_Inference
- 权重 https://huggingface.co/datasets/jakeoneijk/FlashSR_weights
