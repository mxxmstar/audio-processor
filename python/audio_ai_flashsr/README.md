# audio_ai_flashsr

FlashSR 后端 Worker：用一步蒸馏扩散模型替代 AudioSR 的 50 步 DDIM，
解决音质提升过慢的问题。

本模块是**独立进程**，通过 JSONL 协议（`audio_quality/ai_worker.rs`）与
Rust 侧通信，协议版本与 `python/audio_ai` 完全一致（v1）。

上游源码快照与来源信息见 [`vendor/VENDOR.md`](vendor/VENDOR.md)。

---

## 1. 环境搭建

必须与 AudioSR 的 `.venv` **分开**：AudioSR 锁定 `numpy==1.23.5`，
FlashSR 依赖链解析到 `numpy==2.2.6`，共存会互相破坏。

### 1.1 CPU 版（当前本机为 CPU 环境，无 NVIDIA GPU）

```powershell
cd <仓库根目录>
python -m venv .venv-flashsr
.\.venv-flashsr\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-flashsr.txt
```

> `requirements-flashsr.txt` 中的 `torch` 为通用条目，默认会拉到 PyPI 的
> 默认构建。若需要明确安装 CPU 版，改用：
> `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
> 再 `python -m pip install -r requirements-flashsr.txt --no-deps`

### 1.2 CUDA 版

先按本机驱动安装匹配的 PyTorch（示例为 CUDA 11.8）：

```powershell
.\.venv-flashsr\Scripts\Activate.ps1
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-flashsr.txt --no-deps
```

> 安装后务必确认 `python -c "import torch; print(torch.cuda.is_available())"`
> 输出 `True`，否则 `device: "cuda"` 会以 `CUDA_UNAVAILABLE` 失败。

### 1.3 自检

```powershell
cd <仓库根目录>
.\.venv-flashsr\Scripts\python.exe python/audio_ai_flashsr/selfcheck.py
.\.venv-flashsr\Scripts\python.exe python/audio_ai_flashsr/selfcheck.py --weights
```

> 用**直接执行脚本**而非 `python -m`：`-m` 要求包可导入（cwd 需为 `python/`），
> 而直接执行时 `sys.path[0]` 即包目录 —— 这与 Rust 侧通过
> `python <绝对路径>/worker.py` 启动 Worker 的方式一致，自检结果才可信。

---

## 2. 权重

| 文件 | 大小 | 来源 |
|---|---|---|
| `student_ldm.pth` | 1.03 GB | https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/student_ldm.pth |
| `sr_vocoder.pth` | 628 MB | https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/sr_vocoder.pth |
| `vae.pth` | 1.66 GB | https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/vae.pth |

合计约 3.3 GB，落地到 `models/cache/flashsr/`（已被 `.gitignore` 忽略）。

**不要手工下载**：权重由 `python/audio_ai/model_manager.py` 从
`models/manifest.json` 读取并做 SHA-256 校验，经前端「安装模型」显式触发。

> 上游仓库与权重**均未声明 License**。权重不随应用分发，仅由用户显式下载，
> 详见 `docs/FlashSR集成实施计划.md` 风险 R8。

---

## 3. 依赖实测结论（2026-09-03，Python 3.10 + torch 2.14.0+cpu）

### 3.1 上游声明的 9 个依赖中，只有 5 个是推理必需的

`setup.py` 声明 `numpy tqdm psutil pyyaml matplotlib librosa wandb
tensorboardX einops`。导入 `FlashSR.FlashSR` 后检查 `sys.modules`，
`wandb`、`tensorboardX`、`psutil` **均未被加载**，无需安装。

### 3.2 绘图依赖全部打桩屏蔽（`matplotlib` / `sklearn` / `librosa.display`）

上游有四处顶层导入把绘图依赖拉进了推理路径：

| 位置 | 顶层导入 | 实际用途 |
|---|---|---|
| `TorchJaekwon/Util/UtilTorch.py:10-11` | `matplotlib.pyplot`、`sklearn.manifold.TSNE` | 仅 `tsne_plot()`（训练期 t-SNE 绘图，方法体内还引用了模块级并未导入的 `pandas`） |
| `TorchJaekwon/Util/UtilAudioSTFT.py:6,10` | `matplotlib.pyplot`、`librosa.display` | 仅 `specshow()` 等频谱绘图方法 |
| `TorchJaekwon/Util/UtilAudioMelSpec.py:9` | `librosa.display` | **导入后从未使用** |
| `FlashSR/BigVGAN/utils.py:9-10` | `matplotlib.use("Agg")`、`matplotlib.pylab` | 仅 `plot_spectrogram()`；但被 `FlashSR.SRVocoder` 直接导入，位于推理主链路上 |

`backend.ensure_inference_only_imports()` 为这五个模块预置替身，
**`matplotlib` 与 `sklearn` 无需安装**。

两个踩过的坑，改动桩逻辑时务必注意：

1. **替身必须带合法的 `__spec__`**。上游内联的 diffusers 用
   `importlib.util.find_spec("matplotlib")` 探测依赖，而 `find_spec` 对
   已在 `sys.modules` 中的模块直接返回其 `__spec__`，缺失时抛
   `ValueError: matplotlib.__spec__ is None`。
2. **`librosa.display` 必须一并打桩**。它内部
   `from matplotlib import colormaps`，只桩 `matplotlib` 会抛
   `ImportError: cannot import name 'colormaps' from 'matplotlib'`。
   `librosa` 本体仍是真实依赖（`librosa.filters.mel` 被推理使用），
   只替换 `display` 子模块。

`python -m audio_ai_flashsr.selfcheck` 会打印每个替身的生效状态。

### 3.3 启动耗时与 ready 握手（关键约束）

实测（CPU，冷启动）：

| 组合 | 耗时 |
|---|---|
| 安装 matplotlib（未打桩） | 约 5.5 s |
| 安装 matplotlib + 只桩 `sklearn.manifold` | 约 5.5 s（6.05 → 5.48） |
| **不装 matplotlib，五个替身全开** | **4.83 s** |

更重要的是，matplotlib **首次被导入时会构建字体缓存，实测耗时约 30 秒**。
这一开销出现在用户机器的首次启动上，而 Rust 侧 `ai_worker.rs` 的
`READY_TIMEOUT` 只有 **3 s**，`probe_worker()`（即
`audio_quality_check_ai_runtime`）必然超时，表现为"AI 运行时不可用"。
打桩屏蔽 matplotlib 同时消除了这个隐患。

即便如此，4.83 s 仍超过 3 s 超时。**因此本 Worker 必须延迟导入**：先发出
`ready` 事件，收到 `process` 命令后才导入 `torch` 与 `FlashSR`。
详见实施计划 §6 与阶段 2。

### 3.5 权重加载与 `weights_only`

`torch>=2.6` 起 `torch.load` 默认 `weights_only=True`。实测：

| 权重内容 | 默认（True） | `weights_only=False` |
|---|---|---|
| `OrderedDict[str, Tensor]` | 成功 | 成功 |
| 含 `numpy` 标量 | `UnpicklingError` | 成功 |

HF 对三个权重文件的 pickle 扫描结果为
`torch.FloatStorage` / `torch._utils._rebuild_tensor_v2` /
`collections.OrderedDict`，均在 `weights_only=True` 的白名单内，
**预期可直接加载**。

`backend.torch_load_fallback()` 会在构造期间临时包裹 `torch.load`，
走「先默认、失败再回退 `weights_only=False`」，退出时恢复原函数 ——
上游 `FlashSR.__init__` 在自己内部调用 `torch.load`，无法从外部替换加载
逻辑。待权重实际下载后在阶段 6 确认是否真的需要回退。

---

## 4. 模型约束（写死在 `backend.py` 常量里）

| 约束 | 值 | 说明 |
|---|---|---|
| 单次前向样本数 | `245760` | 48 kHz 下 5.12 秒，**硬限制**，不接收更长输入 |
| 采样率 | `48000` | 输入与输出均固定 |
| 声道 | 1 或 2 | 上游按 `[B, T]` 处理，立体声即 `batch=2`；>2 声道须在 FFmpeg 解码时下混 |
| 默认步数 | `num_steps=1` | 一步蒸馏模型，调大无意义且更慢 |
| 默认低通预处理 | `lowpass_input=False` | 上游 `forward` 默认 `True`，但其官方 `Example.py` 用 `False`。`True` 会走 `UtilAudioLowPassFilter`，存在 Nyquist 陷阱（见实施计划风险 R14），未验证前保持关闭 |

---

## 5. 模块结构与测试

### 5.1 文件职责

| 文件 | 顶层依赖 | 职责 |
|---|---|---|
| `protocol.py` | 标准库 | 事件输出、`WorkerFailure`、ffmpeg/ffprobe 定位 |
| `worker.py` | 标准库 | 主循环、模型清单解析、`ready` 事件 |
| `pipeline.py` | numpy + torch | 探测/解码、定长分块推理、OLA、编码 |
| `backend.py` | numpy + torch | 上游引导、导入期打桩、权重加载、单块推理 |
| `fake_worker.py` | 标准库 | 协议自检假后端，不加载模型 |

`worker.py` 顶层**只导入标准库**，这是 R12 的硬要求：Rust 侧
`READY_TIMEOUT` 只有 3 秒，而 `numpy`（0.9 s）+ `torch`（5.7 s）远超该值。
`pipeline` 由 `_load_pipeline()` 在收到 `process` 命令后才导入。

`protocol.py` 单独存在是为了让 `worker` 与 `pipeline` 共用**同一个**
`WorkerFailure` 类。Rust 以 `python <绝对路径>/worker.py` 启动 Worker，
没有包上下文；若把协议原语留在 `worker.py`，`pipeline` 的
`from worker import ...` 会产生第二个 `worker` 模块对象，异常类随之分裂，
`except WorkerFailure` 将失效、所有错误码退化为 `INFERENCE_FAILED`。

### 5.2 运行测试

```powershell
# 清单与延迟导入（不需要 torch，任意 Python 均可）
.\.venv\Scripts\python.exe python/audio_ai_flashsr/test_worker.py

# 全部 44 项（需要 .venv-flashsr）
cd python/audio_ai_flashsr
..\..\.venv-flashsr\Scripts\python.exe -m unittest discover -p "test_*.py"
```

`test_pipeline.py` 用桩替换 `backend` 的模型加载与推理，因此**不需要
3.3 GB 权重**；探测与解码仍走真实 ffmpeg。

### 5.3 协议自检（无权重、无 GPU）

```powershell
.\.venv\Scripts\python.exe python/audio_ai_flashsr/fake_worker.py --mode success
```

支持 `--mode success|slow|error|crash`，用于验证
`ready` / `progress` / `result` / `error` 四类事件与取消路径。

---

## 6. 上游源码

`vendor/FlashSR_Inference/` 为只读快照，来源、commit、剔除范围与合规说明
见 [`vendor/VENDOR.md`](vendor/VENDOR.md)。**不要在 vendor 内改代码**，
需要修补时在 `backend.py` 中以 monkey patch 处理并注明原因。

上游导入时会向 **stdout** 打印三行诊断信息：

```
There is no Hparams
import error: torch      # 实际指 torchaudio
import error: pydub
```

本 Worker 的 stdout 被 JSONL 协议独占，混入任何非协议文本都会让 Rust 侧
解析失败。`backend.import_flashsr()` 已用 `contextlib.redirect_stdout`
把这些输出改道到 stderr。
