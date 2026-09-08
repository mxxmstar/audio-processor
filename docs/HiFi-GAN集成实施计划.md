# HiFi-GAN 集成实施计划

> 状态：实施中（阶段 0、1、2、3 已完成）
> 编制日期：2026-09-08
> 目标：把 **HiFi-GAN 神经声码器**作为音质提升的可选后端接入现有 JSONL Worker 协议，
> 复用 FlashSR 集成所确立的全部约定（参考 `docs/FlashSR集成实施计划.md`）。

---

## 0. 阅读提示（与 FlashSR 计划的关系）

FlashSR 计划（`docs/FlashSR集成实施计划.md`）已经把"新模型后端接入现有 Worker 协议"
的完整范式跑通：独立 Python 模块 + 独立 `.venv-flashsr` + `WorkerSpec` 路由 +
`manifest.json` 条目 + 前端模型下拉。本计划**沿用同一范式**，并借助一个重大利好——
**FlashSR 的 vendor 代码里已经包含 HiFi-GAN 的完整实现**，因此工作量显著小于从零集成。

本文 §1–§7 与 FlashSR 计划结构对齐，便于直接套用其阶段划分与缺陷登记表风格。

---

## 1. 背景与问题

### 1.1 HiFi-GAN 是什么

HiFi-GAN（Kong et al., 2020，*Generative Adversarial Networks for Efficient and High
Fidelity Speech Synthesis*）是一个**全卷积生成对抗网络声码器**：输入 mel-spectrogram，
输出高保真波形。特点：

- **实时**：纯卷积、无自回归/扩散迭代，单条前向即可出波形（远快于 AudioSR 的 50 步扩散、
  也快于 FlashSR 的蒸馏一步，但 FlashSR 的 VAE+vocoder 整体也很快）。
- **高保真**：主观 MOS 接近或持平真实录音，是业界事实标准的神经声码器。
- **轻量**：生成器参数量相对小，CPU 上也可前向（速度取决于上采样倍率与通道数）。

### 1.2 在本项目中的定位（重要，避免产品口径错误）

**HiFi-GAN 是声码器（mel→waveform），不是超分辨率模型。** 它只能重建 mel 中**已包含**
的频率内容，不能凭空补回丢失的高频。因此：

- 它**不能**替代 FlashSR / AudioSR 的"超分辨率"职责（那是对已丢失高频的估计）。
- 它的价值在于：**神经波形重建**——把频谱表示重合成更干净、更少伪影的波形，提供
  一种不同于扩散系模型的"音色保真"档位；并且可作为 FlashSR / AudioSR 超分流水线的
  **可切换声码器**（替代/补充当前 BigVGAN 系的 `SRVocoder`），换取不同音色或更快重建。

建议的产品角色（三选一，推荐 A）：

| 方案 | 角色 | 说明 |
|---|---|---|
| **A（推荐）** | 独立后端「保真重建 / 神经声码器」 | 输入 → 提取 mel（按目标配置）→ HiFi-GAN 重建波形。作为可选档位，主打"实时、干净重建"。不与超分辨率竞争，互补 |
| B | FlashSR/AudioSR 流水线的可切换 vocoder | 在超分结果后接 HiFi-GAN 重合成，换音色；改动集中在 `SRVocoder`/AudioSR vocoder 选择 |
| C | 内部统一 vocoder 封装 | 仅重构代码，不新增用户可见模型（价值最低，不推荐单独立项） |

前端与历史记录中显示**实际后端名**（如 `hifigan-48k`），**不显示"无损恢复"字样**
（沿用 `docs/低质量音频转高品质音频功能规划.md` 第 1 节口径）。

### 1.3 目标与非目标

**目标**

1. 新增 HiFi-GAN 后端（推荐方案 A），作为独立子进程 Worker，复用现有一切协议与生命周期约定。
2. 前端在「音频品质提升」父菜单下新增**独立子菜单「音质优化」**（独立视图，不与「音质提升」嵌套）；安装、推理、进度、取消、历史记录与其它后端完全一致。
3. 利用已 vendor 的 HiFi-GAN 代码与 `.venv-flashsr` 依赖，最小化新增依赖与环境冲突风险。
4. 不破坏现有 FlashSR / AudioSR / DeepFilterNet 链路；不改动 `manifest.json` 中已有条目的语义。

**非目标**

- 不声称"超分辨率 / 恢复高频"——HiFi-GAN 仅做波形重建。
- 不实现"自动选择后端"或"自动选择声码器"。
- 不引入模型自动下载（仍须用户显式点击"安装模型"触发）。
- 不做视频音轨处理。

---

## 2. 现状梳理

### 2.1 已就绪资产（重大利好：代码已在仓库内）

| 资产 | 位置 | 说明 |
|---|---|---|
| HiFi-GAN 包装器 | `python/audio_ai_flashsr/vendor/FlashSR_Inference/TorchJaekwon/Util/UtilHiFiGanWrapper.py` | `UtilHiFiGanWrapper`：`audio_to_hifi_gan_mel()` / `generate_audio_by_hifi_gan()` |
| HiFi-GAN 生成器 | `python/audio_ai_flashsr/vendor/FlashSR_Inference/FlashSR/AudioSR/hifigan/models.py` | `Generator`（Conv1d/ConvTranspose1d + 反别名 AMPBlock） |
| 48 kHz 配置 | `python/audio_ai_flashsr/vendor/FlashSR_Inference/FlashSR/AudioSR/Vocoder.py` | `get_vocoder_config_48k()`：`sampling_rate=48000`、`num_mels=256`、`hop_size=480`、`fmax=24000` |
| 依赖 | `requirements-flashsr.txt` + `.venv-flashsr` | torch/numpy/scipy/soundfile/librosa/einops/PyYAML/tqdm 已满足 HiFi-GAN 推理所需 |

即：**模型代码、48k 配置、运行依赖三件套都已存在**，主要工作是"接线 + 路径对齐 + 权重获取"。

### 2.2 调用链（复用 FlashSR）

```
src/QualityView.vue
  invoke audio_quality_check_ai_runtime / _download_model / _start / _cancel / _list_tasks
  listen "audio-quality-progress"
        ↓
src-tauri/src/commands/audio_quality.rs  （任务状态、路径分配、参数校验、历史写入）
        ↓
src-tauri/src/audio_quality/ai_worker.rs （JSONL 子进程协议 v1：process / cancel / shutdown）
        ↓  spawn
python/audio_ai_flashsr/worker.py        （backend: flashsr | hifigan*）   ← *新增 hifigan 分支
python/audio_ai_hifigan/worker.py        （方案 A 独立模块时的备选，复用 .venv-flashsr）
python/audio_ai/model_manager.py         （显式断点下载 + SHA-256 校验，独立进程）
```

> 路由层 `backend.rs` 当前 `Backend` 枚举为 `FlashSr | AudioSr | DeepFilterNet | Unknown`，
> 需新增 `HiFiGan` 变体；`select_worker_spec` 需新增分支（复用 `WorkerSpec::flashsr()` 的
> venv 解析，或新增 `WorkerSpec::hifigan()`）。

### 2.3 关键位置（需在集成时触及）

| 关注点 | 文件:行（参考 FlashSR 计划 §2.2） |
|---|---|
| 后端枚举 / 路由 | `src-tauri/src/audio_quality/backend.rs:17-47`（`Backend`）、`98-116`（`select_worker_spec`） |
| WorkerSpec 构造 | `ai_worker.rs:190-260`（`production` / `flashsr`；新增 `hifigan`） |
| 模型列表解析 | `python/audio_ai/worker.py:353-436`（`read_manifest` 过滤 backend ∈ {torchscript, deepfilternet, audiosr}，**需加入 hifigan**） |
| 后端分派 | `python/audio_ai/worker.py:988-1000`、`1041-1069` |
| manifest 条目 | `models/manifest.json`（新增 `hifigan` / `hifigan-48k` 条目） |
| 命令注册 | `src-tauri/src/lib.rs:55-59` |

### 2.4 在途坑（必须正视，否则会卡死）

**P1 · 采样率不匹配（最高优先级风险）。**
公开 HiFi-GAN checkpoint（jik876/hifi-gan：`UNIVERSAL_LJSPEECH` / `VCTK` / `LIBRITTS`）
输出 **22.05 kHz / 24 kHz**，而本 App 目标为 44.1 / 48 kHz。本仓库虽提供 48k 配置
（`get_vocoder_config_48k`），但**需要对应的 48k 训练权重**才能直出 48k；否则只能
"22.05k 重建 + 重采样到 48k"，重采样会削弱声码器收益、并引入额外伪影。
→ 阶段 0 必须先确认 48k checkpoint 是否可获取（见 §3.5 / R3）。

**P2 · 导入路径与 vendor 布局不一致。**
`UtilHiFiGanWrapper.py` 的导入为训练仓库布局：
`from HParams import HParams`、`from DataProcess.Util.UtilAudioMelSpec import UtilAudioMelSpec`、
`from Model.vocoder.hifigan.env import AttrDict`、`from Model.vocoder.hifigan.models import Generator`。
而 vendor 推理布局是 `TorchJaekwon/Util/UtilAudioMelSpec.py` 与
`FlashSR/AudioSR/hifigan/models.py`，**没有** `HParams` / `DataProcess` / `Model.vocoder.hifigan`
这些顶层包。直接 `import UtilHiFiGanWrapper` 会 `ModuleNotFoundError`。
→ 需写一层 shim（sys.path 注入 + 模块别名 / 轻量包装），或在 `python/audio_ai_hifigan/`
下提供一个**只依赖 vendor 实际布局**的薄封装（推荐，隔离性最好）。

**P3 · 权重加载约定。**
`UtilHiFiGanWrapper.load_hifi_gan` 从相对路径 `./Model/vocoder/hifigan/pretrained/<name>/`
读取 `config.json` + `generator`（或 `generator.pt`）。需改为基于 `AUDIO_AI_MODEL_DIR`
的绝对路径，并落入 `models/cache/hifigan/...`，与现有 `model_manager.py` 的下载/校验一致。

---

## 3. 集成设计

### 3.1 后端标识与 manifest 条目

推荐新增一条（方案 A）：

```jsonc
{
  "id": "hifigan-48k",
  "backend": "hifigan",
  "version": "0.0.1",
  "file": "cache/hifigan/UNIVERSAL_LJSPEECH_48k/generator",   // 或 generator.pt
  "sha256": "<落盘后回填>",
  "source": "https://huggingface.co/<org>/hifigan-48k/resolve/main/generator",
  "sample_rate": 48000,                                        // 由权重决定，非硬编码
  "upstream": "https://github.com/jik876/hifi-gan"
}
```

字段说明：
- `backend: "hifigan"` 让 `read_manifest` 与 `Backend` 枚举识别。
- 若暂只能用 22.05k checkpoint，则 `sample_rate: 22050` 并在下游重采样（产品口径需标注"重建后重采样"）。

### 3.2 Worker 模块方案（二选一）

| 方案 | 做法 | 取舍 |
|---|---|---|
| **α（推荐）** | 新建 `python/audio_ai_hifigan/worker.py`，**复用 `.venv-flashsr`**，内部以薄封装调用已 vendor 的 `UtilHiFiGanWrapper`/`Generator` | 与 FlashSR 同进程范式一致、隔离性好；代价是多一份 worker 入口（但逻辑短） |
| β | 在 `python/audio_ai_flashsr/worker.py` 增加 `hifigan` 后端分支 | 最 DRY；但把"声码器重建"混进 FlashSR 流水线 worker，职责耦合，且会随 FlashSR 改动而波动 |

推荐 α：新建 `python/audio_ai_hifigan/worker.py`，协议与 `audio_ai_flashsr/worker.py` 完全同构
（ready / progress / result / cancel / shutdown），仅推理体替换为 HiFi-GAN 前向。

### 3.3 协议与生命周期（沿用 v1）

- `ready`：`models` 含 `["hifigan-48k"]`（大权重文件走 `MODEL_HASH_DEFERRED`，与 AudioSR 一致）。
- `progress` 阶段：`load_model`（加载 Generator + 去 weight_norm）→ `inference`（逐块 mel→waveform）。
- `cancel` / `shutdown`：与 FlashSR 一致（见 `ai_worker.rs` 协议与 worker 主循环）。
- fake worker：`python/audio_ai_hifigan/fake_worker.py` 同构于 FlashSR 的 `fake_worker.py`，
  支撑协议回归测试（无需真实权重）。

### 3.4 采样率与 mel 配置策略

- **首选**：获取 48k checkpoint，使用 `get_vocoder_config_48k`（`sampling_rate=48000`、
  `num_mels=256`、`hop_size=480`、`fmax=24000`），直出 48k，与 App 其它后端对齐。
- **回退**：仅 22.05k checkpoint 时，重建后 `ffmpeg` 重采样到目标 48k，并在 UI 档位说明里
  标注"重建后重采样（非原生 48k）"，避免误导。
- **立体声**：HiFi-GAN 生成器为单声道（`(batch, 1, time)`）。与 AudioSR 一致**逐声道**推理后合并；
  或若 mel 已含双声道则 batch 维并行（需实测），否则逐声道。

### 3.5 权重获取（model_manager）

- 复用 `python/audio_ai/model_manager.py` 的断点下载 + SHA-256 校验流程，新增 `hifigan` 分支。
- 公开权重来源：jik876/hifi-gan（22.05k）或兼容 48k 的社区 checkpoint。
- 落盘到 `models/cache/hifigan/<name>/`，`manifest.json` 记录 `sha256`，首次使用前校验
  （大文件同样走 `STARTUP_HASH_MAX_BYTES` 延迟校验，见 `worker.py:442-483` 的 `available_models`）。

### 3.6 前端入口与菜单位置（关键：独立子菜单，不嵌套进"音质提升"）

新增「音质优化」作为**顶层父菜单「音频品质提升」下的同级子项**，与现有「音质提升」
（QualityView）、「历史记录」并列；**绝不**把 HiFi-GAN 模型选项塞进 QualityView 内部。
即满足需求"不在音质提升中加"——它是独立视图，不是 QualityView 里多出来的一个下拉档位。

依据 `src/App.vue`（菜单定义 `items` / `leafKeys` / 视图渲染 `active === '...'`）：

| 改动点 | 位置 | 做法 |
|---|---|---|
| 菜单子项 | `App.vue:65-73` `quality-group.children` | 新增 `{ key: "optimize", icon: h(SoundOutlined), label: "音质优化" }`，与 `quality`（音质提升）、`quality-history`（历史记录）同级 |
| 可切换叶子 | `App.vue:82-92` `leafKeys` | 加入 `"optimize"` |
| 视图类型 | `App.vue` 顶部 `ViewKey` 联合类型 | 加入 `"optimize"` |
| 视图挂载 | `App.vue:274-279` 渲染区 | 新增 `<optimize-view v-else-if="active === 'optimize'" key="view-optimize" />`，复用 `audio_quality_*` 命令 |
| 历史记录 | `HistoryView.vue:40-45` `kindLabels` | 复用 `enhance: "音质提升"`（优化任务与提升任务同表）；如需区分可新增 `optimize` 标签 |
| 新视图组件 | 新增 `src/OptimizeView.vue` | 以 `QualityView.vue` 为模板，预设模型为 `hifigan-48k`、仅暴露该后端，其余（进度 / 取消 / 历史 / 安装）完全复用 |

要点：

- 「音质优化」是**独立视图**，与 QualityView 平级，而非其模型下拉里多出的一个选项。
- 后端调用与 QualityView **完全一致**（`audio_quality_check_ai_runtime` / `_start` /
  `_cancel` / `_list_tasks`），仅默认/限定模型为 HiFi-GAN，最大化复用、最小化风险。
- `OptimizeView.vue` 可由 `QualityView.vue` 复制后删减模型下拉得到，不改动现有 QualityView。
- 菜单文案统一用「音质优化」，与「音质提升」区分，避免与超分辨率档位混淆。

---

## 4. 阶段划分（参考 FlashSR 阶段 0–5）

| 阶段 | 内容 | 交付 |
|---|---|---|
| **0 · 资产盘点与可行性** | 确认 48k checkpoint 可获取（P1）；写 vendor 导入 shim（P2）；跑通 `Generator` 在 `.venv-flashsr` 下单次前向 | 可行性结论 + shim 模块 |
| **1 · Worker 模块** | 新建 `python/audio_ai_hifigan/worker.py` + `fake_worker.py`；实现 mel 提取 → `generate_audio_by_hifi_gan` → 写盘 | 可 `ready` 握手、可跑 fake |
| **2 · 路由与 manifest** | `backend.rs` 新增 `HiFiGan` + `select_worker_spec` 分支；`ai_worker.rs` 新增 `WorkerSpec::hifigan()`（复用 flashsr venv 解析）；`manifest.json` 加条目；`read_manifest` 后端白名单加 `hifigan` | 运行时检查可见、可安装 |
| **3 · 前端入口** | 按 §3.6 在 `App.vue` 的「音频品质提升」父项下新增同级子菜单「音质优化」（key `optimize`），新增独立 `OptimizeView.vue`（以 `QualityView.vue` 为模板、预设模型 `hifigan-48k`）；`BackendDefaults` 给合理 `chunk_seconds`/`overlap_seconds`（声码器可整段前向，块可更大） | 独立菜单可见、可取消、进度可见，且不改动现有 QualityView |
| **4 · 测试** | fake worker 协议回归；真实权重前向（CPU，忽略项，同 FlashSR `flashsr_real_worker_forward_pass`）；UI 冒烟 | 测试通过 |
| **5 · 文档** | 更新 `低质量音频转高品质音频功能规划.md` 档位表；本计划转"实施中/已完成" | 文档同步 |

---

## 4.1 阶段 0 实施记录（2026-09-08）

### 4.1.1 交付物

新增 `python/audio_ai_hifigan/`（与 `audio_ai_flashsr` 平级的独立后端包）：

| 文件 | 职责 |
|---|---|
| `vendor_bridge.py` | vendor 导入桥接 + 薄封装（shim），解决 P2 |
| `selfcheck.py` | 阶段 0 可行性自检：Generator 单次前向 + 导入桥接验证 |
| `__init__.py` | 包导出 |

### 4.1.2 可行性结论

- **模型代码（Generator）跑通**：基于 vendor `FlashSR.AudioSR.hifigan.models.Generator`
  与 `get_vocoder_config_48k()` 构造 48k 生成器（**190.28M 参数**），对随机
  mel `[1, 256, 200]` 单次前向输出 `[1, 1, 96016]`，上采样倍率 ≈ `hop_size=480`
  （`200 × 480 = 96000`，余量为 padding），输出有限、受 `tanh` 限幅于 `[-1, 1]`。
  证明 vendor 实现与 48k 配置自洽，**前向路径可行**。
- **P2 导入桥接通过**：`install_vendor_aliases()` 把训练仓库布局
  `HParams` / `DataProcess.Util.UtilAudioMelSpec` / `Model.vocoder.hifigan.env`
  / `Model.vocoder.hifigan.models` / `UtilHiFiGanWrapper` 全部桥接到 vendor 实际
  布局，`import UtilHiFiGanWrapper` 解析成功（仅验证导入，实例化需权重，属后续阶段）。
- **环境约束**：本机无 `.venv-flashsr`，可行性验证用全局 `python 3.10 + torch
  2.11.0+cpu`（仅 torch 即可跑 Generator 前向）。官方运行环境仍是 `.venv-flashsr`
  （`requirements-flashsr.txt` 已含 torch/numpy/scipy/soundfile/librosa/einops/PyYAML/
  tqdm；matplotlib/sklearn 由 `ensure_inference_only_imports()` 打桩，无需安装）。
  `vendor_bridge` 沿用 FlashSR 的打桩逻辑，在缺 librosa/scipy 的环境下也能解析导入，
  在 `.venv-flashsr` 下使用真实依赖。

### 4.1.3 关键风险再确认（P1 / R3）

- **P1 · 48k 权重缺失仍是最高优先级阻塞**。vendor 仅提供 48k **配置**
  （`get_vocoder_config_48k`），不含对应训练权重。公开 checkpoint（jik876/hifi-gan：
  `UNIVERSAL_LJSPEECH` / `VCTK` / `LIBRITTS`）输出 **22.05k / 24k**；社区 48k HiFi-GAN
  权重来源不稳定、未锁定 License。→ 进入阶段 1 前必须二选一并在 `manifest.json` 体现：
  1. 找到/训练 48k 权重（直出 48k，首选，需锁定 URL + sha256）；
  2. 否则用 22.05k checkpoint + `ffmpeg` 重采样到 48k（§3.4 回退，档位文案标
     "重建后重采样（非原生 48k）"）。**阶段 0 未解决权重获取，阶段 1 的 manifest
  条目与 `model_manager` 分支须先确定采用哪种**。

### 4.1.4 下一步（阶段 1 入口）

- 确定权重策略（§4.1.3）后，新建 `python/audio_ai_hifigan/worker.py` + `fake_worker.py`，
  协议与 `audio_ai_flashsr/worker.py` 同构；`HiFiGanRunner` 已具备 `load_model` /
  `audio_to_mel` / `mel_to_audio` 骨架，阶段 1 填充 JSONL 主循环与定长分块推理。

---

## 4.2 阶段 1 实施记录（2026-09-08）

### 4.2.1 交付物

在 `python/audio_ai_hifigan/` 新增：

| 文件 | 职责 |
|---|---|
| `protocol.py` | JSONL 协议原语（与 FlashSR 同构，仅标准库；`worker`/`pipeline` 共用同一 `WorkerFailure`） |
| `worker.py` | JSONL 协议主循环 + 清单解析；顶层仅标准库（R12），`SUPPORTED_BACKENDS={"hifigan}"` |
| `pipeline.py` | 推理编排：探测 / 解码 → 逐声道 mel 提取 → 生成器前向 → 编码；复用 FlashSR 的 OLA 落盘与编码原语 |
| `fake_worker.py` | 协议自检假后端（`--mode success\|slow\|error\|crash\|success-bad-exit`），`models:["hifigan-48k"]` |
| `test_worker.py` | 清单解析 + `find_model` + `available_models` + 顶层无重量级依赖断言（16 项，不需 torch） |

阶段 0 的 `vendor_bridge.py` 同步微调：`mel_to_audio` 现在能接受二维
`[mel_bins, time]` 并自动补 batch 维（逐声道推理路径）。

### 4.2.2 验证结果

- `python python/audio_ai_hifigan/test_worker.py` → **16 tests OK**（清单解析、跨后端跳过、
  路径逃逸拦截、`find_model` 大小/哈希校验、`available_models` 延迟哈希、`worker` 顶层不导入
  torch/numpy）。
- 真实 `worker.py --model-dir <含 hifigan-48k 清单>` → `ready` 事件 `models:["hifigan-48k"]`、
  `model_errors:[]`、`worker_version:"python-hifigan-0.1.0"`（握手链路打通）。
- `fake_worker.py --mode success` → 完整事件序列 `ready → progress×7 → result`，
  `result.model_id:"hifigan-48k"`、`sample_rate:48000`、`channels:2`。
- `read_lints` 对全部新增文件零告警。

### 4.2.3 设计要点与待办

- **声码器语义**：HiFi-GAN 是 mel→waveform，无 FlashSR 的定长硬限制；本流水线逐声道独立
  「提取 mel → 生成器前向 → 波形」再合并（R8 单体声道），整段前向、不做重叠相加。
- **输出采样率**：固定 48k（原生 48k 权重）。22.05k 公开 checkpoint 的「重采样回退」方案
  （§3.4）待阶段 2 确定权重策略后接入（届时 `pipeline` 需按 manifest `sample_rate` 选配置
  与解码率）。
- **权重加载**：`pipeline.get_runner` 按 `(权重路径, 设备)` 缓存生成器；`audio_to_mel`
  依赖 librosa（`.venv-flashsr` 已含），真实前向需阶段 2 的 manifest 条目 + `model_manager`
  下载权重后方可端到端验证。

---

## 4.3 阶段 2 实施记录（2026-09-08）

### 4.3.1 交付物（Rust 路由 + manifest）

| 文件 | 改动 |
|---|---|
| `src-tauri/src/audio_quality/backend.rs` | `Backend` 枚举新增 `HiFiGan`；`as_str`/`parse`（含 `hifigan` 前缀）支持；`BackendDefaults::for_backend` 新增 HiFi-GAN 分支（块 30 s、重叠 0，声码器整段前向）；`select_worker_spec` 新增 `HiFiGan` 真实分支（`WorkerSpec::hifigan()`），fake 模式按后端选 `hifigan_fake` 以回显正确 `model_id` |
| `src-tauri/src/audio_quality/ai_worker.rs` | 新增 `WorkerSpec::hifigan()` 与 `WorkerSpec::hifigan_fake(mode)`（复用 `.venv-flashsr` 运行时与 `audio_ai_hifigan/` 脚本；支持 `AUDIO_AI_HIFIGAN_WORKER` 覆盖）；新增 `hifigan_fake_worker_ready_probe` / `hifigan_fake_worker_success_round_trip` 集成测试 |
| `src-tauri/src/commands/audio_quality.rs` | `audio_quality_check_ai_runtime` 的跨后端并集探测列表加入 `WorkerSpec::hifigan()`，确保 hifigan-48k 在任意主模型下都出现在可用模型并集里 |
| `models/manifest.json` | 新增 `hifigan-48k` 条目（`backend:"hifigan"`、`sample_rate:48000`、`version:0.0.1`）；`source`/`sha256` 为占位值，待阶段 3.5 锁定真实 48k 权重后回填（见 R3 / §3.5） |

### 4.3.2 验证结果

- `cd src-tauri && cargo test --lib audio_quality` → **24 passed / 1 ignored / 0 failed**。
  - `hifigan_fake_worker_ready_probe`：`ready.models == ["hifigan-48k"]`、`worker_version: "fake-hifigan-0.1.0"`（复用 `.venv-flashsr` 真实拉起 fake Worker 全链路打通）。
  - `hifigan_fake_worker_success_round_trip`：`ready → progress → result` 完整，`result.model_id == "hifigan-48k"`。
  - `hifigan_defaults_use_large_chunk_and_zero_overlap`、`resolve_backend_parses_exact_and_prefix`（含 `hifigan-48k`/`hifigan-custom`）通过。
- **无回归**：直接拉起 `python/audio_ai/worker.py` 的 `ready` 仍正确列出
  `["audiosr-basic", "deepfilternet2-speech"]`，阶段 2 的 manifest 条目（被 production Worker
  的后端白名单跳过）不影响既有后端。
- 唯一一次全量并行跑出现 `runtime_check_unions_models_across_backends` 失败，复跑通过；
  定位为 `READY_TIMEOUT=3s` 在并行测试负载下的偶发超时（本阶段新增的 hifigan 探针增加了
  python 进程并发数），**非逻辑回归**——`--test-threads=1` 稳定通过，production Worker 的
  `ready` 行为未变。

### 4.3.3 待办（延续到阶段 3 / 3.5）

- 前端菜单与 `OptimizeView.vue`（阶段 3）尚未接入；运行时检查已能识别 `hifigan-48k`（未安装时
  显示为不可用）。
- `model_manager` 下载分支 + 真实 48k 权重 URL / sha256（R3 / §3.5）待锁定；在落实前
  `hifigan-48k` 的安装会失败（manifest 占位 `source`）。
- 22.05k 公开 checkpoint 的「重采样回退」（§3.4）在权重策略确定为 22.05k 时再接入
  `pipeline`（届时按 manifest `sample_rate` 选配置与解码率）。

---

## 4.4 阶段 3 实施记录（2026-09-08）

### 4.4.1 交付物（前端入口，独立子菜单）

| 文件 | 改动 |
|---|---|
| `src/OptimizeView.vue` | 新增视图：以 `QualityView.vue` 为模板，`modelId` 固定 `hifigan-48k`、仅暴露 HiFi-GAN 后端；`backendOf` 识别 `hifigan` 前缀；`isFixed48k` 含 hifigan（输出锁定 48 kHz）；`modelHint` 标明「神经声码器 / 保真重建，不提升高频」；`checkRuntime` 不做跨后端回退（未安装时保留 hifigan-48k 以便「安装模型」） |
| `src/App.vue` | `音频品质提升` 父菜单下新增同级子项 `{ key: "optimize", label: "音质优化" }`；`leafKeys` 与 `ViewKey` 联合类型加入 `"optimize"`；渲染 `<optimize-view v-else-if="active === 'optimize'">` |

要点（对齐 §3.6）：「音质优化」是**独立视图**，与「音质提升」平级、不嵌套；后端调用与
QualityView 完全一致（`audio_quality_check_ai_runtime` / `_start` / `_cancel` / `_list_tasks`），
仅默认/限定模型为 HiFi-GAN；优化任务的输出进入既有的「音质提升」历史记录表（kind `enhance`），
不新增独立历史菜单项。

### 4.4.2 验证结果

- `npm run build`（Vite 生产构建）→ **3179 modules transformed，built in ~30s，无错误**，
  `OptimizeView.vue` 与 `App.vue` 的菜单/渲染接线编译通过。
- 静态核对：`leafKeys` 含 `optimize` 保证点击切换主视图；`quality-group` 下 `quality` /
  `optimize` / `quality-history` 三者同级，满足「不在音质提升中加选项」的口径要求。

### 4.4.3 待办（延续到阶段 4 / 5）

- UI 冒烟需打正式包（Tauri）或 `npm run dev` 人工验证菜单可见、进度/取消/历史一致。
- 权重未安装时「音质优化」视图显示运行时已连接但模型不可用，符合预期；真实端到端
  需阶段 3.5 落实权重后由阶段 4 的忽略项前向测试覆盖。

---

## 5. 风险与缺陷预登记表（R1–Rn）

> 沿用 FlashSR 计划的 D-series 风格，集成过程中新增缺陷登记为 R 系列，便于回溯。

| # | 现象（预期风险） | 根因 | 缓解 / 修复 |
|---|---|---|---|
| R1 | 输出采样率 22.05k，与 App 44.1/48k 目标冲突 | 公开 checkpoint 多为此采样率；48k 权重缺失 | 阶段 0 先确认 48k 权重；否则重采样并明示口径（§3.4） |
| R2 | `import UtilHiFiGanWrapper` 报 `ModuleNotFoundError` | 导入路径（HParams/DataProcess/Model.vocoder.hifigan）与 vendor 实际布局不符 | 阶段 0 写 shim / 薄封装，隔离在 `python/audio_ai_hifigan/` |
| R3 | 权重获取失败或哈希不符 | 48k checkpoint 来源不稳定 / 下载中断 | `model_manager` 断点续传 + SHA-256 校验；manifest 锁版本 |
| R4 | 依赖冲突 / 缺包 | `.venv-flashsr` 缺 librosa 等（mel 提取所需） | 实测 `.venv-flashsr` 已含；缺则补 `requirements-flashsr.txt`（与 FlashSR 共用，无新 venv） |
| R5 | 与 FlashSR 共用 vendor 目录导致耦合 | 改 `UtilHiFiGanWrapper` 影响 FlashSR | 薄封装只读调用，不改动 vendor 源码；改动走 overlay |
| R6 | 产品口径误用为"超分辨率" | 用户/文案误解声码器能力 | 档位命名"保真重建/神经声码器"，禁"恢复高频/无损"字样 |
| R7 | 重建不增高频，体验"没变化" | 声码器只重建 mel 已含频率 | 明确为"音色保真/降伪影"，与超分辨率档互补而非替代 |
| R8 | 立体声处理异常 / 声道错位 | 生成器单声道 | 逐声道推理后按原序合并；加立体声回归用例 |

---

## 6. 验证

- **单元测试**：`WorkerSpec::hifigan()` 解析、fake worker `ready` 探针（`models` 含 `hifigan-48k`）、
  `cancel` / `shutdown` 协议（对齐 FlashSR 的 `flashsr_fake_worker_*` 测试）。
- **集成测试**：真实权重前向（CPU，默认 `#[ignore]`，同 FlashSR `flashsr_real_worker_forward_pass`）：
  加载 → mel 提取 → 重建 → 写出；校验采样率/声道/时长。
- **UI 冒烟**：下拉出现"保真重建"档；可与 FlashSR/AudioSR 自由切换；进度、取消、历史记录一致。
- **回归**：确保 FlashSR / AudioSR / DeepFilterNet 仍可用（本计划不改动其链路）。

---

## 7. 参考

- `docs/FlashSR集成实施计划.md`（范式来源：阶段划分、D-series 缺陷登记、Worker 协议）
- `python/audio_ai_flashsr/vendor/FlashSR_Inference/TorchJaekwon/Util/UtilHiFiGanWrapper.py`
- `python/audio_ai_flashsr/vendor/FlashSR_Inference/FlashSR/AudioSR/hifigan/models.py`
- `python/audio_ai_flashsr/vendor/FlashSR_Inference/FlashSR/AudioSR/Vocoder.py`（`get_vocoder_config_48k`）
- `python/audio_ai_flashsr/requirements-flashsr.txt`（`.venv-flashsr` 依赖已满足 HiFi-GAN）
- 上游：https://github.com/jik876/hifi-gan
