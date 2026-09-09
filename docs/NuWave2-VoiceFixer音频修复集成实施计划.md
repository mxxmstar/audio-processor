# 音频修复增强 · NuWave2 & VoiceFixer 集成实施计划

> 状态：**阶段 0 已完成**（2026-09-09，环境/权重/缓存/性能均已实测，见 §4.1）；阶段 1 待实施
> 编制日期：2026-09-09
> 目标：评估并为 **NuWave2**（通用音频上采样）与 **VoiceFixer**（语音综合修复）两个后端
> 制定接入方案，补齐现有"只能升采样 / 只能降噪"的能力缺口。
> **许可结论（已核实）：NuWave2 = BSD 3-Clause，VoiceFixer = MIT —— 均为宽松许可，无 CC-BY-NC 类商用障碍。**

---

## 0. 阅读提示

| 文档 | 贡献 |
|---|---|
| `docs/FlashSR集成实施计划.md` | 新后端接入 JSONL Worker 协议的范式（独立 venv / WorkerSpec 路由 / manifest / D-series） |
| `docs/HiFi-GAN集成实施计划.md` | 前端菜单位置口径；`python/audio_ai_hifigan/` 分层结构模板 |
| `docs/Real-ESRGAN集成实施计划.md` | 横向复制到新领域的做法（图像域） |
| `docs/AudioCraft-MusicGen集成实施计划.md` | 门禁机制（授权 / 硬件 / 环境）——本文档沿用，并先给出结论 |
| 本文档 | 补齐**音频修复**能力；两个模型**对比选型 + 分优先级落地** |

**与 MusicGen 的关键区别**：本计划两个模型**均为宽松许可**，且 **VoiceFixer 原生支持 CPU**
（默认 `cuda=False`），**不受"无 GPU"门禁阻塞**——这是它能优先落地的核心原因。

---

## 1. 背景与问题

### 1.1 现有音频能力版图与缺口

| 现有后端 | 能做什么 | 不能做什么 |
|---|---|---|
| `flashsr` | 超分辨率 → 48 kHz（快，蒸馏一步） | 不降噪、不去混响、不修复削波 |
| `audiosr-basic` | 超分辨率 → 48 kHz（慢，50 步扩散） | 同上 |
| `deepfilternet2-speech` | **降噪** | 不提升带宽、不去混响、不修复削波 |
| `hifigan`（规划中） | 声码器波形重建 | 不修复退化 |

**缺口**：真实"低质量音频"往往**同时**存在**噪声 + 混响 + 低带宽 + 削波**，
而现有后端各自只解决其中**一项**。用户需要多次处理或无法处理（如削波、混响）。

### 1.2 NuWave2 是什么

- 仓库：`maum-ai/nuwave2`（MINDsLab / SNU），论文 arXiv:2206.08545
- **通用神经音频上采样**扩散模型：可从**多种输入采样率**生成 **48 kHz** 音频。
- 官方提供两个 checkpoint：主模型（目标 48 kHz）与**目标 16 kHz 版本**（源 3.2 kHz~16 kHz）。
- 推理：`inference.py -c {ckpt} -i {wav} --sr {输入采样率} [--steps] [--gt] [--device cuda|cpu]`
  —— **`--device cpu` 官方支持**；`--steps` 控制扩散步数。
- 许可：**BSD 3-Clause**（宽松）。

### 1.3 VoiceFixer 是什么

- 仓库：`haoheliu/voicefixer`；论文 arXiv:2109.13731（与 AudioSR 同一作者）
- **一个模型内同时处理**：**噪声 / 混响 / 低分辨率（2 kHz~44.1 kHz）/ 削波（0.1–1.0 阈值）**。
- 结构：分析模块（mel 修复）+ 神经声码器；并提供 **44.1 kHz 通用说话人无关声码器**。
- **运行模式**（官方定义）：

| Mode | 说明 |
|---|---|
| `0` | 原始模型（**默认推荐**） |
| `1` | 增加预处理模块（去除高频） |
| `2` | 训练模式（对严重退化的真实语音有时更有效） |
| `all` | 跑全部模式，各输出一个文件 |

- 安装：`pip install voicefixer`（PyPI 有包）。
- API：`VoiceFixer().restore(input, output, cuda=False, mode=N)`；`Vocoder(sample_rate=44100).oracle(...)`。
- **默认 `cuda=False` → CPU 优先设计，CPU 可用**。
- 权重：两个文件 `vf.ckpt`（分析模块）+ `model.ckpt-1490000_trimed.pt`（合成/声码器，44100），
  默认落 `~/.cache/voicefixer/...`；官方提供**百度网盘**镜像（对国内网络友好）。
- **支持替换自定义声码器**（`your_vocoder_func`，需 44.1 kHz / 128 mel bins）
  → 与已规划的 **HiFi-GAN 模块有协同空间**。
- 许可：**MIT**（宽松）。

### 1.4 对比与选型建议

| 维度 | NuWave2 | **VoiceFixer** |
|---|---|---|
| 任务 | 音频**上采样**（→48 kHz） | **语音综合修复**（噪声+混响+带宽+削波） |
| 与现有后端重叠 | ⚠️ **与 FlashSR / AudioSR 高度重叠**（都是→48 kHz 超分） | ✅ **互补**，填补降噪以外能力 |
| 推理范式 | **扩散**（`--steps`）→ 慢 | 分析 U-Net + 声码器 → **相对快** |
| CPU 可行性 | 支持 `--device cpu`，但扩散步数多 → **慢** | ✅ **默认 CPU**（`cuda=False`） |
| 输出采样率 | 48 kHz | **44.1 kHz** |
| 权重获取 | ⚠️ **Google Drive**（需处理确认令牌，集成摩擦大） | pip 包 + 自动下载 + **百度网盘镜像** |
| 依赖 | ⚠️ torch>=1.7.0 + **pytorch-lightning==1.2.10**（2021 年旧版，与 torch 2.x 不兼容） | pip 包，依赖较现代 |
| 许可 | BSD 3-Clause ✅ | MIT ✅ |
| 适用对象 | 通用音频 | **人声/语音**（非音乐） |

**选型建议**

1. **优先落地 VoiceFixer**：能力互补、CPU 可用、许可宽松、权重易得、见效最快。
2. **NuWave2 作为次优先 / 可选**：与现有超分后端功能重叠，且扩散慢、权重在 Google Drive、
   依赖旧版 pytorch-lightning，投入产出比低。**建议先做 VoiceFixer，再评估是否真需要它**。

### 1.5 定位说明（避免口径错误）

- VoiceFixer 面向**人声/语音**（播客、录音、视频人声、电话录音），**不是音乐增强**；
  音乐场景仍用 FlashSR / AudioSR。
- 输出为 **44.1 kHz**（VoiceFixer 原生），**不要**强制成 48 kHz（见 §3.5）。
- 两者都是**生成式修复**：对已丢失信息做估计，**不可宣称"无损还原"**。
- 沿用既有纪律：UI 与历史显示**实际后端名**。

### 1.6 目标与非目标

**目标**

1. 新增 **VoiceFixer** 后端（优先），补齐噪声/混响/削波/低带宽的**综合修复**能力。
2. 复用既有 JSONL Worker 协议，支持进度、取消、模型安装、历史记录。
3. **NuWave2** 作为第二阶段可选项，先完成可行性评估（§4 阶段 5）。
4. 不破坏现有 FlashSR / AudioSR / DeepFilterNet / HiFi-GAN 链路。

**非目标**

- 不做批量队列、不做实时处理。
- NuWave2 首版**不实现**（仅评估），避免与 AudioSR 功能重复。
- 不做模型训练/微调。

---

## 2. 现状梳理

### 2.1 可复用资产

| 资产 | 位置 | 复用方式 |
|---|---|---|
| JSONL Worker 协议 v1 | `src-tauri/src/audio_quality/ai_worker.rs` | 复制为各新模块的 `ai_worker.rs` |
| 后端路由 | `audio_quality/backend.rs`（`Backend` 枚举 + `select_worker_spec`） | 新增 `VoiceFixer`（及后续 `NuWave2`）变体与分支 |
| Python 分层模板 | `python/audio_ai_hifigan/` | **照搬**（protocol / pipeline / worker / fake_worker / selfcheck / tests） |
| 下载 + SHA-256 | `python/audio_ai/model_manager.py` | 新增分支（VoiceFixer 权重） |
| 历史记录 | `src-tauri/src/history.rs:11-47` | `kind` 为 TEXT 列，**新增 kind 无需迁移** |
| 命令注册 | `src-tauri/src/lib.rs:50-93` | 新增模块与命令 |
| ffmpeg | 应用已带 `bin/ffmpeg.exe` | 音频解码/编码沿用既有方式 |

### 2.2 与既有增强后端的差异（决定设计取舍）

| 维度 | FlashSR / AudioSR | **VoiceFixer** |
|---|---|---|
| 输出采样率 | 48 kHz | **44.1 kHz** |
| 处理对象 | 通用音频 | **人声/语音** |
| 分块 | 按时间 `chunk_seconds` | 需确认（整段或分块，阶段 0 验证） |
| 额外参数 | device / chunk | **mode（0/1/2）** |
| 是否需要 GPU | CPU 可跑（慢） | ✅ **CPU 优先** |

---

## 3. 集成设计

### 3.1 优先级与落地顺序

```
阶段 1–4：VoiceFixer（优先，CPU 可用、能力互补、见效快）
阶段 5  ：NuWave2 可行性评估（决定是否实现）
```

### 3.2 模块与目录（照搬 `audio_ai_hifigan` 分层）

```
python/audio_ai_voicefixer/
  ├ __init__.py
  ├ protocol.py        # JSONL 事件
  ├ pipeline.py        # VoiceFixer 加载 + mode 分派 + 进度/取消
  ├ worker.py          # 主循环
  ├ fake_worker.py     # 协议回归（无需真实权重）
  ├ selfcheck.py       # 依赖 / 权重 / 缓存目录自检
  └ test_worker.py

requirements-voicefixer.txt
.venv-voicefixer/                        # 独立环境（阶段 0 验证能否复用现有 venv）

# 后续（若实施 NuWave2）
python/audio_ai_nuwave2/  （同构）
requirements-nuwave2.txt     # 需锁旧版 torch + pytorch-lightning==1.2.10
.venv-nuwave2/

src-tauri/src/audio_quality/backend.rs   # 新增 Backend::VoiceFixer（及 NuWave2）
src-tauri/src/audio_quality/ai_worker.rs # 新增 WorkerSpec::voicefixer() + find_python_voicefixer()
src/commands/audio_quality.rs            # 复用既有命令（按 model_id 路由）
src/QualityView.vue                      # 模型下拉新增档位（见 §3.6）
```

> 说明：VoiceFixer 与 FlashSR/AudioSR 同属"音频品质提升"，走**既有 `audio_quality` 命令**即可，
> 通过 `model_id` 路由到新 Worker——**不必新建一套命令**（与图像模块不同）。

### 3.3 协议与生命周期

沿用 JSONL v1（`process` / `cancel` / `shutdown`；`ready` / `progress` / `result`）。载荷：

- 请求：`model`（`voicefixer`）、`mode`（0/1/2）、`device`、`input_path`、`output_path`。
- `progress.phase`：`load_model` → `restore`（修复）→ `encode`（写盘）。
- `result`：输出路径 + **采样率 44100** + 声道 + 时长 + 所用 mode。
- **取消**：若 VoiceFixer 单次 `restore()` 不可打断，则按**时间分块**调用并在块间检查取消标志
  （与音频分块、图像 tile 同源思路）；块间需注意**重叠与衔接**（阶段 1 实测）。

### 3.4 权重、manifest 与缓存目录控制

VoiceFixer 默认把权重放到 **`~/.cache/voicefixer/`**，与本项目"权重集中到 `models/cache/`、
由用户显式安装"的纪律冲突。**必须解决缓存目录可控性**（阶段 0 验证）：

- 优先：查找 VoiceFixer 是否提供环境变量 / API 指定缓存目录（阶段 0 核实）。
- 兜底：为 Worker 子进程设置 `HOME`（或对应缓存变量）指向 `models/cache/voicefixer/`。
- 校验：仍走 `model_manager` 的 SHA-256 校验；manifest 记录两个权重文件。

```jsonc
{
  "id": "voicefixer",
  "backend": "voicefixer",
  "version": "0.0.1",
  "files": [
    { "name": "vf",       "file": "cache/voicefixer/analysis_module/checkpoints/vf.ckpt",
      "source": "<官方/镜像直链>", "sha256": "<回填>" },
    { "name": "vocoder",  "file": "cache/voicefixer/synthesis_module/44100/model.ckpt-1490000_trimed.pt",
      "source": "<官方/镜像直链>", "sha256": "<回填>" }
  ],
  "upstream": "https://github.com/haoheliu/voicefixer"
}
```

> NuWave2（若实施）：checkpoint 在 **Google Drive**，需处理下载确认令牌；
> 建议改为自托管直链或镜像后纳入 manifest（见 R6）。

### 3.5 输出采样率差异（易踩坑）

- FlashSR / AudioSR：48 kHz；**VoiceFixer：44.1 kHz**。
- 现有代码中 FlashSR 有"强制 48 kHz 输出"的校验（`validate_flashsr_sample_rate`）。
  **新增 VoiceFixer 后端时不得套用该校验**，否则会把 44.1 kHz 结果重采样成 48 kHz。
- 处理：`BackendDefaults` 与前端输出采样率控件需**按后端区分**（VoiceFixer 锁定/默认 44100）。

### 3.6 前端入口

VoiceFixer 属于"音频品质提升"域，**直接加进既有的 `QualityView.vue` 模型下拉**（与新增独立菜单组不同）：

| 改动点 | 做法 |
|---|---|
| 模型下拉 | `QualityView.vue` 档位新增「语音修复（VoiceFixer）」，选中后 `modelId = "voicefixer"` |
| mode 选择 | 新增 0/1/2 下拉（默认 0），映射到请求 `mode` |
| 输出采样率 | 按 §3.5，选中 VoiceFixer 时默认/锁定 44.1 kHz |
| 历史 | 复用 `enhance` kind（同属音质提升）；如需区分可另加 kind |
| 文案 | 标注"面向人声/语音"，避免用户拿去修音乐 |

---

## 4. 阶段划分

| 阶段 | 内容 | 交付 |
|---|---|---|
| **0 · 可行性验证** ✅ **已完成 2026-09-09** | 建 `.venv-voicefixer`（或验证能否复用现有 venv）；`pip install voicefixer`；**验证缓存目录可重定向到 `models/cache/`**；CPU 跑通一次 mode 0；实测耗时/内存 | 环境 OK + 实测数据回填 + 缓存方案定稿（见 §4.1） |
| **1 · Python 模块** | 照 §3.2 建 `python/audio_ai_voicefixer/`；`pipeline.py` 实现 mode 分派 + 分块进度/取消；fake / selfcheck / test 齐备 | 可 `ready` 握手、fake 可跑 |
| **2 · Rust 侧** | `backend.rs` 新增 `VoiceFixer` 枚举与路由；`ai_worker.rs` 新增 `WorkerSpec::voicefixer()` + `find_python_voicefixer()`；`manifest.json` 条目 | 运行时检查可见、可安装 |
| **3 · 前端** | §3.6 下拉档位 + mode 选择 + 采样率联动 | 可选、可取消、进度可见 |
| **4 · 测试** | fake 协议回归；真实权重修复用例（含噪声/削波/低带宽样本）；UI 冒烟 | 测试通过 |
| **5 · NuWave2 评估** | 评估 NuWave2 是否值得做（对比 AudioSR 的差异化价值、Google Drive 权重方案、旧版 lightning 环境可行性）→ 决定实现或搁置 | 评估结论；若实施则复用阶段 1–4 流程 |

### 4.1 阶段 0 实测结论（2026-09-09，验证机：Python 3.10 / CPU / 无 GPU / 12 线程）

**环境**：`.venv-voicefixer`（`torch==2.14.0+cpu` + `voicefixer==0.1.3`），依赖清单见 `requirements-voicefixer.txt`。

**① 权重获取（修正 §3.4 / R6 相关前提）**

- 官方文档给的 **Zenodo 直链（`zenodo.org/record/5600188/...`）在本机返回 403**，不可用。
- 实测可用的公开镜像：**Hugging Face `Diogodiogod/voicefixer-models`**（两文件同名，HTTP 200，支持断点续传）：

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `vf.ckpt`（分析模块） | 489,307,071 B（466.7 MB） | `748411b70089cadf34a6c11054f95f3a454e614af562c23b13a82f6cb413109f` |
| `model.ckpt-1490000_trimed.pt`（声码器 44100） | 135,613,039 B（129.3 MB） | `9410d0b528c10a251ae947bd299d1939b0b3247df680c81c4164e94f5d87dc45` |

> manifest 用 HF 镜像作为 `source`，Zenodo 仅作 `upstream` 备注（R6 的思路在此沿用：不走需确认令牌的网盘）。

**② 缓存目录重定向（R1 定稿）**

- VoiceFixer 把路径硬编码为 `os.path.expanduser("~") + "/.cache/voicefixer/..."`，**无参数可改**。
- **实测：Windows 下 `expanduser("~")` 只认 `USERPROFILE`，设 `HOME` 完全无效**（POSIX 相反）。
  因此 Worker 必须在 `import voicefixer` **之前**设置 `USERPROFILE`（同时设 `HOME` 以兼容 POSIX），
  且**全进程生命周期内保持**——`Config.ckpt` 在**导入期**求值、`VoiceFixer.analysis_module_ckpt` 在**构造期**求值，
  "导入后恢复环境变量"会失效。
- 采用 home = `<repo>/models/cache/voicefixer-home`，权重实际落：

```
models/cache/voicefixer-home/.cache/voicefixer/analysis_module/checkpoints/vf.ckpt
models/cache/voicefixer-home/.cache/voicefixer/synthesis_module/44100/model.ckpt-1490000_trimed.pt
```

  仍在 `models/cache/` 内（已被 .gitignore 忽略），`selfcheck` 只需断言两路径前缀为 `models/cache`。
  副作用：matplotlib 会把 `.matplotlib` 缓存写进该 home，同样被忽略，可接受。

**③ CPU 实测数据**（20 s / 44.1 kHz 单声道样本；60 s 用于观察跨块）

| 项目 | 实测 |
|---|---|
| 模型加载（冷启动，含 466 MB ckpt 读取） | **81.4 s**（磁盘缓存后复测 **2.7–2.9 s**） |
| 模型常驻内存 | **约 1.45 GB RSS**（基线） |
| mode 0 · 20 s | **21.0 s → 0.95× 实时**，峰值 RSS 1.32 GB |
| mode 1 · 20 s | **26.7–35.0 s → 0.57–0.75×**，峰值 RSS 1.33 GB |
| mode 2 · 20 s | **25.6–26.6 s → 0.75–0.78×**，峰值 RSS 1.57 GB |
| 整段 60 s（内部 30 s 分块） | **73.5 s → 0.82× 实时**，峰值 RSS **1.85 GB** |
| 输出采样率 / 时长 | **44.1 kHz**；mode 0/2 **精确等于输入时长**；mode 1 少 336 样本（7.6 ms，`librosa.istft` 截断） |

**④ 分块与取消（R4 / R5 结论，修正 §3.3 的分块假设）**

- `restore_inmem()` **内部已按 30 s 分段**（`seg_length = 44100*30`），但**对外是黑盒、不可打断**。
- 实测"外层自己切 10 s 块逐块调用"：耗时 **13.0 s**（比整段 20 s 的 21.0 s 更快，前向近似超线性），
  峰值 RSS 1.65 GB，**总长度仍精确对齐**。
- ⚠️ **但分块会改变修复结果**：与整段结果的相关性 前 10 s **0.909**、后 10 s **仅 0.197**
  （mode 0，eval 模式、确定性推理）。即"同样的音频，切块与否听感/波形不同"，块边界还会引入电平跳变
  （`restore_inmem` 内部按块做 `max>1.0` 的能量归一化）。
- **阶段 1 设计决定**：为支持取消与内存可控，仍按外层分块（默认 **10 s**），但必须做
  **块间重叠 + 交叉淡化**（overlap-add，重叠 0.3–0.5 s）来掩盖边界差异；进度按块上报，
  取消在块边界生效。块长与重叠系数在阶段 1 实测微调，并在 UI 不暴露该内部参数。

**⑤ 依赖精简（R11）**

- `pip install voicefixer` 会额外拉入 `streamlit` / `pandas` / `scikit-learn`（约 200 MB），
  仅服务于官方 demo。实测推理路径不导入这三者，**卸载后推理结果一致** → `requirements-voicefixer.txt`
  给出"先装 torch(CPU) → `voicefixer --no-deps` → 补核心依赖"的精简安装顺序。

---

## 5. 风险与缺陷预登记表（R1–Rn）

| # | 现象（预期风险） | 根因 | 缓解 / 修复 |
|---|---|---|---|
| **R1** | 权重被下载到 `~/.cache/voicefixer/`，不在 `models/cache/` | VoiceFixer 硬编码缓存路径 | ✅ 阶段 0 已定稿：设 **`USERPROFILE`**（Windows 只认它，`HOME` 无效）指向 `models/cache/voicefixer-home`，且需在 import 前设置并全程保持；纳入 `selfcheck` 断言（§4.1 ②） |
| R2 | 输出被强制成 48 kHz | 复用 FlashSR 的采样率校验 | §3.5：采样率校验按后端区分，VoiceFixer 走 44.1 kHz |
| R3 | 拿去修音乐效果差 / 用户误解 | VoiceFixer 面向**人声** | UI 文案标注"面向人声/语音"；音乐场景引导到 FlashSR/AudioSR |
| R4 | 长音频内存/耗时不可控 | 整段 mel 修复 | 分块处理 + 块间进度/取消（阶段 1 实测块长与重叠） |
| R5 | `restore()` 内部不可打断 → 取消失效 | API 为黑盒 | 按块调用 + 块间检查取消标志（同音频分块 / 图像 tile 思路） |
| R6 | **NuWave2 权重在 Google Drive，下载需确认令牌** | 官方托管方式 | 若实施 NuWave2：改自托管直链/镜像后纳入 manifest；`model_manager` 需支持相应下载方式 |
| R7 | **NuWave2 依赖 pytorch-lightning==1.2.10（2021）** 与现代 torch 不兼容 | 旧版 pin | 若实施：独立旧版 venv；或绕过 lightning 自行加载 checkpoint（阶段 5 评估） |
| R8 | NuWave2 与 AudioSR 功能重叠，投入产出比低 | 都是→48 kHz 超分 | §1.4：列为次优先，阶段 5 先评估差异化价值再决定 |
| R9 | 与现有音频后端互相影响 | 共用 venv / 依赖 | 独立 venv；阶段 0 先验证能否复用，不能则新建 |
| R10 | 修复结果"没变化"或过度平滑 | 生成式修复的固有特性 + mode 选择 | 暴露 mode 0/1/2 供用户切换；默认 mode 0；文档说明各 mode 差异 |
| R11 | 依赖安装失败（voicefixer 的传递依赖） | pip 包依赖链 | `requirements-voicefixer.txt` 锁版本；阶段 0 实测组合 |

---

## 6. 验证

- **单元测试**：`protocol.py` 事件解析、mode 参数校验、fake worker `ready` 探针、`selfcheck` 权重路径断言。
- **集成测试**（真实权重，默认 `#[ignore]`）：
  - 三类退化样本各一：**带噪**、**削波**、**低带宽（如 8 kHz 上采样）**；
  - 校验输出为 **44.1 kHz**、时长与输入一致、可正常播放；
  - 取消：中途取消能在块边界停止、无残留半成品；
  - mode 0/1/2 均可跑通。
- **UI 冒烟**：下拉出现「语音修复（VoiceFixer）」、可选 mode、输出采样率正确、进度可见、可取消、历史可查。
- **回归**：FlashSR / AudioSR / DeepFilterNet / HiFi-GAN 仍可用。

---

## 7. 参考

- NuWave2：https://github.com/maum-ai/nuwave2 （arXiv:2206.08545，**BSD 3-Clause**）
  - Checkpoint（官方 Google Drive，README 内链）；`inference.py -c -i --sr [--steps] [--device cpu]`
- VoiceFixer：https://github.com/haoheliu/voicefixer （arXiv:2109.13731，**MIT**）
  - `pip install voicefixer`；`VoiceFixer().restore(input, output, cuda=False, mode=N)`
  - 权重：`vf.ckpt` + `model.ckpt-1490000_trimed.pt`；默认 `~/.cache/voicefixer/`；有百度网盘镜像
  - 自定义声码器：`your_vocoder_func`（44.1 kHz / 128 mel bins）→ 可与 HiFi-GAN 模块协同
- 本仓库范式：`docs/FlashSR集成实施计划.md`、`docs/HiFi-GAN集成实施计划.md`、`docs/Real-ESRGAN集成实施计划.md`
- 代码位置：`src/QualityView.vue`、`src-tauri/src/audio_quality/{ai_worker,backend}.rs`、`src-tauri/src/history.rs:11-47`
