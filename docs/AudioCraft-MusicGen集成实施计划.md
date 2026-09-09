# AI 音乐生成模块 · AudioCraft (Meta) & MusicGen-Remix 集成实施计划

> 状态：规划中（待评审）——**实施前必须先完成 §1.5「前置门禁」的法律与硬件决策**
> 编制日期：2026-09-09
> 目标：新增 **AI 音乐生成**模块，接入 Meta AudioCraft（MusicGen 系列）作为首个生成式后端，
> 并规划 **Remix（基于已有旋律/音频再创作）** 能力。

---

## 0. 阅读提示（与既有计划的关系）

| 文档 | 贡献 |
|---|---|
| `docs/FlashSR集成实施计划.md` | 确立"新后端接入 JSONL Worker 协议"范式（独立 venv / WorkerSpec 路由 / manifest 条目 / D-series 缺陷登记） |
| `docs/HiFi-GAN集成实施计划.md` | 前端菜单位置口径（独立子菜单、不嵌套进已有视图）；定位纪律 |
| `docs/Real-ESRGAN集成实施计划.md` | 把范式**横向复制到新领域**的做法（图像域），以及"新菜单组 + 独立视图 + 独立状态"的解耦方式 |
| 本文档 | 复制到**生成式音频域**；并首次引入**授权合规**与**硬件可行性**门禁 |

> ⚠️ **本文档与前几份最大的不同**：前几个模型是"增强/重建"，本模块是**生成（create）**，
> 且 Meta 权重为 **CC-BY-NC 4.0（非商业）**、官方要求 **GPU**。这两点决定"能不能做"，
> 必须在阶段 0 之前决策，而不是边做边看。

---

## 1. 背景与问题

### 1.1 为什么引入

应用已覆盖音频识别、音频品质提升（FlashSR / AudioSR / DeepFilterNet / 规划中 HiFi-GAN），
并规划了图像画质提升（Real-ESRGAN）。用户侧存在**创作类**诉求：
从文字描述生成音乐（BGM / 配乐 / 灵感草稿），以及对已有旋律做**再创作（Remix）**。
AudioCraft 是该领域最成熟、生态最好的开源方案。

### 1.2 AudioCraft 是什么

AudioCraft 是 Meta (FAIR) 的 PyTorch 音频生成库，包含（官方 README 模型清单）：

| 模型 | 能力 |
|---|---|
| **MusicGen** | 文本→音乐；单阶段自回归 Transformer，32 kHz EnCodec（4 codebook @50 Hz，50 AR 步/秒） |
| **AudioGen** | 文本→音效 |
| **EnCodec** | 高保真神经音频编解码器 |
| **Multi Band Diffusion** | EnCodec 兼容的扩散解码器 |
| **MAGNeT** | 非自回归文本→音乐/音效 |
| **AudioSeal** | 音频水印 |
| **MusicGen Style** | 文本 + 风格→音乐 |
| **JASCO** | 以和弦 / 旋律 / 鼓点为条件的高质量文本→音乐 |

**MusicGen 官方预训练模型（10 个）**：

| 模型 | 参数 | 能力 |
|---|---|---|
| `facebook/musicgen-small` | 300M | 文本→音乐 |
| `facebook/musicgen-medium` | 1.5B | 文本→音乐 |
| `facebook/musicgen-melody` | 1.5B | 文本→音乐 **+ 文本&旋律→音乐** |
| `facebook/musicgen-large` | 3.3B | 文本→音乐 |
| `facebook/musicgen-melody-large` | 3.3B | 文本 + 旋律 |
| `facebook/musicgen-stereo-*` | 各规格 | 上述模型的**立体声**微调版 |

官方建议：质量/算力最佳权衡为 `musicgen-medium` 或 `musicgen-melody`。

### 1.3 关于 "MusicGen-Remix" 的澄清（重要，避免按错误对象立项）

**"MusicGen-Remix" 并不在 AudioCraft 官方预训练模型清单中。** 需要按以下理解立项：

| 说法 | 实际情况 | 建议 |
|---|---|---|
| 官方 Remix 模型 | ❌ 不存在 | 不要按"官方 remix 权重"设计 |
| **官方最接近 Remix 的能力** | ✅ `musicgen-melody` / `-melody-large` 的**旋律（chroma）条件生成**：`model.generate_with_chroma(descriptions, melody, sr)` —— 以已有音频的旋律为条件 + 文本描述 → 生成"再创作"版本 | **作为 Remix 的官方实现路径（推荐）** |
| 官方进阶条件控制 | ✅ `MusicGen Style`（文本+风格）、`JASCO`（和弦/旋律/鼓点条件） | 后续增强 |
| 社区 Remixer | ⚠️ 如 `cog-musicgen-remixer`（第三方，基于 MusicGen-Chord / MusicGenMelody 改版） | **不建议**作为首版依赖（授权、维护、权重来源均不受控） |

→ **结论**：Remix 能力以 `musicgen-melody` 的 chroma 条件生成为官方实现；
若用户确有"社区 Remixer"诉求，单独立项评估（见 R4）。

### 1.4 定位说明（重要，避免产品口径错误）

- 本模块是**生成（generation）**，不是"音质提升/增强"。生成结果是**全新作品**，
  不是对输入音频的修复或还原。
- MusicGen 输出为 **32 kHz**（与音质提升链路的 48 kHz 不同），首版**不做重采样**，按原始 32 kHz 输出。
- 受权重许可限制（§1.5），**不可对外宣称"商用可用"**，且 UI 应标注模型许可。
- 前端与历史记录显示**实际后端名**（如 `musicgen-melody`），沿用既有命名纪律。

### 1.5 前置门禁（阶段 0 之前必须决策，否则不要开工）

| # | 门禁 | 依据 | 不通过的后果 |
|---|---|---|---|
| **G1 · 授权** | AudioCraft **代码 MIT**，但**模型权重 CC-BY-NC 4.0（非商业）** | 官方 README License 段 | 若本产品用于商业用途，**不得分发/内置 MusicGen 权重**；需改选可商用模型（如 MAGNeT 同为 NC？需逐个核实）或放弃 |
| **G2 · 硬件** | 官方：本地推理**必须有 GPU**；medium(1.5B) 建议 **≥16 GB 显存**；小显存可用 `small` 生成短序列 | 官方 MUSICGEN.md | 纯 CPU 环境生成 30 s 音乐将慢到不可接受（分钟~数十分钟级），需限制时长/仅 small/或标注"需 GPU" |
| **G3 · Python 版本** | AudioCraft 要求 **Python 3.9 + PyTorch 2.1.0** | 官方 README | 与现有 `.venv`/`.venv-flashsr`（Python 3.10）冲突，**必须新建独立 Python 3.9 环境**，不能复用 |

> 建议：G1 由产品/法务确认；G2 由实测确认（阶段 0 给出 CPU/GPU 实测耗时）；G3 由工程确认安装方案。

### 1.6 目标与非目标

**目标**

1. 新增 **AI 音乐生成**模块（Rust 模块 + Python Worker + 前端菜单），与音频增强模块解耦。
2. 首发支持：文本→音乐（`musicgen-small` / `musicgen-medium`）+ **Remix（旋律条件，`musicgen-melody`）**。
3. 走现有 JSONL Worker 协议，支持进度、取消、模型安装、历史记录。
4. 不破坏现有音频增强链路与图像模块。

**非目标**

- 首版**不做**训练/微调、批量队列、音频水印（AudioSeal）、音效生成（AudioGen）。
- 不做"自动作曲/歌词"等上层创作功能。
- 不引入社区第三方 Remixer（见 §1.3 / R4）。
- 不处理授权合规本身（需产品/法务决策，工程只负责标注与提示）。

---

## 2. 现状梳理

### 2.1 生成侧现状：完全空白

| 层面 | 现状 |
|---|---|
| 前端 | 无任何音乐生成相关代码 |
| Rust | 无生成模块（仅 `audio_quality`） |
| Python | 仅 `audio_ai*`；无生成 worker |
| manifest | 仅音频增强模型条目 |
| 历史 `HistoryKind` | 仅 `Recognize` / `Download` / `Enhance` |

### 2.2 可复用的既有资产

| 资产 | 位置 | 复用方式 |
|---|---|---|
| JSONL Worker 协议 v1 | `src-tauri/src/audio_quality/ai_worker.rs` | 复制为 `music_gen/ai_worker.rs` |
| 后端路由范式 | `audio_quality/backend.rs` | 镜像为 `music_gen/backend.rs`（`musicgen` 后端） |
| Python 分层模板 | `python/audio_ai_hifigan/` | **照搬**（protocol / pipeline / worker / fake_worker / selfcheck / tests） |
| 下载 + SHA-256 | `python/audio_ai/model_manager.py` | 新增生成模型分支 |
| 历史记录 | `src-tauri/src/history.rs` | `kind` 为 TEXT 列，**新增 kind 无需迁移** |
| ffmpeg | 应用已带 `bin/ffmpeg.exe`（`biliapi::media`） | AudioCraft 官方也建议有 ffmpeg，可直接复用 |
| 命令注册 | `src-tauri/src/lib.rs:50-93` | 新增 `pub mod music_gen;` + 命令 |

### 2.3 生成 vs 增强：范式差异（决定设计取舍）

| 维度 | 增强类（FlashSR/Real-ESRGAN） | **生成类（MusicGen）** |
|---|---|---|
| 输入 | 已有媒体文件 | **文本提示词 +（可选）旋律音频** |
| 输出 | 与输入同源的"更好版本" | **全新作品** |
| 时长 | 由输入决定 | **由 `duration` 参数决定**（新增 UI 参数） |
| 进度来源 | 分块/分 tile | **自回归步数 / 分段生成** |
| 采样率 | 48 kHz（音频）/ 原图 | **32 kHz**（MusicGen 原生） |
| 硬件 | CPU 可跑（慢） | **官方要求 GPU**（R-G2） |
| 许可 | 多为宽松/自训 | **CC-BY-NC（非商业）**（R-G1） |
| 随机性 | 确定性 | **有随机性**（需 seed 参数支持复现） |

---

## 3. 集成设计

### 3.1 运行时选型

| 方案 | 做法 | 依赖 | 可控性 | 结论 |
|---|---|---|---|---|
| **A · 官方 `audiocraft` 库（推荐）** | `pip install -U audiocraft`；`MusicGen.get_pretrained()` | Python 3.9 + torch 2.1.0 + audiocraft（重） | **最好**：官方 API 完整，含 `generate_with_chroma`（Remix 刚需） | ✅ 首版 |
| B · 🤗 Transformers | `MusicgenForConditionalGeneration`（transformers ≥4.31.0） | 依赖较轻 | 中：Remix/melody 条件支持不如官方直接 | 备选（若安装 A 受阻） |
| C · 社区 Remixer | 第三方（如 cog-musicgen-remixer） | 不受控 | 低 | ❌ 不用（R4） |

选 A：Remix 依赖 `generate_with_chroma`，官方库支持最直接。

### 3.2 模块与目录结构（照搬 `audio_ai_hifigan` 分层）

```
python/music_ai/                       # 或 python/audio_ai_musicgen/
  ├ __init__.py
  ├ protocol.py        # JSONL 事件（ready / progress / result / error）
  ├ pipeline.py        # MusicGen 加载 + 分段生成 + Remix(chroma) + 进度/取消
  ├ worker.py          # 主循环
  ├ fake_worker.py     # 协议回归（无需真实权重，极重要——权重数 GB）
  ├ selfcheck.py       # 依赖 / 权重 / CUDA 自检
  └ test_worker.py

requirements-musicgen.txt              # 锁 Python3.9 + torch 2.1.0 + audiocraft
.venv-musicgen/                        # 独立 Python 3.9 环境（★ 与现有 3.10 venv 不同版本）

src-tauri/src/music_gen/
  ├ mod.rs
  ├ ai_worker.rs       # 协议 v1 + WorkerSpec::musicgen() + find_python_musicgen()
  └ backend.rs         # Backend::MusicGen + 路由 + 默认参数（duration/top_k/temperature…）

src-tauri/src/commands/music_gen.rs    # 命令 + MusicGenState（与音频状态解耦）
src/MusicGenView.vue                   # 新视图（提示词 / 时长 / 模型 / Remix 上传）
```

### 3.3 协议与生命周期（沿用 JSONL v1，载荷扩展）

沿用 `process` / `cancel` / `shutdown` 与 `ready` / `progress` / `result` 事件。载荷：

- **请求**：`prompt`（文本）、`duration`（秒）、`model`（如 `musicgen-melody`）、
  `melody_path`（Remix 时）、`seed`、`top_k` / `top_p` / `temperature`、`cfg_coef`。
- **`progress.phase`**：`load_model` → `generate`（按**分段/步数**推进）→ `encode`（写盘）。
- **`result`**：输出路径 + **采样率(32000)** + 声道数 + 时长 + seed + 所用模型。
- **进度与取消（关键设计）**：MusicGen 是自回归生成，单次 `generate()` 内部不可打断。
  → **分段生成**：按目标总时长切成若干段（如每段 5 s，后续段以前一段做 continuation/条件），
  **段与段之间检查取消标志**并上报进度。这样既拿到进度又能协作式取消（思路同音频分块、图像 tile）。
  ⚠️ 分段拼接的听感连贯性需实测（见 R9）。

### 3.4 模型与 manifest

MusicGen 权重体积为 **GB 级**（small < medium < large/melody-large；具体体积**阶段 0 实测回填**），
每个 HF 仓库通常还含 **EnCodec** 压缩模型（`compression_state_dict.bin`）；`*-melody` 另需 **Demucs**（torch.hub 下载）。

```jsonc
{
  "id": "musicgen-melody",              // 或 musicgen-small / musicgen-medium
  "backend": "musicgen",
  "model_name": "facebook/musicgen-melody",
  "file": "cache/musicgen/musicgen-melody/state_dict.bin",
  "extra_files": ["compression_state_dict.bin"],   // EnCodec
  "sha256": "<下载后回填>",
  "source": "https://huggingface.co/facebook/musicgen-melody/resolve/main/state_dict.bin",
  "upstream": "https://github.com/facebookresearch/audiocraft"
}
```

要点：
- **缓存目录可控**：AudioCraft 支持 `AUDIOCRAFT_CACHE_DIR` 环境变量；HF 侧用 `HF_HUB_OFFLINE=1` +
  既有离线约定，确保权重落在 `models/cache/musicgen/...` 而非用户 HOME（沿用应用"显式安装模型"纪律）。
- `*-melody` 还依赖 **Demucs**（走 torch.hub，缓存位置不同），需单独验证并纳入安装流程（R5）。
- 权重 GB 级 → 走**延迟哈希**（`MODEL_HASH_DEFERRED`，同 audiosr-basic 做法），避免启动超时。

### 3.5 前端入口与菜单位置

沿用既有口径：**新菜单组 + 独立子菜单，不嵌套进已有视图**。

```
AI 音乐生成              (父菜单, key: music-group)     ← 新增
  ├ 音乐生成 / Remix      (key: music-gen)              ← 新增，MusicGenView
  └ 历史记录              (key: music-history)          ← 新增，HistoryView kind="music-gen"
```

| 改动点 | 位置 | 做法 |
|---|---|---|
| 菜单组 | `src/App.vue:37-79` `items` | 新增 `music-group` 父项，children 含 `music-gen`、`music-history` |
| 可切换叶子 | `App.vue:82-92` `leafKeys` | 加入 `"music-gen"`、`"music-history"` |
| 视图类型 | `App.vue` 顶部 `ViewKey` | 加入上述 key |
| 视图挂载 | `App.vue:248-281` `<keep-alive>` 区 | 新增 `<music-gen-view v-else-if="active === 'music-gen'" />` 等 |
| 历史标签 | `HistoryView.vue:40-45` `kindLabels` | 加入 `music-gen: "音乐生成"` |
| 后端侧 | `src-tauri/src/history.rs:11-47` | `HistoryKind` 新增 `MusicGen`（`as_str: "music-gen"`、`label: "音乐生成"`），**无需 DB 迁移** |
| 新视图 | 新增 `src/MusicGenView.vue` | 提示词输入、时长、模型选择、**Remix 旋律上传**、seed、生成/取消、试听、输出路径 |

UI 须**标注模型许可（CC-BY-NC 4.0 非商业）**，并提示"生成需 GPU / CPU 较慢"。

### 3.6 命令与状态

`src-tauri/src/commands/music_gen.rs` 提供对称命令：
`music_gen_check_runtime` / `_download_model` / `_start` / `_cancel` / `_list_models` / `_list_tasks`，
持**独立的 `MusicGenState`**；`lib.rs` 追加 `pub mod music_gen;` 并注册命令。

---

## 4. 阶段划分

| 阶段 | 内容 | 交付 / 门禁 |
|---|---|---|
| **0 · 门禁与可行性** | **G1 授权决策（法务/产品）**、**G2 硬件实测**（CPU 与 GPU 各生成 10 s/30 s 的耗时）、**G3 Python 3.9 + torch 2.1.0 + audiocraft 安装验证**；跑通官方 CLI/API 一次文本生成与一次 `generate_with_chroma` | 三门禁结论 + 实测数据回填；**任一门禁不通过则暂停/改方案** |
| **1 · Python 模块** | 照 §3.2 建模块；`pipeline.py` 实现分段生成 + Remix + 进度 + 取消；`fake_worker.py` / `selfcheck.py` / `test_worker.py` | 可 `ready` 握手、fake 可跑 |
| **2 · Rust 侧** | `music_gen/`（ai_worker + backend + mod）、`commands/music_gen.rs`、`lib.rs` 注册、manifest 条目、`find_python_musicgen()` 指向 `.venv-musicgen`（Python 3.9） | 运行时检查可见、可安装 |
| **3 · 前端** | §3.5 菜单组/子菜单/`MusicGenView.vue`；`HistoryKind::MusicGen` | 可输入提示词/上传旋律、可取消、进度可见、可试听 |
| **4 · 测试** | fake worker 协议回归；真实权重生成（短时长，默认 `#[ignore]`）；Remix 用例；UI 冒烟 | 测试通过 |
| **5 · 文档** | 新建/更新音乐生成功能规划档位表；本计划转"实施中/已完成" | 文档同步 |

---

## 5. 风险与缺陷预登记表（R1–Rn）

| # | 现象（预期风险） | 根因 | 缓解 / 修复 |
|---|---|---|---|
| **R1** | **授权不合规（最高优先级）** | MusicGen 权重 **CC-BY-NC 4.0（非商业）** | 阶段 0 前由法务/产品决策（G1）；UI 明确标注；若商用则改用可商用模型或放弃 |
| **R2** | **无 GPU 时慢到不可用** | 自回归生成；官方要求 GPU，medium 建议 ≥16 GB | G2 实测；优先 `small`；限制最大时长；UI 提示"建议 GPU" |
| **R3** | **Python 3.9 与现有 3.10 venv 冲突** | AudioCraft 要求 Python 3.9 + torch 2.1.0 | 新建独立 `.venv-musicgen`（Python 3.9）；`find_python_musicgen()` 单独解析，不复用 `find_python*` |
| **R4** | 按"官方 MusicGen-Remix"立项但实际不存在 | 名称误解 | 已澄清（§1.3）：Remix = `musicgen-melody` 的 chroma 条件生成；社区 Remixer 不作为首版依赖 |
| **R5** | 权重下载/缓存不在预期目录；`*-melody` 额外拉 Demucs | HF/torch.hub 默认缓存到用户 HOME | 设 `AUDIOCRAFT_CACHE_DIR` + `HF_HUB_OFFLINE`；Demucs 纳入安装流程并验证；全部走 manifest + SHA-256 |
| **R6** | 权重数 GB、磁盘与安装耗时 | 模型本身体量 | 延迟哈希；UI 显示下载进度；明确磁盘占用提示 |
| **R7** | 依赖安装失败（torch/xformers 顺序、setuptools/wheel 缺失） | audiocraft 安装较挑剔 | `requirements-musicgen.txt` 锁版本；README 记录安装顺序（先 torch 后其它） |
| **R8** | 输出采样率 32 kHz 与其它模块 48 kHz 不一致 | MusicGen 原生 32 kHz | 首版按 32 kHz 原样输出并在 UI 标明；不擅自重采样 |
| **R9** | **分段生成的听感断裂 / 进度取消实现复杂** | `generate()` 内部不可打断，需自行分段 | 段间做 continuation 条件；实测连贯性；必要时退化为"整段生成 + 仅按步数报进度（牺牲取消粒度）" |
| **R10** | 与音频增强/图像模块互相影响 | 共用 venv / 状态 / 命令 | 独立 venv(3.9)、独立 Rust 模块、独立 `MusicGenState`、独立命令命名空间 |
| **R11** | 生成结果不可复现 / 用户想复现 | 采样随机性 | 暴露 `seed` 参数并记入历史 `payload` |
| **R12** | 提示词注入/不当内容 | 生成模型固有 | 首版做基础提示词长度/字符校验；后续按需要加内容安全策略 |

---

## 6. 验证

- **单元测试**：`protocol.py` 事件解析、分段调度逻辑、fake worker `ready` 探针、`seed` 复现性。
- **集成测试**（真实权重，默认 `#[ignore]`）：
  - 文本→音乐：短时长（如 5 s）生成成功，输出为 32 kHz、时长符合 `duration`；
  - **Remix**：`generate_with_chroma` 以旋律音频为条件生成成功；
  - **取消**：中途取消能在本段边界停止，无残留半成品；
  - 同 `seed` + 同参数 → 输出一致（复现性）。
- **UI 冒烟**：菜单出现「AI 音乐生成 → 音乐生成 / 历史记录」；可输入提示词、选模型、设时长、
  上传旋律、生成、看进度、取消、试听、查历史。
- **回归**：确认音频增强（FlashSR/AudioSR/DeepFilterNet/HiFi-GAN）与图像模块仍可用（零耦合）。

---

## 7. 参考

- 官方仓库：https://github.com/facebookresearch/audiocraft （README：安装 / 模型清单 / License）
- MusicGen 文档：https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md
- 论文：Simple and Controllable Music Generation（arXiv:2306.05284）
- HF 模型：`facebook/musicgen-small` / `-medium` / `-melody` / `-large` / `-melody-large` / `-stereo-*`
- 🤗 Transformers 路径：`MusicgenForConditionalGeneration`（transformers ≥ 4.31.0）
- 本仓库范式：`docs/FlashSR集成实施计划.md`、`docs/HiFi-GAN集成实施计划.md`、`docs/Real-ESRGAN集成实施计划.md`
- 代码位置：`src/App.vue:37-92`、`248-281`；`src-tauri/src/history.rs:11-47`；`python/audio_ai_hifigan/`（结构模板）
