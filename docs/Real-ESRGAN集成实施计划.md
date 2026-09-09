# 图像画质提升模块 · Real-ESRGAN 集成实施计划

> 状态：实施中（**阶段 0 完成**：独立 `.venv-realesrgan` 已建、依赖锁定、`requirements-realesrgan.txt` 已落地；官方 `RealESRGANer` 推理路径跑通小图 4× 与大图内部分块，权重体积/SHA-256/直连下载均已实测回填）
> 编制日期：2026-09-09
> 目标：新增**图像画质提升**模块，并率先接入 **Real-ESRGAN** 作为首个超分辨率后端，
> 复用音频侧已跑通的「Python Worker + JSONL 协议 + manifest + 独立 venv」范式。

---

## 0. 阅读提示（与既有计划的关系）

| 文档 | 贡献 |
|---|---|
| `docs/FlashSR集成实施计划.md` | 确立"新模型后端接入现有 Worker 协议"的完整范式（独立 venv / WorkerSpec 路由 / manifest 条目 / UI 档位 / D-series 缺陷登记） |
| `docs/HiFi-GAN集成实施计划.md` | 确立**前端菜单位置口径**（独立子菜单，不嵌套进已有视图）与"声码器≠超分"的定位纪律 |
| 本文档 | 把上述范式**横向复制到图像域**，并处理图像特有的分块/内存/色彩/格式差异 |

重大利好：`python/audio_ai_hifigan/` 已形成清晰的分层结构
（`protocol.py` / `pipeline.py` / `worker.py` / `vendor_bridge.py` / `selfcheck.py` / `fake_worker.py`），
比早期 `audio_ai_flashsr` 的单体 worker 更清晰，**应作为图像模块的结构模板**。

---

## 1. 背景与问题

### 1.1 为什么做图像画质提升

本应用已覆盖"音频品质提升"（FlashSR / AudioSR / DeepFilterNet / 规划中的 HiFi-GAN）。
用户侧存在同源诉求：对**低清/压缩/缩放过的图片**做画质增强（超分辨率）。
Real-ESRGAN 是该类任务的事实标准开源方案，作为首个后端风险最低、资料最全。

### 1.2 Real-ESRGAN 是什么

Real-ESRGAN（Wang et al., 2021，*Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*）
是 ESRGAN 的实用化升级：以 U-Net 判别器 + 纯合成退化数据训练，擅长**真实退化图像**的盲超分。

| 常见官方权重 | 说明 | 体积（阶段 0 实测回填） |
|---|---|---|
| `RealESRGAN_x4plus` | 通用 4×，**首发推荐** | 约 64 MB（**实测 67,040,989 B / 63.9 MB**，见 §3.4） |
| `RealESRGAN_x4plus_anime_6B` | 动漫插画 4×（更轻） | 约 17 MB |
| `realesr-animevideov3` | 动漫/视频 4×（SRVGGNet） | 较小 |
| `RealESRGAN_x2plus` | 通用 2× | 中等 |

> 权重为 **MB 级**，远轻于音频侧的 GB 级（audiosr-basic 6.18 GB / flashsr 3.3 GB），
> 因此**无需延迟哈希校验**，可直接在启动时完成 SHA-256 校验（见 §3.4）。

### 1.3 定位说明（重要，避免产品口径错误）

Real-ESRGAN 是**盲超分辨率**：对已丢失的高频细节做**生成式估计**，不是"还原无损原图"。
动漫/插画模型与通用模型风格差异明显，需按图源选择。

建议产品口径：

| 档位 | 后端 | 说明 |
|---|---|---|
| 通用（默认） | `realesrgan-x4plus` | 照片 / 截图 / 通用图片 |
| 动漫 | （后续）`realesrgan-anime-6b` | 插画 / 动漫图，避免通用模型"油画感" |

前端与历史记录显示**实际后端名**，**不显示"无损还原 / 原画质恢复"**字样
（沿用音频侧 `docs/低质量音频转高品质音频功能规划.md` 的口径纪律）。

### 1.4 目标与非目标

**目标**

1. 新增**图像画质提升**模块（Rust 模块 + Python Worker + 前端菜单），与音频模块解耦、互不干扰。
2. 首个后端 Real-ESRGAN（通用 4×），走现有 JSONL Worker 协议，支持进度、取消、安装、历史记录。
3. 大图安全：按 tile 分块推理，避免 OOM；进度按 tile 上报、取消在 tile 边界生效。
4. 不破坏现有音频链路（FlashSR / AudioSR / DeepFilterNet / HiFi-GAN）。

**非目标**

- 首版**不做**人脸增强（GFPGAN）、视频超分（逐帧处理）、批量队列（可后续补）。
- 不引入模型自动下载（仍须用户显式点击"安装模型"触发）。
- 不做视频音轨/字幕等媒体封装处理。
- 首版不处理 EXIF / ICC 色彩空间的保留（见 R9）。

---

## 2. 现状梳理

### 2.1 图像侧现状：完全空白

| 层面 | 现状 |
|---|---|
| 前端 `src/` | **无任何**图像/画质相关代码（搜索"图像/图片/image/画质"命中 0） |
| Rust 模块 | 仅有 `audio_quality`；无图像模块 |
| Python | 仅 `audio_ai*` 系列；无图像 worker |
| manifest | 仅音频模型条目 |
| 历史 `HistoryKind` | 仅 `Recognize` / `Download` / `Enhance`（音质提升） |

因此本模块需**从菜单到 Worker 全链路新建**，但可大量复用既有范式。

### 2.2 可复用的既有资产

| 资产 | 位置 | 复用方式 |
|---|---|---|
| JSONL Worker 协议 v1 | `src-tauri/src/audio_quality/ai_worker.rs` | 复制为 `image_quality/ai_worker.rs`（process/cancel/shutdown；ready/progress/result） |
| 后端路由 | `audio_quality/backend.rs`（`Backend` 枚举 + `select_worker_spec`） | 镜像为 `image_quality/backend.rs`（`realesrgan`） |
| WorkerSpec / Python 查找 | `ai_worker.rs:190-260`、`804-843` | 新增 `WorkerSpec::realesrgan()` + `find_python_image()`（`.venv-realesrgan`） |
| Python 分层模板 | `python/audio_ai_hifigan/` | **直接照搬目录结构**（protocol/pipeline/worker/fake_worker/selfcheck/test） |
| 模型清单 | `models/manifest.json` | 新增 `realesrgan` 后端条目 |
| 下载 + SHA-256 | `python/audio_ai/model_manager.py` | 新增图像分支，复用断点续传 + 校验 |
| 历史记录 | `src-tauri/src/history.rs` | `kind` 为 TEXT 列，**新增 kind 值无需迁移** |
| 命令注册 | `src-tauri/src/lib.rs:50-93` | 新增 `pub mod image_quality;` + 命令 |

### 2.3 关键差异：图像 vs 音频（决定设计取舍）

| 维度 | 音频（既有） | 图像（本模块） |
|---|---|---|
| 解码 | `ffmpeg` / `ffprobe`（`biliapi::media`） | **Pillow / OpenCV**，无需 ffmpeg |
| 分块单位 | 时间：`chunk_seconds` + `overlap_seconds` | **空间：tile + tile_pad**（重叠拼接） |
| 输出 | flac / wav，采样率固定 | **png（默认，无损）/ jpg（可选，带质量参数）** |
| 通道 | 单/双声道 | **RGB / RGBA**（alpha 需专门处理） |
| 权重体积 | GB 级（6.18 GB / 3.3 GB） | **MB 级**（可直接 sha256 校验） |
| 内存峰值 | 音频块小 | **大图 ×4 放大后张量可达数 GB，必须 tiling** |
| 进度 | 按时间块 | **按 tile** |
| 取消 | 块边界 | **tile 边界** |

---

## 3. 集成设计

### 3.1 推理运行时选型（三选一）

| 方案 | 做法 | 体积 | 速度 | 进度/取消可控性 | 结论 |
|---|---|---|---|---|---|
| **A · Python（推荐）** | 新 `.venv-realesrgan`：`torch` + `basicsr` + `realesrgan` + `opencv` + `Pillow` | 大（torch 约 800 MB+） | CPU 慢 / GPU 快 | **最强**（可自研 tile 循环上报进度、协作取消） | ✅ 首版采用 |
| B · `realesrgan-ncnn-vulkan` | 官方预编译 Vulkan 可执行 | 小 | 快（Vulkan GPU） | 弱（黑盒进程，只能解析输出/强杀） | 备选/"轻量模式"后续 |
| C · ONNX Runtime | `onnxruntime` + ONNX 权重 | 中（约 50 MB runtime） | CPU 尚可 | 强 | 进一步瘦身选项 |

**推荐 A 的理由**：本仓库已在音频侧建立成熟范式，A 可完整复用协议 / 进度 / 取消 / manifest /
历史；且 Real-ESRGAN 官方 Python 推理允许**自行切 tile**，从而实现细粒度进度与协作式取消
（B 的黑盒进程做不到）。B 记录为后续"轻量模式"候选（有 Vulkan 的机器速度优势明显）。

### 3.2 模块与目录结构（照搬 `audio_ai_hifigan` 分层）

```
python/image_ai_realesrgan/
  ├ __init__.py
  ├ protocol.py        # JSONL 事件收发（ready / progress / result / error）
  ├ pipeline.py        # Real-ESRGAN 加载 + tile 调度推理（核心）
  ├ worker.py          # 主循环（读 stdin 请求 → 分派 → 回事件）
  ├ fake_worker.py     # 协议回归（无需真实权重）
  ├ selfcheck.py       # 依赖 / 权重自检
  └ test_worker.py     # 单测

requirements-realesrgan.txt      # 新 venv 依赖锁
.venv-realesrgan/                # 独立虚拟环境（与音频 venv 物理隔离）

src-tauri/src/image_quality/
  ├ mod.rs
  ├ ai_worker.rs       # 协议 v1 + WorkerSpec::realesrgan() + find_python_image()
  └ backend.rs         # Backend::RealEsrgan + 路由 + 默认参数

src-tauri/src/commands/image_quality.rs   # 命令 + ImageQualityState（与音频状态解耦）
src/ImageQualityView.vue                  # 新视图
```

### 3.3 协议与生命周期（沿用 JSONL v1，仅换载荷）

沿用 `process` / `cancel` / `shutdown` 与 `ready` / `progress` / `result` 事件，差异仅在载荷：

- **输入**：图片路径（png / jpg / webp / bmp）。
- **输出**：图片路径，**默认 png**（无损；jpg 时附质量参数）。
- **`progress.phase`**：`load_model` → `inference`（按 tile 推进）→ `encode`（写盘）。
- **`result`**：回填**输出宽/高 + 缩放倍数 + 输出格式**（类比音频的采样率/声道/时长）。
- **取消**：每个 tile 处理前检查取消标志 → 协作式退出（与音频块边界一致）。

### 3.4 模型与 manifest

首发只上通用模型，降低验证成本：

```jsonc
{
  "id": "realesrgan-x4plus",
  "backend": "realesrgan",
  "model_name": "RealESRGAN_x4plus",
  "version": "v0.1.0",
  "file": "cache/realesrgan/RealESRGAN_x4plus.pth",
  "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
  "size_bytes": 67040989,
  "source": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
  "upstream": "https://github.com/xinntao/Real-ESRGAN"
}
```

要点：
- 权重 **MB 级**，走常规 SHA-256 校验即可（< `STARTUP_HASH_MAX_BYTES`，无需 `MODEL_HASH_DEFERRED`）。
- 后续加动漫模型只需**新增条目 + 新的 `model_name`**，不动代码结构。
- `backend: "realesrgan"` 供 `read_manifest` 白名单与 Rust `Backend` 枚举识别。

### 3.5 大图分块（tiling）与进度 / 取消 —— 核心技术点

官方 `RealESRGANer.enhance()` 内部虽支持 `tile`，但是**黑盒**：无法上报进度、无法中途取消。
因此**自行实现 tile 调度**（思路同音频的分块处理）：

1. 按 `tile`（默认 512，可按可用内存自适应下调）把输入切成网格。
2. 每块带 `tile_pad`（默认 10）送入模型，输出后**裁掉 pad** 再拼回，避免接缝。
3. 每完成一块 `emit progress(processed_tiles / total_tiles)`。
4. 每块前检查 cancel 标志 → 立即返回，实现协作式取消。
5. **小图（≤ tile）直接整图推理**，省掉 pad/拼接开销。
6. 单线程逐块执行，控制内存峰值；输出张量一次性分配、按块写入。

> 这是与音频侧"分块写盘"同源的设计，也是把 **R2（OOM）** 与 **进度/取消体验** 同时解决的关键。

### 3.6 前端入口与菜单位置

沿用 HiFi-GAN 计划确立的口径：**独立菜单组 + 独立子菜单，不嵌套进已有视图**。

```
图像画质提升            (父菜单, key: image-group)   ← 新增
  ├ 画质提升            (key: image-enhance)        ← 新增，ImageQualityView
  └ 历史记录            (key: image-history)        ← 新增，HistoryView kind="image-enhance"
```

（与音频组 `音频品质提升 → 音质提升 / 历史记录` 完全对称。）

| 改动点 | 位置 | 做法 |
|---|---|---|
| 菜单组 | `src/App.vue:37-79` `items` | 新增 `image-group` 父项（icon 可用 `PictureOutlined`），children 含 `image-enhance`、`image-history` |
| 可切换叶子 | `App.vue:82-92` `leafKeys` | 加入 `"image-enhance"`、`"image-history"` |
| 视图类型 | `App.vue` 顶部 `ViewKey` | 加入上述两个 key |
| 视图挂载 | `App.vue:248-281` `<keep-alive>` 区 | 新增 `<image-quality-view v-else-if="active === 'image-enhance'" />` 与 `<history-view ... kind="image-enhance" />` |
| 历史标签 | `HistoryView.vue:40-45` `kindLabels` | 加入 `image-enhance: "画质提升"` |
| 新视图 | 新增 `src/ImageQualityView.vue` | 以 `QualityView.vue` 为模板，改为选图 / 输出格式 / 缩放倍数 |
| 后端侧 | `src-tauri/src/history.rs:11-47` | `HistoryKind` 新增 `ImageEnhance`（`as_str: "image-enhance"`、`label: "画质提升"`），**无需 DB 迁移**（`kind` 为 TEXT 列） |

### 3.7 命令与状态

`src-tauri/src/commands/image_quality.rs` 提供与音频对称的一组命令：
`image_quality_check_ai_runtime` / `_download_model` / `_start` / `_cancel` / `_list_models` / `_list_tasks`，
并持有**独立的 `ImageQualityState`**（与 `AudioQualityState` 解耦，避免互相影响）。
在 `lib.rs` 追加 `pub mod image_quality;` 并注册上述命令。

---

## 4. 阶段划分

| 阶段 | 内容 | 交付 |
|---|---|---|
| **0 · 选型落地与环境验证** | 建 `.venv-realesrgan` + `requirements-realesrgan.txt`（锁 torch/basicsr/realesrgan 版本）；用官方 CLI 跑通一张小图 4×；确认权重体积/哈希/下载直连可用性 | 环境 OK + 实测数据回填本文档 |
| **1 · Python 模块** | 照 §3.2 建 `python/image_ai_realesrgan/`；`pipeline.py` 实现 §3.5 的 tile 调度 + 进度 + 取消；`fake_worker.py` / `selfcheck.py` / `test_worker.py` 齐备 | 可 `ready` 握手、fake 可跑 |
| **2 · Rust 侧** | `image_quality/`（ai_worker + backend + mod）、`commands/image_quality.rs`、`lib.rs` 注册、`manifest.json` 条目、`find_python_image()` 指向 `.venv-realesrgan` | 运行时检查可见、可安装 |
| **3 · 前端** | 按 §3.6 加菜单组/子菜单/`ImageQualityView.vue`；`HistoryKind::ImageEnhance` | 可选图、可取消、进度可见、历史可查 |
| **4 · 测试** | fake worker 协议回归；真实权重前向（小图 + **大图 tiling**，校验输出尺寸 = 输入 × scale）；UI 冒烟 | 测试通过 |
| **5 · 文档** | 更新 `低质量音频转高品质音频功能规划.md`（或新建图像功能规划）档位表；本计划转"实施中/已完成" | 文档同步 |

---

## 5. 风险与缺陷预登记表（R1–Rn）

> 沿用 FlashSR / HiFi-GAN 的缺陷登记风格；实施中新增缺陷继续追加编号。

| # | 现象（预期风险） | 根因 | 缓解 / 修复 |
|---|---|---|---|
| R1 | `torch` + `basicsr` 安装慢 / 版本冲突（basicsr 对 torch、numpy 版本敏感） | 依赖链重且对版本挑剔 | 独立 `.venv-realesrgan` 与音频 venv 物理隔离；`requirements-realesrgan.txt` 锁版本（阶段 0 实测组合） |
| R2 | **大图 OOM**（4× 放大后张量可达数 GB） | 输出分辨率平方级增长 | §3.5 的 tile 分块 + 自适应 tile 尺寸 + 单线程逐块；必要时限制最大输入边 |
| R3 | RGBA / 灰度 / 16-bit 图处理异常或色彩错乱 | 模型期望 3 通道 RGB | 统一转 RGB；**alpha 通道单独保存并在输出时合并**（或首版要求输入为 RGB 并提示） |
| R4 | CPU 推理慢（大图分钟级），体感"卡死" | 无 GPU / PyTorch CPU 开销 | tile 级细粒度进度 + 预计耗时提示；后续可切 B（ncnn-vulkan）走 GPU |
| R5 | 输出 PNG 体积巨大（4× 面积 → 数倍体积） | 无损格式 + 分辨率提升 | 默认 png 保真，同时提供 jpg（可调质量）选项 |
| R6 | 进度/取消失效（若直接调用官方 `enhance`） | 官方实现是黑盒，无回调 | **必须**自研 tile 循环（§3.5），在 tile 边界上报与检查取消 |
| R7 | 权重下载失败 / 直连 GitHub 慢 | 网络环境 | 走 `model_manager` 断点续传 + SHA-256；必要时提供镜像源并锁 manifest |
| R8 | 与音频模块互相影响 | 共用 venv / 共用状态 | 独立 venv、独立 Rust 模块、独立 `ImageQualityState`、独立命令命名空间 |
| R9 | EXIF / ICC 色彩空间丢失 | 解码-编码链路未保留元数据 | 首版接受并记录；后续用 Pillow 保留/回写 EXIF |
| R10 | 历史记录 kind 扩展 | `HistoryKind` 枚举缺图像类型 | 新增 `ImageEnhance` 枚举值即可，**无需 DB 迁移**（`kind` 为 TEXT 列） |
| R11 | 接缝 / 色块（tiling 痕迹） | pad 不足或拼接裁切错误 | `tile_pad` 默认 10 并做接缝回归用例；提供"整图优先"的小图直通路径 |

---

## 6. 验证

- **单元测试**：tile 切分与拼接（含 pad 裁切边界）、`protocol.py` 事件解析、fake worker `ready` 探针。
- **集成测试**：真实权重前向（默认 `#[ignore]`，同音频 `flashsr_real_worker_forward_pass`）：
  - 小图直通：输出尺寸 == 输入 × scale；
  - **大图 tiling**：走分块路径，校验输出尺寸正确、无接缝（与整图结果做容差对比）；
  - 取消：在中途取消，确认 tile 边界退出、无残留半成品文件。
- **UI 冒烟**：菜单出现「图像画质提升 → 画质提升 / 历史记录」；可选图、安装模型、进度推进、可取消、历史可查。
- **回归**：确认音频侧 FlashSR / AudioSR / DeepFilterNet / HiFi-GAN 仍全部可用（本模块与其零耦合）。

---

## 7. 参考

- `docs/FlashSR集成实施计划.md`（范式来源：阶段划分、缺陷登记、Worker 协议）
- `docs/HiFi-GAN集成实施计划.md`（前端菜单位置口径、独立视图纪律）
- `python/audio_ai_hifigan/`（**新模块结构模板**：protocol / pipeline / worker / fake_worker / selfcheck）
- `src-tauri/src/audio_quality/ai_worker.rs`、`backend.rs`（协议与路由）
- `src-tauri/src/history.rs:11-47`（`HistoryKind`，新增 kind 无需迁移）
- `src/App.vue:37-92`、`248-281`（菜单与视图挂载）
- 上游：https://github.com/xinntao/Real-ESRGAN
