# audio-processor 实施文档（Rust + Tauri 重写方案）

本文档定义将当前 Python 版音频指纹识别工具重写为 **Rust 后端 + Tauri 桌面界面** 的实施计划。
目标与最初规划一致：用 Tauri 做 GUI，后端用 Rust 实现识别逻辑，最终产出单文件桌面应用。

---

## 0. 背景与现状

当前仓库是一个 **Python 原型**（已能跑通），核心文件：

| 文件 | 作用 |
|------|------|
| `recognizer.py` | 主逻辑：调 `pyacoustid` 生成指纹 → 查 AcoustID → 查 MusicBrainz 补充专辑信息 |
| `requirements.txt` | Python 依赖（pyacoustid / musicbrainzngs / audioread ...） |
| `bin/fpcalc.exe` | Chromaprint 命令行工具（C++ 二进制，用于生成指纹） |
| `DEBUGGING.md` | Python 版调试记录（含 Windows 中文路径踩坑，可作参考） |
| `01. 風の声を聴きながら.mp3` | 测试样本（已纳入 git） |

Python 版已验证完整链路可用（识别「風の声を聴きながら」，置信度 100%）。

### 重写动机

- 当前运行依赖 Python 运行时 + venv + `bin/fpcalc.exe`，分发与部署重
- 遇到 Windows 中文路径 / PowerShell 截断 / fpcalc 子进程乱码等问题（见 `DEBUGGING.md` 问题三）
- Rust + Tauri 可编译为**单文件原生 exe**，无运行时依赖，跨平台，且 Rust 原生 UTF-8 路径处理可彻底规避中文路径坑

---

## 1. 目标架构

```
┌─────────────────────────────────────────────┐
│                  Tauri 前端 (GUI)            │
│   技术: Svelte / Vue / React (任选) + Vite   │
│   职责: 文件选择、结果显示、进度反馈          │
└───────────────┬─────────────────────────────┘
                │ Tauri invoke (JSON)
┌───────────────▼─────────────────────────────┐
│               Rust 后端 (核心)               │
│  ├─ 音频解码  (symphonia)                    │
│  ├─ 指纹生成  (chromaprint crate)            │
│  ├─ AcoustID 查询 (reqwest + serde_json)      │
│  └─ MusicBrainz 查询 (reqwest + serde_json)   │
└───────────────────────────────────────────────┘
```

- **前端**：Tauri 内嵌 WebView，用前端框架做界面；通过 `invoke` 调用 Rust 命令。
- **后端**：Rust 编译进 Tauri 二进制，无独立进程、无 Python 运行时。
- **指纹计算**：用 Rust `chromaprint` crate（Chromaprint 算法的 Rust 绑定），**不再需要 `fpcalc.exe` 外部二进制**。

---

## 2. 技术选型

| 功能 | Python 版 | Rust 方案 | crate / 说明 |
|------|-----------|-----------|--------------|
| 指纹生成 | `pyacoustid` + `fpcalc.exe` | `chromaprint` crate | 纯 Rust 绑定 Chromaprint；构建时可能需要 C 编译器（Chromaprint 底层为 C 库 `libchromaprint`，crate 通过 FFI 链接）。建议用 `chromaprint` 并开启其内建特性，或改用 `chromaprint-sys` + 系统/捆绑的 libchromaprint |
| 音频解码 | `audioread` | `symphonia` | 纯 Rust 媒体解码栈，支持 mp3/flac/m4a 等，无需 ffmpeg |
| AcoustID 查询 | `acoustid.match()` | `reqwest` + `serde_json` | 直接打 `https://api.acoustid.org/v2/lookup`，参数 `client=KEY&duration=&fingerprint=&meta=recordings` |
| MusicBrainz 查询 | `musicbrainzngs.get_recording_by_id` | `reqwest` + `serde` | 直接打 `https://musicbrainz.org/ws/2/recording/{id}?fmt=json&inc=artists+releases`，务必设置 `User-Agent` |
| GUI | （无，纯 CLI） | Tauri | `tauri` + 前端框架；`tauri::command` 暴露识别接口 |
| 异步 | — | `tokio` | Tauri 默认异步 runtime，HTTP 请求用 async |

> 注：`musicbrainzngs` 要求客户端自报 `User-Agent`（app + 版本 + 联系方式）。Rust 侧用 `reqwest` 的 `header` 显式设置，否则会被 MusicBrainz 拒绝。

### 依赖清单（拟，`Cargo.toml`）

```toml
[dependencies]
tauri = { version = "2", features = [] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["full"] }
symphonia = { version = "0.5", features = ["mp3", "flac", "wav"] }
chromaprint = "0.2"
```

> 前端构建依赖（任选其一）：`npm` + `vite` + 所选框架。Tauri 2 默认用 Vite 模板。

---

## 3. 目标目录结构

```
audio-processor/
├── src-tauri/                # Rust 后端 + Tauri 配置
│   ├── Cargo.toml
│   ├── tauri.conf.json       # Tauri 窗口/构建配置
│   ├── build.rs
│   └── src/
│       ├── main.rs           # Tauri 入口，注册命令
│       ├── fingerprint.rs    # 音频解码 + chromaprint 指纹
│       ├── acoustid.rs       # AcoustID 查询
│       └── musicbrainz.rs    # MusicBrainz 查询
├── src/                      # 前端（Vue/Svelte/React）
│   ├── App.vue (或 .svelte/.tsx)
│   └── main.ts
├── public/
├── package.json
├── vite.config.ts
├── Cargo.toml                # workspace 根（可选）
├── IMPLEMENTATION.md         # 本文档
└── tests/                    # 测试音频样本（沿用现有 mp3）
    └── 01. 風の声を聴きながら.mp3
```

> 现有 `bin/fpcalc.exe`、`recognizer.py`、`requirements.txt`、`DEBUGGING.md` 在重写完成后可归档或删除（见第 6 节迁移步骤）。

---

## 4. 实施步骤

### 阶段 1：脚手架

1. 安装前置：Rust 工具链（`rustup`）、Node.js + npm、Tauri CLI。
   ```powershell
   rustup update stable
   npm install -g @tauri-apps/cli@latest
   ```
2. 用 Tauri 官方模板初始化（保留当前目录内容，避免覆盖已有 git）：
   ```powershell
   # 在临时目录生成，再合并到当前项目，或手动建立 src-tauri/ 与前端结构
   npm create tauri-app@latest
   ```
3. 确认 `src-tauri/tauri.conf.json` 中 `bundle` 目标包含 `windows`（msi / nsis）。

### 阶段 2：Rust 后端（核心，等价于 `recognizer.py`）

4. `fingerprint.rs`：用 `symphonia` 解码音频为 PCM → 重采样至 Chromaprint 要求的 11025 Hz 单声道 → 调 `chromaprint` 生成指纹字符串。
5. `acoustid.rs`：async 函数 `lookup(key, duration, fingerprint)` → `reqwest::get` 调 AcoustID → 解析 `results[0]` 拿到 `recording_id, title, artist, score`。
6. `musicbrainz.rs`：async 函数 `get_recording(id)` → 调 MusicBrainz REST → 解析 `artist-credit` / `release-list[0]` 拿艺术家与专辑。
7. `main.rs`：用 `#[tauri::command]` 暴露 `identify(path: String) -> Result<SongInfo, String>`，内部串起上面三步（含错误映射）。
8. 设置 `reqwest` 的 `User-Agent`（MusicBrainz 强制要求）。

### 阶段 3：前端 GUI

9. 文件选择：用 Tauri 的 dialog 插件（`@tauri-apps/plugin-dialog`）选音频文件，或直接拖拽。
10. 调用 `invoke('identify', { path })` 显示结果（标题 / 艺术家 / 专辑 / 置信度）。
11. 中文路径：Rust `String` 为 UTF-8，Tauri 跨进程传参不走 shell，天然无 Python 版的乱码/截断问题——这是重写的关键收益之一。

### 阶段 4：构建与分发

12. `npm run tauri build` 产出 `src-tauri/target/release/bundle/` 下的安装包（单个 msi/nsis exe）。
13. 验证：用现有 `01. 風の声を聴きながら.mp3` 跑通，确认识别结果与 Python 版一致（置信度 100%）。

---

## 5. 从 Python 迁移的注意事项

- **API Key**：沿用现有 `ACOUSTID_KEY`（已验证有效），放入 Rust 配置或 `tauri.conf.json` 的 `bundle` 之外的安全处（避免硬编码可放环境变量，但本地工具硬编码亦可）。
- **中文路径**：Python 版需要 `_resolve_path` 模糊匹配 + ASCII 临时副本兜底（见 `DEBUGGING.md`）；Rust 版无需这些 hack。
- **指纹一致性**：`chromaprint` crate 与 `fpcalc.exe` 同源于 Chromaprint，理论上指纹一致；上线前用同一音频对比两边 `lookup` 结果。
- **删除 fpcalc**：Rust 版用 crate 内建指纹计算，不再需要 `bin/fpcalc.exe`，可移除该依赖与 `.gitignore` 中相关条目。

---

## 6. 迁移 / 清理步骤（完成后执行）

1. 确认 Rust 版用测试音频验证通过。
2. 归档旧 Python 文件：`recognizer.py`、`requirements.txt`、`DEBUGGING.md`、`bin/` 可移入 `legacy/` 或直接删除（保留 `DEBUGGING.md` 作历史参考亦可）。
3. 更新 `.gitignore`：移除 `.venv/`、`*.mp3`（样本按需保留）、`bin/ffmpeg.exe` 等 Python 相关条目；改为忽略 `src-tauri/target/`、`node_modules/`。
4. 提交新结构到 git（远程 `origin` 已配置为 `https://github.com/mxxmstar/audio-processor`，分支 `main`）。

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| `chromaprint` crate 维护不活跃（2022 后少更新） | 算法稳定；必要时可改用 `libchromaprint-sys` + 自写 FFI，或 fork 维护 |
| Chromaprint C 库链接需 C 编译器 | Windows 上确保装了 MSVC/LLVM 构建工具；或用预编译 `libchromaprint` 静态链接 |
| Tauri 2 前端模板改动 | 先用官方模板跑通空窗，再逐步加逻辑，降低排查成本 |
| MusicBrainz 限流（匿名调用） | 设置合规 `User-Agent`，控制请求频率 |

---

## 8. 验收标准

- [ ] `tauri build` 产出可在 Windows 直接安装的 exe
- [ ] GUI 选同一测试音频 → 识别「風の声を聴きながら / 三月のパンタシア」，置信度约 100%
- [ ] 含中文/日文文件名的音频可直接识别（无需临时副本）
- [ ] 不再依赖 Python 运行时与 `fpcalc.exe`
- [ ] 代码提交至 `main` 分支并推送到 GitHub
