# Audio Processor — Tauri 重构说明

本项目已用 **Tauri 2 + Rust + Svelte** 重写，替代原先基于 Python 的命令行方案。
桌面应用界面（Svelte）调用 Rust 后端命令完成音频指纹识别，识别链路为：

```
音频文件 → fpcalc (Chromaprint 指纹) → AcoustID (查录音 ID/标题/艺术家) → MusicBrainz (补专辑信息)
```

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 前端 | Svelte 5 + Vite | 文件选择 + 结果展示，端口 23334（dev）/ 23335（preview） |
| 后端 | Rust + Tauri 2 | 提供 `identify` 命令，编排指纹与网络查询 |
| 指纹 | `fpcalc.exe` | Chromaprint 官方工具，打包为 Tauri 资源随附 |
| 查询 | `reqwest` + `tokio` | 异步请求 AcoustID / MusicBrainz（均免费、无需密钥授权） |

## 目录结构

```
audio-processor/
├── src/                    # 前端 Svelte 源码（main.ts / App.svelte / app.css）
├── src-tauri/
│   ├── Cargo.toml         # Rust 依赖
│   ├── tauri.conf.json    # Tauri 配置（窗口、打包、fpcalc 资源）
│   ├── icons/             # 应用图标（由 tauri icon 生成）
│   └── src/
│       ├── main.rs        # 程序入口
│       ├── lib.rs         # Tauri 运行时构建与命令注册
│       ├── commands.rs    # `identify` 命令（核心流程编排）
│       ├── fingerprint.rs # 调用 fpcalc 生成指纹
│       ├── acoustid.rs    # AcoustID 查询
│       ├── musicbrainz.rs # MusicBrainz 查询（补专辑）
│       └── error.rs       # 统一错误类型
├── bin/fpcalc.exe         # Chromaprint 指纹工具（原项目依赖，继续沿用）
└── 01. 風の声を聴きながら.mp3  # 测试样本
```

## 关键实现要点

1. **指纹生成**：直接调用 `fpcalc.exe`（位于 `bin/`，打包后复制到资源目录 `fpcalc.exe`），
   解析其 `-json` 输出得到 `duration` 与 `fingerprint`。
2. **AcoustID 查询**：
   - `duration` 必须为**整数秒**（带小数服务端会误报缺参数）。
   - 使用 POST 表单提交，`meta` 用空格分隔（`recordings artists`）以正确展开录音信息。
   - 录音 ID 优先取 `recordings[].id`，未展开时回退到 `result.id`。
3. **MusicBrainz 查询**：需设置 `User-Agent`，按 `recording/{id}?inc=artists+releases` 取专辑。
4. **命令拆分**：`identify` 命令放在独立 `commands` 模块，避免与 `generate_handler!` 宏同名冲突。

## 常用命令

```bash
npm install            # 安装前端依赖
npm run dev            # 仅启动前端（Vite，端口 23334）
npm run tauri dev      # 启动完整桌面应用（开发模式）
npm run tauri build    # 打包为 MSI / NSIS 安装包
```

## 配置说明（端口）

- 前端 dev server：`vite.config.ts` 中 `server.port = 23334`，`preview.port = 23335`
- Tauri dev URL：`tauri.conf.json` 中 `build.devUrl = http://localhost:23334`
