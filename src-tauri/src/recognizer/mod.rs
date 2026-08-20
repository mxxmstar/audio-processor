//! 音频识别模块（独立子模块）
//!
//! 该模块封装了"音频指纹识别"的完整链路，与 Tauri 应用框架解耦：
//! - `fingerprint`：调用 fpcalc 生成 Chromaprint 指纹
//! - `acoustid`：用指纹查询 AcoustID，拿到录音 ID / 标题 / 艺术家 / 置信度
//! - `musicbrainz`：用录音 ID 补充专辑信息
//! - `error`：模块统一的错误类型
//!
//! 对外只暴露 `run_identify`（无 Tauri 依赖的识别入口）与 `SongInfo`（返回结构），
//! Tauri 命令层（`crate::commands`）仅负责把前端调用桥接到这里。

// 子模块声明
pub mod acoustid;     // AcoustID 指纹查询
pub mod error;        // 统一错误类型
pub mod fingerprint;  // 音频解码 + Chromaprint 指纹生成
pub mod musicbrainz;  // MusicBrainz 曲目详情查询

// 重导出常用类型，方便命令层直接 `use crate::recognizer::*` 取到
pub use error::{AppError, Result};
pub use fingerprint::compute;

use serde::Serialize;

/// 返回给前端（GUI）的歌曲信息结构。
/// 使用 `serde::Serialize` 以便 Tauri 能把它序列化为 JSON 传给前端。
#[derive(Serialize)]
pub struct SongInfo {
    pub title: String,            // 标题
    pub artist: String,           // 艺术家
    pub album: Option<String>,    // 专辑（可能为空）
    pub album_date: Option<String>, // 专辑发行日期（可能为空）
    pub confidence: f64,          // 识别置信度（百分比，如 100.0）
}

/// 音频识别流程入口（不依赖 Tauri 上下文，供命令层与测试/示例共用）。
///
/// `fpcalc_path` 为 fpcalc 可执行文件绝对路径；`path` 为待识别音频路径。
/// 内部依次执行：指纹生成 → AcoustID 查询 → MusicBrainz 补充专辑信息。
pub async fn run_identify(fpcalc_path: &str, path: &str) -> Result<SongInfo> {
    // 第 1 步：调用 fpcalc 生成 Chromaprint 指纹（含时长，单位秒）
    let (duration, fingerprint) = fingerprint::compute(fpcalc_path, path)?;

    // 第 2 步：用指纹 + 时长查询 AcoustID，拿到录音 ID、标题、艺术家、置信度
    let acoustid_res = acoustid::lookup(&duration, &fingerprint).await?;

    // 第 3 步：用录音 ID 查询 MusicBrainz，补充专辑信息（失败不影响主结果）
    let (album, album_date) = match musicbrainz::get_recording(&acoustid_res.recording_id).await {
        Ok(info) => (info.album, info.album_date),
        Err(_) => (None, None),
    };

    // 组装最终返回结构
    Ok(SongInfo {
        title: acoustid_res.title,
        artist: acoustid_res.artist,
        album,
        album_date,
        confidence: acoustid_res.confidence,
    })
}
