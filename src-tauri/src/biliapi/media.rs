//! 音视频合并模块（media）
//!
//! 专门负责调用 ffmpeg 进行音视频合并/转封装，与下载编排（`task` 模块）解耦。
//!
//! 职责：
//! - ffmpeg 二进制探测（PATH / 注入目录 / 开发期 `bin/`）
//! - 注入并读取 ffmpeg 搜索目录（由 Tauri 启动时 `resource_dir/bin` 注入）
//! - 合并音视频流为单个 mp4（默认流复制 `-c copy`，可选重编码）
//! - 合并失败时清理已生成的半成品输出
//!
//! 设计目标：让 `task` 模块只关心"下载什么、下到哪"，合并细节全部收敛到这里。

use crate::biliapi::error::BiliApiError;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// ffmpeg 搜索目录（可由 Tauri 层在启动时注入）。
/// `Some(None)` 表示已显式重置为"仅探测 PATH / 开发期 `bin/`"；
/// `None` 表示尚未注入，使用默认候选逻辑。
static FFMPEG_DIR: Mutex<Option<Option<PathBuf>>> = Mutex::new(None);

/// 设置 ffmpeg 搜索目录（Tauri 启动时调用，传入 `resource_dir/bin` 或项目 `bin/`）。
/// 传 `None` 恢复为仅探测 PATH / 开发期 `bin/`。
pub fn set_ffmpeg_dir(dir: Option<&Path>) {
    *FFMPEG_DIR.lock().unwrap() = Some(dir.map(|p| p.to_path_buf()));
}

/// 读取当前注入的 ffmpeg 搜索目录（用于调试/日志）。
pub fn ffmpeg_dir() -> Option<PathBuf> {
    FFMPEG_DIR.lock().unwrap().clone().flatten()
}

/// 返回 ffmpeg 候选路径列表（按优先级）：
/// 1. PATH 中的 `ffmpeg` / `ffmpeg.exe`
/// 2. 注入的搜索目录（可能直接放 ffmpeg，也可能放在 `bin/` 子目录）
/// 3. 开发期 `CARGO_MANIFEST_DIR/../bin`（编译时常量）
fn ffmpeg_candidates() -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from("ffmpeg"), PathBuf::from("ffmpeg.exe")];
    if let Some(Some(d)) = FFMPEG_DIR.lock().unwrap().clone() {
        // 注入目录里 ffmpeg 可能以多种方式摆放，全部纳入候选
        v.push(d.join("ffmpeg.exe"));
        v.push(d.join("ffmpeg"));
        v.push(d.join("bin/ffmpeg.exe"));
        v.push(d.join("bin/ffmpeg"));
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let base = PathBuf::from(manifest);
        v.push(base.join("../bin/ffmpeg.exe"));
        v.push(base.join("../bin/ffmpeg"));
    }
    v
}

/// 探测单个候选路径是否真实可用（能跑 `-version` 且退出码为 0）。
fn candidate_works(p: &Path) -> bool {
    std::process::Command::new(p)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 检查 ffmpeg 是否可用（PATH 或注入/开发期 `bin/ffmpeg`）。
pub fn ffmpeg_available() -> bool {
    ffmpeg_candidates().iter().any(|p| candidate_works(p))
}

/// 返回首个可用的 ffmpeg 路径；都不存在时返回 `None`。
pub fn find_ffmpeg() -> Option<PathBuf> {
    ffmpeg_candidates().into_iter().find(|p| candidate_works(p))
}

/// 合并选项
#[derive(Debug, Clone, Default)]
pub struct MergeOptions {
    /// 音频/视频流复制（`copy`，默认，最快且不损画质）。
    /// 若设为 `Some("aac")` / `Some("libx264")` 等则重编码（适用于流不兼容时）。
    pub codec: Option<String>,
    /// 额外 ffmpeg 参数（如 `["-movflags", "+faststart"]`）。
    pub extra_args: Vec<String>,
}

/// 用 ffmpeg 将音视频合并为单个文件。
///
/// - `video`：视频输入路径
/// - `audio`：音频输入路径
/// - `output`：输出路径（合并结果）
/// - `options`：合并选项（默认流复制）
///
/// 合并失败时若已生成 `output` 会尝试删除，避免残留半成品。
pub fn merge(video: &str, audio: &str, output: &str) -> Result<(), BiliApiError> {
    merge_with_options(video, audio, output, &MergeOptions::default())
}

/// 同 [`merge`]，但允许自定义合并选项。
pub fn merge_with_options(
    video: &str,
    audio: &str,
    output: &str,
    options: &MergeOptions,
) -> Result<(), BiliApiError> {
    let bin = find_ffmpeg()
        .ok_or_else(|| BiliApiError::Other("未找到可用的 ffmpeg，无法合并音视频（请安装 ffmpeg 或放入 bin/ 目录）".into()))?;

    let codec = options.codec.as_deref().unwrap_or("copy");
    let mut cmd = std::process::Command::new(&bin);
    cmd.args(["-y", "-i", video, "-i", audio, "-c", codec, output]);
    for a in &options.extra_args {
        cmd.arg(a);
    }

    let status = cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| BiliApiError::Other(format!("调用 ffmpeg 失败: {}", e)))?;

    if !status.status.success() {
        // 合并失败：清理可能已生成的半成品输出，避免误导用户
        let _ = std::fs::remove_file(output);
        let stderr = String::from_utf8_lossy(&status.stderr);
        return Err(BiliApiError::Other(format!(
            "ffmpeg 合并失败（退出码 {:?}）：{}",
            status.status.code(),
            stderr.lines().last().unwrap_or("<无 stderr>")
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidates_non_empty() {
        // 至少包含 PATH 中的 ffmpeg / ffmpeg.exe
        assert!(!ffmpeg_candidates().is_empty());
    }

    #[test]
    fn test_merge_options_default_is_copy() {
        let o = MergeOptions::default();
        assert_eq!(o.codec.as_deref(), None);
    }
}
