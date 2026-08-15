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
    let dir_str = dir.map(|p| p.to_string_lossy().into_owned());
    println!("[media] set_ffmpeg_dir -> {:?}", dir_str);
    *FFMPEG_DIR.lock().unwrap() = Some(dir.map(|p| p.to_path_buf()));
}

/// 读取当前注入的 ffmpeg 搜索目录（用于调试/日志）。
pub fn ffmpeg_dir() -> Option<PathBuf> {
    FFMPEG_DIR.lock().unwrap().clone().flatten()
}

/// 从 `start` 目录开始，逐级向上回溯，把每一级下的 `bin/ffmpeg.exe` 与 `bin/ffmpeg`
/// 加入候选（直到文件系统根）。这样无论 exe 嵌套多深，都能命中项目根 `bin/ffmpeg.exe`。
fn walk_up_bin(mut dir: PathBuf, v: &mut Vec<PathBuf>) {
    loop {
        v.push(dir.join("bin").join("ffmpeg.exe"));
        v.push(dir.join("bin").join("ffmpeg"));
        // 也兼容直接放在该目录（打包后 resource_dir 下的 ffmpeg.exe）
        v.push(dir.join("ffmpeg.exe"));
        v.push(dir.join("ffmpeg"));
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
}

/// 返回 ffmpeg 候选路径列表（按优先级）：
/// 1. PATH 中的 `ffmpeg` / `ffmpeg.exe`
/// 2. 注入的搜索目录（Tauri setup 注入的 `resource_dir/bin` 等）
/// 3. 从可执行文件所在目录逐级向上回溯 `bin/ffmpeg.exe`（覆盖 dev / 打包各种结构）
/// 4. 当前工作目录逐级向上回溯 `bin/ffmpeg.exe`
/// 5. 编译期 `CARGO_MANIFEST_DIR/../bin`（仅编译期有效，作为兜底）
fn ffmpeg_candidates() -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from("ffmpeg"), PathBuf::from("ffmpeg.exe")];

    // 注入的搜索目录（Tauri setup 注入的 resource_dir/bin 或项目 bin）
    if let Some(Some(d)) = FFMPEG_DIR.lock().unwrap().clone() {
        walk_up_bin(d, &mut v);
    }

    // 从可执行文件所在目录逐级向上回溯（dev: target/debug -> ../../.. -> 项目根 bin）
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            walk_up_bin(parent.to_path_buf(), &mut v);
        }
    }

    // 从当前工作目录逐级向上回溯
    if let Ok(cwd) = std::env::current_dir() {
        walk_up_bin(cwd, &mut v);
    }

    // 编译期常量（仅 `cargo test` 或编译期调用有效）
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let base = PathBuf::from(manifest);
        v.push(base.join("..").join("bin").join("ffmpeg.exe"));
        v.push(base.join("..").join("bin").join("ffmpeg"));
    }
    v
}

/// 去掉 Windows 的 `\\?\` 长路径前缀。
/// 带该前缀启动可执行文件会导致其依赖 DLL 加载失败
/// （表现为退出码 -1073741515 / STATUS_DLL_NOT_FOUND），因此探测前必须去除。
fn strip_verbatim(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    let s = s.strip_prefix("\\\\?\\").unwrap_or(&s);
    PathBuf::from(s)
}

/// 探测单个候选路径是否真实可用（能跑 `-version` 且退出码为 0）。
fn candidate_works(p: &Path) -> bool {
    let cleaned = strip_verbatim(p);
    match std::process::Command::new(&cleaned)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
    {
        Ok(status) => status.success(),
        Err(e) => {
            // 仅在真正启动失败时提示，便于后续排查（如缺少依赖 DLL）
            println!("[media] ffmpeg 候选 {} 启动失败: {}", cleaned.display(), e);
            false
        }
    }
}

/// 检查 ffmpeg 是否可用（PATH 或注入/开发期 `bin/ffmpeg`）。
pub fn ffmpeg_available() -> bool {
    let ok = find_ffmpeg().is_some();
    println!("[media] ffmpeg_available -> {}", ok);
    ok
}

/// 返回首个可用的 ffmpeg 路径；都不存在时返回 `None`。
pub fn find_ffmpeg() -> Option<PathBuf> {
    let candidates = ffmpeg_candidates();
    let found = candidates.into_iter().find(|p| candidate_works(p));
    match &found {
        Some(p) => println!("[media] find_ffmpeg -> 命中: {}", p.display()),
        None => println!("[media] find_ffmpeg -> 未找到可用 ffmpeg"),
    }
    found
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
    // 去掉 \\?\ 前缀，否则 ffmpeg 无法加载其依赖 DLL
    let bin = strip_verbatim(&bin);

    let codec = options.codec.as_deref().unwrap_or("copy");
    let mut cmd = std::process::Command::new(&bin);
    cmd.args(["-y", "-i", video, "-i", audio, "-c", codec, output]);
    for a in &options.extra_args {
        cmd.arg(a);
    }

    println!(
        "[media] 调用 ffmpeg: {} -y -i {} -i {} -c {} {}",
        bin.display(),
        video,
        audio,
        codec,
        output
    );

    let status = cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| BiliApiError::Other(format!("调用 ffmpeg 失败: {}", e)))?;

    if !status.status.success() {
        // 合并失败：清理可能已生成的半成品输出，避免误导用户
        let _ = std::fs::remove_file(output);
        let stderr = String::from_utf8_lossy(&status.stderr);
        println!(
            "[media] ffmpeg 合并失败，退出码={:?}，stderr 尾部={}",
            status.status.code(),
            stderr.lines().last().unwrap_or("<无 stderr>")
        );
        return Err(BiliApiError::Other(format!(
            "ffmpeg 合并失败（退出码 {:?}）：{}",
            status.status.code(),
            stderr.lines().last().unwrap_or("<无 stderr>")
        )));
    }
    println!("[media] ffmpeg 合并成功 -> {}", output);
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
