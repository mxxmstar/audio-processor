use serde::Deserialize;
use std::process::Command;

use crate::recognizer::error::{AppError, Result};

/// fpcalc 进程输出的 JSON 结构。
/// 形如：{"duration": 242.04, "fingerprint": "<base64 压缩指纹>"}
#[derive(Deserialize)]
struct FpcalcOutput {
    duration: f64,
    fingerprint: String,
}

/// 调用 Chromaprint 官方工具 `fpcalc` 生成指纹。
///
/// `fpcalc_path` 为 `fpcalc.exe` 的绝对路径（开发期或打包后的资源路径）。
/// 返回 `(时长秒数, 指纹字符串)`，时长取整后用于 AcoustID 匹配，
/// 指纹字符串直接发给 AcoustID 查询。
pub fn compute(fpcalc_path: &str, audio_path: &str) -> Result<(f64, String)> {
    // 调用 fpcalc，输出 JSON 格式（含 duration 与 fingerprint）
    let output = Command::new(fpcalc_path)
        .arg("-json")
        .arg(audio_path)
        .output()
        .map_err(|e| AppError::Fingerprint(format!("无法启动 fpcalc: {e}")))?;

    // fpcalc 非零退出码视为失败
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::Fingerprint(format!("fpcalc 执行失败: {stderr}")));
    }

    // 解析 JSON 输出
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: FpcalcOutput = serde_json::from_str(&stdout)
        .map_err(|e| AppError::Fingerprint(format!("解析 fpcalc 输出失败: {e}；原始输出: {stdout}")))?;

    Ok((parsed.duration, parsed.fingerprint))
}
