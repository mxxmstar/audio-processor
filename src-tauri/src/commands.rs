use crate::acoustid;
use crate::error::AppError;
use crate::fingerprint;
use crate::musicbrainz;
use serde::Serialize;
use tauri::AppHandle;
use tauri::Manager;

/// 返回给前端（GUI）的歌曲信息结构。
/// 使用 `serde::Serialize` 以便 Tauri 能把它序列化为 JSON 传给前端。
#[derive(Serialize)]
pub struct SongInfo {
    pub title: String,        // 标题
    pub artist: String,       // 艺术家
    pub album: Option<String>,    // 专辑（可能为空）
    pub album_date: Option<String>, // 专辑发行日期（可能为空）
    pub confidence: f64,      // 识别置信度（百分比，如 100.0）
}

/// Tauri 命令：供前端通过 `invoke('identify', { path })` 调用。
/// 接收音频文件绝对路径，返回识别结果或错误信息字符串。
///
/// 内部网络请求为异步，这里用 Tauri 的异步运行时 `block_on` 在同步命令中等待结果，
/// 使命令签名保持简单（同步），并将命令放在独立模块以规避宏同名冲突问题。
///
/// `app` 用于解析打包后随附的 `fpcalc.exe` 资源路径。
#[tauri::command]
pub fn identify(app: AppHandle, path: String) -> Result<SongInfo, String> {
    // 解析随附的 fpcalc 工具路径（开发期与打包后位置不同）
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("获取资源目录失败: {e}"))?;
    let mut fpcalc_path = resource_dir.join("fpcalc.exe");
    // 开发期（tauri dev）资源目录可能不含 fpcalc，回退到仓库 bin 目录
    if !fpcalc_path.exists() {
        fpcalc_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("bin")
            .join("fpcalc.exe");
    }
    let fpcalc_path = fpcalc_path.to_string_lossy().to_string();

    tauri::async_runtime::block_on(async move {
        run_identify(&fpcalc_path, &path)
            .await
            .map_err(|e| e.to_string())
    })
}

/// 不依赖 Tauri 上下文的识别入口（供命令与测试/示例共用）。
///
/// `fpcalc_path` 为 fpcalc 可执行文件绝对路径；`path` 为待识别音频路径。
pub async fn run_identify(fpcalc_path: &str, path: &str) -> Result<SongInfo, AppError> {
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
