//! 音视频下载编排（阶段 3）
//!
//! 将阶段 1 的解析结果（`resolve_video` / `resolve_collection` / `resolve_season`）
//! 与阶段 2 的流式下载（`HttpClient::download_to_file`）组合为可执行的下载任务。
//!
//! 支持三种模式（对齐 Go 版 download）：
//! - `AudioOnly`：仅下载音频
//! - `VideoOnly`：仅下载视频
//! - `Merge`：下载音 + 视频后用 ffmpeg 合并为 mp4

use crate::biliapi::error::BiliApiError;
use crate::biliapi::video::{PageStream, ResolveResult};
use crate::http_client::client::HttpClient;
use crate::http_client::types::Progress as DlProgress;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// 下载模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DownloadMode {
    /// 仅音频
    AudioOnly,
    /// 仅视频
    VideoOnly,
    /// 音视频合并（需 ffmpeg）
    Merge,
}

/// 任务状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DownloadStatus {
    Pending,
    Downloading,
    Completed,
    Failed,
}

/// 单个下载任务
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DownloadTask {
    /// 任务 ID
    pub id: String,
    /// 展示标题（文件命名用）
    pub title: String,
    /// 视频直链（AudioOnly 模式下可为 None）
    pub video_url: Option<String>,
    /// 音频直链（VideoOnly 模式下可为 None）
    pub audio_url: Option<String>,
    /// 下载模式
    pub mode: DownloadMode,
    /// 输出目录（不含文件名）
    pub output_dir: String,
    /// 当前状态
    pub status: DownloadStatus,
    /// 失败原因（status == Failed 时）
    pub error: Option<String>,
}

impl DownloadTask {
    /// 从解析结果的一个分 P 构造任务
    pub fn from_page(res: &ResolveResult, page: &PageStream, mode: DownloadMode, output_dir: &str) -> Self {
        let id = format!("{}#{}", res.bvid, page.page);
        DownloadTask {
            id,
            title: if page.part.is_empty() {
                format!("{} P{}", res.title, page.page)
            } else {
                format!("{}", page.part)
            },
            video_url: page.video_url.clone(),
            audio_url: page.audio_url.clone(),
            mode,
            output_dir: output_dir.to_string(),
            status: DownloadStatus::Pending,
            error: None,
        }
    }

    /// 将一批解析结果展开为任务列表（每分 P 一个任务）
    pub fn from_resolves(results: &[ResolveResult], mode: DownloadMode, output_dir: &str) -> Vec<Self> {
        let mut tasks = Vec::new();
        for res in results {
            for page in &res.pages {
                tasks.push(Self::from_page(res, page, mode, output_dir));
            }
        }
        tasks
    }
}

/// 安全化文件名（去除文件系统非法字符）
fn sanitize(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars() {
        if c == '\\' || c == '/' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|' {
            s.push('_');
        } else {
            s.push(c);
        }
    }
    s.trim().to_string()
}

/// 检查 ffmpeg 是否可用（在 PATH 或指定路径）
pub fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 用 ffmpeg 合并音视频为 mp4
fn merge_with_ffmpeg(video: &str, audio: &str, output: &str) -> Result<(), BiliApiError> {
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-i", video, "-i", audio, "-c", "copy", output])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| BiliApiError::Other(format!("调用 ffmpeg 失败: {}", e)))?;
    if !status.success() {
        return Err(BiliApiError::Other(format!(
            "ffmpeg 合并失败，退出码 {:?}",
            status.code()
        )));
    }
    Ok(())
}

/// 执行单个下载任务
///
/// - `AudioOnly`：下音频 → `<title>.m4a`
/// - `VideoOnly`：下视频 → `<title>.mp4`
/// - `Merge`：下音 + 视频 → ffmpeg 合并 `<title>.mp4`；若 ffmpeg 不可用则回退为仅下载音/视频并标记提示
pub async fn run_task(
    client: &HttpClient,
    task: &mut DownloadTask,
    on_progress: Option<Arc<dyn Fn(&DownloadTask, DlProgress) + Send + Sync>>,
) -> Result<(), BiliApiError> {
    task.status = DownloadStatus::Downloading;
    let dir = Path::new(&task.output_dir);
    std::fs::create_dir_all(dir)
        .map_err(|e| BiliApiError::Other(format!("创建输出目录失败: {}", e)))?;
    let base = sanitize(&task.title);

    // 进度回调闭包：用 task 快照（Send）包裹，满足 download_to_file 的 Arc<dyn Fn+Send+Sync> 要求
    let snapshot = task.clone();
    let prog_cb: Option<Arc<dyn Fn(DlProgress) + Send + Sync>> = on_progress
        .map(|f| Arc::new(move |p: DlProgress| f(&snapshot, p)) as Arc<dyn Fn(DlProgress) + Send + Sync>);

    match task.mode {
        DownloadMode::AudioOnly => {
            let audio = match &task.audio_url {
                Some(u) => u,
                None => {
                    task.status = DownloadStatus::Failed;
                    task.error = Some("该分 P 无音频直链".into());
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            };
            let out = dir.join(format!("{}.m4a", base));
            client
                .download_to_file(audio, out.to_str().unwrap(), prog_cb.clone())
                .await
                .map_err(|e| BiliApiError::Other(format!("音频下载失败: {}", e)))?;
        }
        DownloadMode::VideoOnly => {
            let video = match &task.video_url {
                Some(u) => u,
                None => {
                    task.status = DownloadStatus::Failed;
                    task.error = Some("该分 P 无视频直链（清晰度不可用）".into());
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            };
            let out = dir.join(format!("{}.mp4", base));
            client
                .download_to_file(video, out.to_str().unwrap(), prog_cb.clone())
                .await
                .map_err(|e| BiliApiError::Other(format!("视频下载失败: {}", e)))?;
        }
        DownloadMode::Merge => {
            let video = match &task.video_url {
                Some(u) => u,
                None => {
                    task.status = DownloadStatus::Failed;
                    task.error = Some("Merge 模式需视频直链，但目标清晰度不可用".into());
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            };
            let audio = task
                .audio_url
                .clone()
                .ok_or_else(|| {
                    task.status = DownloadStatus::Failed;
                    task.error = Some("Merge 模式需音频直链".into());
                    BiliApiError::Other(task.error.clone().unwrap())
                })?;
            if !ffmpeg_available() {
                // 回退：仅下载音 + 视频，提示用户手动合并
                let vout = dir.join(format!("{}.video.mp4", base));
                let aout = dir.join(format!("{}.audio.m4a", base));
                client
                    .download_to_file(video, vout.to_str().unwrap(), prog_cb.clone())
                    .await
                    .map_err(|e| BiliApiError::Other(format!("视频下载失败: {}", e)))?;
                client
                    .download_to_file(&audio, aout.to_str().unwrap(), None)
                    .await
                    .map_err(|e| BiliApiError::Other(format!("音频下载失败: {}", e)))?;
                task.error = Some("ffmpeg 不可用，已分别下载音/视频，请手动合并".into());
                task.status = DownloadStatus::Completed;
                return Ok(());
            }
            let vtmp = dir.join(format!("{}.video.mp4", base));
            let atmp = dir.join(format!("{}.audio.m4a", base));
            let out = dir.join(format!("{}.mp4", base));
            client
                .download_to_file(video, vtmp.to_str().unwrap(), prog_cb.clone())
                .await
                .map_err(|e| BiliApiError::Other(format!("视频下载失败: {}", e)))?;
            client
                .download_to_file(&audio, atmp.to_str().unwrap(), None)
                .await
                .map_err(|e| BiliApiError::Other(format!("音频下载失败: {}", e)))?;
            merge_with_ffmpeg(
                vtmp.to_str().unwrap(),
                atmp.to_str().unwrap(),
                out.to_str().unwrap(),
            )?;
            // 清理临时文件
            let _ = std::fs::remove_file(&vtmp);
            let _ = std::fs::remove_file(&atmp);
        }
    }

    task.status = DownloadStatus::Completed;
    task.error = None;
    Ok(())
}

/// 并发执行一批任务（Semaphore 限制并发数）
///
/// 返回每个任务的执行结果（`Err` 表示失败），单个失败不影响其他任务。
/// `on_progress` 需为 `Arc` 包裹的闭包以满足 `'static`（跨任务线程共享）。
pub async fn run_batch(
    client: Arc<HttpClient>,
    tasks: &mut [DownloadTask],
    concurrency: usize,
    on_progress: Option<Arc<dyn Fn(&DownloadTask, DlProgress) + Send + Sync>>,
) -> Vec<Result<(), BiliApiError>> {
    let sem = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut handles = Vec::new();

    for idx in 0..tasks.len() {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let mut task = tasks[idx].clone();
        let cb = on_progress.clone();
        let handle = tokio::spawn(async move {
            let res = match &cb {
                Some(arc_cb) => {
                    run_task(&client, &mut task, Some(arc_cb.clone())).await
                }
                None => run_task(&client, &mut task, None).await,
            };
            drop(permit);
            (idx, task, res)
        });
        handles.push(handle);
    }

    let mut results: Vec<Result<(), BiliApiError>> = (0..tasks.len()).map(|_| Ok(())).collect();
    for h in handles {
        let (idx, task, res) = h.await.unwrap();
        tasks[idx] = task;
        results[idx] = res;
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page_stream(video: Option<&str>, audio: Option<&str>) -> PageStream {
        PageStream {
            page: 1,
            part: "P1".into(),
            video_url: video.map(|s| s.to_string()),
            audio_url: audio.map(|s| s.to_string()),
            actual_format: 80,
        }
    }

    fn resolve_result(pages: Vec<PageStream>) -> ResolveResult {
        ResolveResult {
            bvid: "BV1xx".into(),
            title: "测试视频".into(),
            pages,
        }
    }

    #[test]
    fn test_from_page_mode_and_title() {
        let res = resolve_result(vec![page_stream(Some("v"), Some("a"))]);
        let task = DownloadTask::from_page(&res, &res.pages[0], DownloadMode::Merge, "/tmp");
        assert_eq!(task.id, "BV1xx#1");
        assert_eq!(task.mode, DownloadMode::Merge);
        assert_eq!(task.video_url.as_deref(), Some("v"));
        assert_eq!(task.audio_url.as_deref(), Some("a"));
        assert_eq!(task.status, DownloadStatus::Pending);
    }

    #[test]
    fn test_from_resolves_expands_per_page() {
        let res = resolve_result(vec![
            page_stream(Some("v1"), Some("a1")),
            page_stream(Some("v2"), Some("a2")),
        ]);
        let tasks = DownloadTask::from_resolves(&[res], DownloadMode::AudioOnly, "/tmp");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].audio_url.as_deref(), Some("a1"));
        assert_eq!(tasks[1].video_url.as_deref(), Some("v2"));
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize("a/b:c*?"), "a_b_c__");
        assert_eq!(sanitize(" normal "), "normal");
    }
}
