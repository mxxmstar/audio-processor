//! 音视频下载编排（阶段 3）
//!
//! 将阶段 1 的解析结果（`resolve_video` / `resolve_collection` / `resolve_season`）
//! 与阶段 2 的流式下载（`HttpClient::download_to_file`）组合为可执行的下载任务。
//!
//! 支持三种模式（对齐 Go 版 download）：
//! - `AudioOnly`：仅下载音频
//! - `VideoOnly`：仅下载视频
//! - `Merge`：下载音 + 视频后用 `media` 模块调用 ffmpeg 合并为 mp4

use crate::audio_rename;
use crate::biliapi::error::BiliApiError;
use crate::biliapi::media;
use crate::biliapi::video::{PageStream, ResolveResult};
use crate::http_client::client::HttpClient;
use crate::http_client::error::HttpClientError;
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
    /// 已暂停（保留已下载部分，可断点续传）
    Paused,
    /// 已停止（已删除已下载部分）
    Cancelled,
}

/// 音频识别和自动重命名状态，与下载传输状态独立维护。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RecognitionStatus {
    Disabled,
    Pending,
    Recognizing,
    Renamed,
    NoMatch,
    BelowThreshold,
    Failed,
    RenameFailed,
}

impl Default for RecognitionStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// 下载控制句柄：通过 `Arc<AtomicBool>` 信号实现暂停 / 停止。
///
/// - `pause` 置位：当前任务下载完成后进入 `Paused`，保留已下载文件，可续传。
/// - `stop` 置位：立即中断下载并删除已下载部分，状态置为 `Cancelled`。
#[derive(Clone, Default)]
pub struct DownloadControl {
    pub pause: Arc<std::sync::atomic::AtomicBool>,
    pub stop: Arc<std::sync::atomic::AtomicBool>,
}

/// 合集/系列分组信息（用于前端折叠展示）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct TaskGroup {
    /// 分组 ID（合集 season_id / 系列 id，稳定唯一）
    pub id: String,
    /// 分组展示标题（合集/系列名）
    pub title: String,
}

/// 单个下载任务
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DownloadTask {
    /// 任务 ID
    pub id: String,
    /// 展示标题（文件命名用）
    pub title: String,
    /// B 站原始标题，自动重命名后仍保留，用于兜底和问题排查。
    #[serde(default)]
    pub source_title: String,
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
    /// 音频识别/重命名状态。
    #[serde(default)]
    pub recognition_status: RecognitionStatus,
    /// 识别结果（仅自动识别成功或得到匹配时存在）。
    #[serde(default)]
    pub recognition_result: Option<crate::recognizer::SongInfo>,
    /// 识别或重命名错误，不覆盖下载错误。
    #[serde(default)]
    pub recognition_error: Option<String>,
    /// 下载过程中的稳定临时文件路径。
    #[serde(default)]
    pub staged_path: Option<String>,
    /// 实际最终落盘路径。
    #[serde(default)]
    pub output_path: Option<String>,
    /// 所属分组（合集/系列）；非合集任务为 None
    #[serde(default)]
    pub group: Option<TaskGroup>,
    /// 视频封面 URL（用于前端展示），无则 None
    #[serde(default)]
    pub cover: Option<String>,
}

impl DownloadTask {
    /// 从解析结果的一个分 P 构造任务
    /// `multi_page` 为 true 时该视频含多个分 P，标题需用 `part` 区分；
    /// 单 P 视频优先用视频真实标题 `res.title`（`part` 常为占位名如 "v3"）。
    pub fn from_page(
        res: &ResolveResult,
        page: &PageStream,
        mode: DownloadMode,
        output_dir: &str,
        group: Option<TaskGroup>,
        multi_page: bool,
    ) -> Self {
        let id = format!("{}#{}", res.bvid, page.page);
        let title = if multi_page {
            // 多 P：用分 P 名区分，part 为空时回退「标题 Pn」
            if page.part.is_empty() {
                format!("{} P{}", res.title, page.page)
            } else {
                page.part.clone()
            }
        } else {
            // 单 P：直接用视频真实标题。分 P 名（part）常为占位名
            // （如 "v3"）或默认等于标题，不应覆盖真实标题。
            res.title.clone()
        };
        let cover = if res.cover.is_empty() {
            None
        } else {
            Some(res.cover.clone())
        };
        DownloadTask {
            id,
            title: title.clone(),
            source_title: title,
            video_url: page.video_url.clone(),
            audio_url: page.audio_url.clone(),
            mode,
            output_dir: output_dir.to_string(),
            status: DownloadStatus::Pending,
            error: None,
            recognition_status: if mode == DownloadMode::AudioOnly {
                RecognitionStatus::Pending
            } else {
                RecognitionStatus::Disabled
            },
            recognition_result: None,
            recognition_error: None,
            staged_path: None,
            output_path: None,
            group,
            cover,
        }
    }

    /// 将一批解析结果展开为任务列表（每分 P 一个任务）
    /// `group` 为该批所属的分组（合集/系列），用于前端折叠展示。
    pub fn from_resolves(
        results: &[ResolveResult],
        mode: DownloadMode,
        output_dir: &str,
        group: Option<TaskGroup>,
    ) -> Vec<Self> {
        let mut tasks = Vec::new();
        for res in results {
            let multi = res.pages.len() > 1;
            for page in &res.pages {
                tasks.push(Self::from_page(
                    res,
                    page,
                    mode,
                    output_dir,
                    group.clone(),
                    multi,
                ));
            }
        }
        tasks
    }
}

/// 安全化文件名（去除文件系统非法字符）
fn sanitize(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars() {
        if c == '\\'
            || c == '/'
            || c == ':'
            || c == '*'
            || c == '?'
            || c == '"'
            || c == '<'
            || c == '>'
            || c == '|'
        {
            s.push('_');
        } else {
            s.push(c);
        }
    }
    s.trim().to_string()
}

impl DownloadTask {
    /// 返回本任务的稳定临时文件路径，用于断点续传和下载后识别。
    pub fn staged_file(&self) -> std::path::PathBuf {
        self.staged_path
            .as_ref()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                audio_rename::staging_path(Path::new(&self.output_dir), &self.id, "m4a")
            })
    }

    /// 推算本任务最终落盘的绝对路径（与 `run_task` 内命名规则保持一致）。
    /// - AudioOnly → `<dir>/<title>.mp3`
    /// - VideoOnly / Merge → `<dir>/<title>.mp4`
    pub fn output_file(&self) -> std::path::PathBuf {
        if let Some(path) = &self.output_path {
            return std::path::PathBuf::from(path);
        }
        let dir = Path::new(&self.output_dir);
        let base = sanitize(&self.title);
        let ext = match self.mode {
            DownloadMode::AudioOnly => "mp3",
            DownloadMode::VideoOnly | DownloadMode::Merge => "mp4",
        };
        dir.join(format!("{}.{}", base, ext))
    }
}

/// ffmpeg 的探测与合并逻辑已迁移至 [`crate::biliapi::media`] 模块。
/// 本模块仅保留下载编排，通过 `media::ffmpeg_available` / `media::merge` 调用。

/// 执行单个下载任务
///
/// - `AudioOnly`：下音频到稳定 `.m4a.part`，完成后的 MP3 由命令层后处理
/// - `VideoOnly`：下视频 → `<title>.mp4`
/// - `Merge`：下音 + 视频 → ffmpeg 合并 `<title>.mp4`；若 ffmpeg 不可用则回退为仅下载音/视频并标记提示
///
/// `control` 为可选取消控制：
/// - `control.stop` 置位 → 立即中断，删除已下载部分，返回 `Err`，由调用方置 `Cancelled`。
/// - `control.pause` 置位 → 当前文件下载完成后停止，保留已下载部分，置 `Paused`。
pub async fn run_task(
    client: &HttpClient,
    task: &mut DownloadTask,
    on_progress: Option<Arc<dyn Fn(&DownloadTask, DlProgress) + Send + Sync>>,
    control: Option<&DownloadControl>,
) -> Result<(), BiliApiError> {
    task.status = DownloadStatus::Downloading;
    let dir = Path::new(&task.output_dir);
    std::fs::create_dir_all(dir)
        .map_err(|e| BiliApiError::Other(format!("创建输出目录失败: {}", e)))?;
    let base = sanitize(&task.title);

    // 进度回调闭包：用 task 快照（Send）包裹，满足 download_to_file 的 Arc<dyn Fn+Send+Sync> 要求
    let snapshot = task.clone();
    let prog_cb: Option<Arc<dyn Fn(DlProgress) + Send + Sync>> = on_progress.map(|f| {
        Arc::new(move |p: DlProgress| f(&snapshot, p)) as Arc<dyn Fn(DlProgress) + Send + Sync>
    });

    // 停止信号：优先于暂停，立即中断并删除已下载部分
    let stop = control.map(|c| c.stop.clone());
    // 暂停信号：当前文件下完即停，保留部分
    let pause = control.map(|c| c.pause.clone());

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
            let out = task.staged_file();
            task.staged_path = Some(out.to_string_lossy().into_owned());
            match client
                .download_to_file(
                    audio,
                    out.to_str().unwrap(),
                    prog_cb.clone(),
                    stop.clone(),
                    pause.clone(),
                )
                .await
            {
                Ok(()) => {}
                Err(e) => {
                    // 取消信号（stop/pause）触发：按信号类型分别处理
                    if let HttpClientError::Cancelled(ref msg) = e {
                        if msg.starts_with("pause:") {
                            // 暂停：保留已下载部分，标记可续传
                            task.status = DownloadStatus::Paused;
                            task.error = Some("已暂停（可续传）".into());
                            return Ok(());
                        } else {
                            // 停止：删除已下载部分
                            let _ = std::fs::remove_file(&out);
                            task.status = DownloadStatus::Cancelled;
                            task.error = Some("已停止（已删除已下载部分）".into());
                            return Err(BiliApiError::Other(task.error.clone().unwrap()));
                        }
                    }
                    task.status = DownloadStatus::Failed;
                    task.error = Some(format!("音频下载失败: {}", e));
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            }
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
            match client
                .download_to_file(
                    video,
                    out.to_str().unwrap(),
                    prog_cb.clone(),
                    stop.clone(),
                    pause.clone(),
                )
                .await
            {
                Ok(()) => {}
                Err(e) => {
                    if let HttpClientError::Cancelled(ref msg) = e {
                        if msg.starts_with("pause:") {
                            task.status = DownloadStatus::Paused;
                            task.error = Some("已暂停（可续传）".into());
                            return Ok(());
                        } else {
                            let _ = std::fs::remove_file(&out);
                            task.status = DownloadStatus::Cancelled;
                            task.error = Some("已停止（已删除已下载部分）".into());
                            return Err(BiliApiError::Other(task.error.clone().unwrap()));
                        }
                    }
                    task.status = DownloadStatus::Failed;
                    task.error = Some(format!("视频下载失败: {}", e));
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            }
        }
        DownloadMode::Merge => {
            println!(
                "[task] 进入 Merge 分支，task.video_url={:?}, task.audio_url={:?}",
                task.video_url, task.audio_url
            );
            let video = match &task.video_url {
                Some(u) => u,
                None => {
                    task.status = DownloadStatus::Failed;
                    task.error = Some("Merge 模式需视频直链，但目标清晰度不可用".into());
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            };
            let audio = task.audio_url.clone().ok_or_else(|| {
                task.status = DownloadStatus::Failed;
                task.error = Some("Merge 模式需音频直链".into());
                BiliApiError::Other(task.error.clone().unwrap())
            })?;
            if !media::ffmpeg_available() {
                // 回退：仅下载音 + 视频，提示用户手动合并
                println!("[task] ffmpeg 不可用 -> 走回退分支（分别下载音/视频）");
                let vout = dir.join(format!("{}.video.mp4", base));
                let aout = dir.join(format!("{}.audio.m4a", base));
                match client
                    .download_to_file(
                        video,
                        vout.to_str().unwrap(),
                        prog_cb.clone(),
                        stop.clone(),
                        pause.clone(),
                    )
                    .await
                {
                    Ok(()) => {}
                    Err(e) => {
                        if let HttpClientError::Cancelled(ref msg) = e {
                            if msg.starts_with("pause:") {
                                task.status = DownloadStatus::Paused;
                                task.error = Some("已暂停（可续传）".into());
                                return Ok(());
                            } else {
                                let _ = std::fs::remove_file(&vout);
                                task.status = DownloadStatus::Cancelled;
                                task.error = Some("已停止（已删除已下载部分）".into());
                                return Err(BiliApiError::Other(task.error.clone().unwrap()));
                            }
                        }
                        task.status = DownloadStatus::Failed;
                        task.error = Some(format!("视频下载失败: {}", e));
                        return Err(BiliApiError::Other(task.error.clone().unwrap()));
                    }
                }
                match client
                    .download_to_file(
                        &audio,
                        aout.to_str().unwrap(),
                        None,
                        stop.clone(),
                        pause.clone(),
                    )
                    .await
                {
                    Ok(()) => {}
                    Err(e) => {
                        if let HttpClientError::Cancelled(ref msg) = e {
                            if msg.starts_with("pause:") {
                                task.status = DownloadStatus::Paused;
                                task.error = Some("已暂停（可续传）".into());
                                return Ok(());
                            } else {
                                let _ = std::fs::remove_file(&aout);
                                task.status = DownloadStatus::Cancelled;
                                task.error = Some("已停止（已删除已下载部分）".into());
                                return Err(BiliApiError::Other(task.error.clone().unwrap()));
                            }
                        }
                        task.status = DownloadStatus::Failed;
                        task.error = Some(format!("音频下载失败: {}", e));
                        return Err(BiliApiError::Other(task.error.clone().unwrap()));
                    }
                }
                task.error = Some("ffmpeg 不可用，已分别下载音/视频，请手动合并".into());
                task.status = DownloadStatus::Completed;
                return Ok(());
            }
            let vtmp = dir.join(format!("{}.video.mp4", base));
            let atmp = dir.join(format!("{}.audio.m4a", base));
            let out = dir.join(format!("{}.mp4", base));
            println!("[task] ffmpeg 可用 -> 准备合并，输出: {}", out.display());
            match client
                .download_to_file(
                    video,
                    vtmp.to_str().unwrap(),
                    prog_cb.clone(),
                    stop.clone(),
                    pause.clone(),
                )
                .await
            {
                Ok(()) => {}
                Err(e) => {
                    if let HttpClientError::Cancelled(ref msg) = e {
                        if msg.starts_with("pause:") {
                            task.status = DownloadStatus::Paused;
                            task.error = Some("已暂停（可续传）".into());
                            return Ok(());
                        } else {
                            let _ = std::fs::remove_file(&vtmp);
                            task.status = DownloadStatus::Cancelled;
                            task.error = Some("已停止（已删除已下载部分）".into());
                            return Err(BiliApiError::Other(task.error.clone().unwrap()));
                        }
                    }
                    task.status = DownloadStatus::Failed;
                    task.error = Some(format!("视频下载失败: {}", e));
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            }
            match client
                .download_to_file(
                    &audio,
                    atmp.to_str().unwrap(),
                    None,
                    stop.clone(),
                    pause.clone(),
                )
                .await
            {
                Ok(()) => {}
                Err(e) => {
                    if let HttpClientError::Cancelled(ref msg) = e {
                        if msg.starts_with("pause:") {
                            task.status = DownloadStatus::Paused;
                            task.error = Some("已暂停（可续传）".into());
                            return Ok(());
                        } else {
                            let _ = std::fs::remove_file(&atmp);
                            task.status = DownloadStatus::Cancelled;
                            task.error = Some("已停止（已删除已下载部分）".into());
                            return Err(BiliApiError::Other(task.error.clone().unwrap()));
                        }
                    }
                    task.status = DownloadStatus::Failed;
                    task.error = Some(format!("音频下载失败: {}", e));
                    return Err(BiliApiError::Other(task.error.clone().unwrap()));
                }
            }
            media::merge(
                vtmp.to_str().unwrap(),
                atmp.to_str().unwrap(),
                out.to_str().unwrap(),
            )?;
            // 清理临时文件（合并成功则 media::merge 不会留半成品；此处再兜底清理）
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
/// `control` 为可选取消控制（暂停 / 停止），传给每个 `run_task`。
pub async fn run_batch(
    client: Arc<HttpClient>,
    tasks: &mut [DownloadTask],
    concurrency: usize,
    on_progress: Option<Arc<dyn Fn(&DownloadTask, DlProgress) + Send + Sync>>,
    control: Option<&DownloadControl>,
) -> Vec<Result<(), BiliApiError>> {
    let sem = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut handles = Vec::new();

    for idx in 0..tasks.len() {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let mut task = tasks[idx].clone();
        let cb = on_progress.clone();
        let ctrl = control.cloned();
        let handle = tokio::spawn(async move {
            let res = match &cb {
                Some(arc_cb) => {
                    run_task(&client, &mut task, Some(arc_cb.clone()), ctrl.as_ref()).await
                }
                None => run_task(&client, &mut task, None, ctrl.as_ref()).await,
            };
            // 任务收尾后再发一次进度事件，携带最终真实状态
            // （Cancelled / Paused / Completed / Failed），供前端实时展示。
            if let Some(ref arc_cb) = cb {
                let final_progress = DlProgress {
                    downloaded: 0,
                    total: None,
                    speed: 0,
                    percent: if matches!(task.status, DownloadStatus::Completed) {
                        100.0
                    } else {
                        0.0
                    },
                };
                arc_cb(&task, final_progress);
            }
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
        page_stream_with_part(video, audio, "P1")
    }

    fn page_stream_with_part(video: Option<&str>, audio: Option<&str>, part: &str) -> PageStream {
        PageStream {
            page: 1,
            part: part.into(),
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
            cover: String::new(),
        }
    }

    #[test]
    fn test_from_page_mode_and_title() {
        let res = resolve_result(vec![page_stream(Some("v"), Some("a"))]);
        let task = DownloadTask::from_page(
            &res,
            &res.pages[0],
            DownloadMode::Merge,
            "/tmp",
            None,
            false,
        );
        assert_eq!(task.id, "BV1xx#1");
        assert_eq!(task.mode, DownloadMode::Merge);
        assert_eq!(task.video_url.as_deref(), Some("v"));
        assert_eq!(task.audio_url.as_deref(), Some("a"));
        assert_eq!(task.status, DownloadStatus::Pending);
    }

    #[test]
    fn test_from_resolves_multi_page_uses_part() {
        // 多 P：标题用分 P 名（part）区分
        let res = resolve_result(vec![
            page_stream_with_part(Some("v1"), Some("a1"), "P1"),
            page_stream_with_part(Some("v2"), Some("a2"), "P2"),
        ]);
        let tasks = DownloadTask::from_resolves(&[res], DownloadMode::AudioOnly, "/tmp", None);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].title, "P1");
        assert_eq!(tasks[1].title, "P2");
        assert_eq!(tasks[0].audio_url.as_deref(), Some("a1"));
        assert_eq!(tasks[1].video_url.as_deref(), Some("v2"));
    }

    #[test]
    fn test_from_resolves_single_page_uses_real_title() {
        // 单 P：标题应取视频真实标题，而非 part 占位名（如 "v3"）
        let res = resolve_result(vec![page_stream_with_part(Some("v"), Some("a"), "")]);
        let tasks = DownloadTask::from_resolves(&[res], DownloadMode::AudioOnly, "/tmp", None);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "测试视频");
    }

    #[test]
    fn test_single_page_placeholder_part_not_used() {
        // 复现真实 bug：单 P 视频 part="v3" 这种占位名不应成为标题
        let res = ResolveResult {
            bvid: "BV1xx".into(),
            title: "伊利亚的赌注：马斯克曾嘲讽的GPT路线【硅基诗篇5】".into(),
            pages: vec![PageStream {
                page: 1,
                part: "v3".into(),
                video_url: Some("v".into()),
                audio_url: Some("a".into()),
                actual_format: 80,
            }],
            cover: String::new(),
        };
        let tasks = DownloadTask::from_resolves(&[res], DownloadMode::AudioOnly, "/tmp", None);
        assert_eq!(
            tasks[0].title,
            "伊利亚的赌注：马斯克曾嘲讽的GPT路线【硅基诗篇5】"
        );
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize("a/b:c*?"), "a_b_c__");
        assert_eq!(sanitize(" normal "), "normal");
    }
}
