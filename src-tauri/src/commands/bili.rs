//! 阶段 5：B 站下载相关 Tauri 命令
//!
//! 命令层把 `biliapi` 的能力暴露给前端：
//! - 解析 URL（视频 / 合集 / 番剧）→ 展开任务
//! - 登录态（生成二维码 / 轮询 / 校验 / 登出）
//! - 启动并发下载，进度经 Tauri 事件 `download-progress` 推送到前端

use crate::audio_rename::{self, AutoRenameConfig};
use crate::audio_quality::ai_worker::{
    self, AiProcessRequest, WorkerError, WorkerEvent, WorkerEventCallback, WorkerSpec,
};
use crate::bili_state::BiliState;
use crate::biliapi::client::BiliClient;
use crate::biliapi::login;
use crate::biliapi::media;
use crate::biliapi::task::{
    DownloadMode, DownloadTask, QualityStatus, RecognitionStatus, TaskGroup,
};
use crate::biliapi::types::{MediaFormat, QrInfo};
use crate::biliapi::video;
use crate::recognizer::run_identify;
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};

/// 解析请求参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolveInput {
    /// 原始输入：BV 号 / AV 号 / 链接 / 合集(ss) / 番剧(ep)
    pub input: String,
    /// 下载模式：audio / video / merge（缺省 audio）
    #[serde(default)]
    pub mode: Option<String>,
    /// 优先清晰度（如 "1080P"、"720P"），缺省 "1080P"
    #[serde(default)]
    pub prefer_format: Option<String>,
    /// 输出目录（缺省用配置目录）
    #[serde(default)]
    pub output_dir: Option<String>,
    /// 仅解析指定的分集 bvid 列表（合集场景下，前端先预览分集列表、
    /// 用户勾选后再传入，避免一次性解析全部分集）。为 None 时按 input 自动识别。
    #[serde(default)]
    pub bvids: Option<Vec<String>>,
}

/// 合集预览中的单个分集
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionEpisode {
    /// 分集序号（从 1 开始）
    pub index: usize,
    /// 分集 BV 号
    pub bvid: String,
    /// 分集标题
    pub title: String,
    /// 分集封面图 URL
    pub cover: String,
}

/// 合集预览结果：视频属于某个合集时返回分集列表，供前端勾选
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionPreview {
    /// 合集 id
    pub id: String,
    /// 合集标题
    pub title: String,
    /// 全部分集
    pub episodes: Vec<CollectionEpisode>,
}

/// 从 `UgcSeason` 提取分集 bvid 列表（兼容 `episodes` 与 `sections[].episodes`）
fn ugc_season_bvids(season: &crate::biliapi::types::UgcSeason) -> Vec<String> {
    season
        .flatten_episodes()
        .into_iter()
        .map(|(_, bvid, _, _)| bvid)
        .collect()
}

/// 启动下载请求
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartDownloadInput {
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default)]
    pub concurrency: Option<usize>,
    /// 仅下载指定 ID 的任务（用于前端手动勾选）。
    /// 为 None 时下载全部已解析任务。
    #[serde(default)]
    pub task_ids: Option<Vec<String>>,
    /// 是否在仅音频下载完成后自动识别并重命名；缺省开启。
    #[serde(default)]
    pub auto_rename: Option<bool>,
    /// 自动识别的最低置信度，缺省 70%。
    #[serde(default)]
    pub confidence_threshold: Option<f64>,
    /// 是否在仅音频下载完成后启动 Python AI 增强；默认关闭。
    #[serde(default)]
    pub python_ai_enhancement_enabled: Option<bool>,
    /// AI 模型 ID；默认 audiosr-basic。
    #[serde(default)]
    pub python_ai_model_id: Option<String>,
    /// AI 推理设备，例如 auto / cpu / cuda。
    #[serde(default)]
    pub python_ai_device: Option<String>,
    /// AI 单次处理的音频秒数。
    #[serde(default)]
    pub python_ai_chunk_seconds: Option<f64>,
    /// AI 分块重叠秒数。
    #[serde(default)]
    pub python_ai_overlap_seconds: Option<f64>,
}

#[derive(Debug, Clone)]
struct PythonAiEnhancementConfig {
    enabled: bool,
    model_id: String,
    device: String,
    chunk_seconds: f64,
    overlap_seconds: f64,
}

fn validate_python_ai_config(config: &PythonAiEnhancementConfig) -> Result<(), String> {
    if config.model_id.trim().is_empty() {
        return Err("pythonAiModelId 不能为空".into());
    }
    if !matches!(config.device.as_str(), "auto" | "cpu" | "cuda") {
        return Err("pythonAiDevice 必须是 auto、cpu 或 cuda".into());
    }
    if !(config.chunk_seconds.is_finite() && config.chunk_seconds > 0.0) {
        return Err("pythonAiChunkSeconds 必须是正数".into());
    }
    if !(config.overlap_seconds.is_finite()
        && config.overlap_seconds >= 0.0
        && config.overlap_seconds < config.chunk_seconds)
    {
        return Err("pythonAiOverlapSeconds 必须大于等于 0 且小于分块长度".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioPostprocessStage {
    Recognize,
    Enhance,
    Encode,
}

fn audio_postprocess_plan(
    recognition_enabled: bool,
    enhancement_enabled: bool,
) -> Vec<AudioPostprocessStage> {
    let mut stages = Vec::with_capacity(3);
    if recognition_enabled {
        stages.push(AudioPostprocessStage::Recognize);
    }
    if enhancement_enabled {
        stages.push(AudioPostprocessStage::Enhance);
    }
    stages.push(AudioPostprocessStage::Encode);
    stages
}

/// 生成的登录二维码
#[derive(Debug, Clone, Serialize)]
pub struct LoginQr {
    /// 二维码 SVG（data URL 可直接 `<img src>` 渲染）
    pub qr_svg: String,
    pub qr_key: String,
}

/// 轮询二维码登录结果
#[derive(Debug, Clone, Serialize)]
pub struct LoginState {
    pub authed: bool,
    pub message: String,
}

/// 已登录用户的 B 站资料（头像 / 昵称）
#[derive(Debug, Clone, Serialize)]
pub struct BiliUserInfo {
    pub name: String,
    pub face: String,
}

/// 进度事件（推送到前端）
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    /// 阶段：`resolve`（解析中）/ `download`（下载中）
    pub phase: String,
    pub task_id: String,
    pub title: String,
    pub status: String,
    pub percent: f64,
    pub downloaded: u64,
    pub total: u64,
    pub speed: u64,
    pub error: Option<String>,
    #[serde(default)]
    pub source_title: String,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub quality_status: Option<String>,
    #[serde(default)]
    pub quality_model_id: Option<String>,
    #[serde(default)]
    pub quality_output_path: Option<String>,
    #[serde(default)]
    pub quality_error: Option<String>,
}

/// 解析完成事件（带最终任务列表）
#[derive(Debug, Clone, Serialize)]
pub struct ResolveFinished {
    pub ok: bool,
    /// 解析出的任务列表（失败时为空）
    pub tasks: Vec<DownloadTask>,
    /// 失败原因（ok == false 时）
    pub error: Option<String>,
    /// 总条目数 / 成功解析数（用于展示「解析 X/Y」）
    pub total: usize,
    pub resolved: usize,
}

fn parse_mode(s: &str) -> DownloadMode {
    match s.to_lowercase().as_str() {
        "video" => DownloadMode::VideoOnly,
        "merge" => DownloadMode::Merge,
        _ => DownloadMode::AudioOnly,
    }
}

/// 清晰度 label → code（用于构建 prefer_format）
fn parse_format(label: &str) -> i64 {
    let l = label.trim().to_uppercase();
    let m = match l.as_str() {
        "360P" => MediaFormat::Q_360P,
        "720P" => MediaFormat::Q_720P,
        "1080P" | "1080" => MediaFormat::Q_1080P,
        "1080P+" | "1080P_PLUS" => MediaFormat::Q_1080P_PLUS,
        "4K" => MediaFormat::Q_4K,
        "DOLBY" => MediaFormat::Q_DOLBY,
        "HDR" => MediaFormat::Q_HDR,
        "8K" => MediaFormat::Q_8K,
        _ => MediaFormat::Q_1080P,
    };
    m.0
}

/// 从用户输入识别目标并解析为下载任务（不立即下载）
#[tauri::command]
pub async fn bili_resolve(
    input: ResolveInput,
    state: State<'_, BiliState>,
) -> Result<Vec<DownloadTask>, String> {
    let sessdata = login::load_and_check(state.config_dir_opt().as_deref())
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "未登录或登录态已失效，请先扫码登录".to_string())?;

    let mode = parse_mode(input.mode.as_deref().unwrap_or("audio"));
    let prefer = parse_format(input.prefer_format.as_deref().unwrap_or("1080P"));
    // 预取 buvid3/buvid4 指纹，规避 web-space 接口 -352 风控（失败不致命）
    let _ = crate::biliapi::buvid_cache::ensure_buvid().await;
    let client = BiliClient::new(&sessdata);

    // 识别目标类型并分发解析
    let target = identify(&input.input);
    let mut group_opt: Option<TaskGroup> = None;
    // 若前端已传入勾选的分集 bvid 列表，则只解析这些分集（跳过合集自动展开）
    let results = if let Some(sel) = input.bvids.clone().filter(|v| !v.is_empty()) {
        group_opt = None;
        video::resolve_bvids(&client, sel, prefer, None)
            .await
            .map_err(|e| e.to_string())?
    } else {
        match target {
        Target::Bv(bvid) => {
            // 先取视频详情，判断是否属于合集（ugc_season）
            let info = video::get_video_info(&client, &bvid)
                .await
                .map_err(|e| e.to_string())?;
            if info.ugc_season.id > 0 {
                // 直接从 view 返回的 ugc_season 取合集分集（兼容 episodes / sections[].episodes），
                // 避免再调 seasons_archives_list（该接口被风控拦截，稳定返回 -352）。
                let mut bvids = ugc_season_bvids(&info.ugc_season);
                if bvids.is_empty() {
                    bvids.push(bvid.clone());
                }
                let group = TaskGroup {
                    id: info.ugc_season.id.to_string(),
                    title: if info.ugc_season.title.is_empty() {
                        info.title.clone()
                    } else {
                        info.ugc_season.title.clone()
                    },
                };
                group_opt = Some(group);
                video::resolve_bvids(&client, bvids, prefer, None)
                    .await
                    .map_err(|e| e.to_string())?
            } else {
                vec![video::resolve_video(&client, &bvid, prefer)
                    .await
                    .map_err(|e| e.to_string())?]
            }
        }
        Target::Av(aid) => {
            // AV 号需先转 BV 号；复用 view 接口拿 bvid 与合集信息
            let info = video::get_video_info(&client, &bv_from_aid(aid))
                .await
                .map_err(|e| e.to_string())?;
            if info.ugc_season.id > 0 {
                let mut bvids: Vec<String> = info
                    .ugc_season
                    .episodes
                    .iter()
                    .map(|e| e.bvid.clone())
                    .filter(|b| !b.is_empty())
                    .collect();
                if bvids.is_empty() {
                    bvids.push(info.bvid.clone());
                }
                let group = TaskGroup {
                    id: info.ugc_season.id.to_string(),
                    title: if info.ugc_season.title.is_empty() {
                        info.title.clone()
                    } else {
                        info.ugc_season.title.clone()
                    },
                };
                group_opt = Some(group);
                video::resolve_bvids(&client, bvids, prefer, None)
                    .await
                    .map_err(|e| e.to_string())?
            } else {
                vec![video::resolve_video(&client, &info.bvid, prefer)
                    .await
                    .map_err(|e| e.to_string())?]
            }
        }
        Target::Collection(mid, sid) => {
            let (r, g) = video::resolve_collection(&client, &mid, &sid, prefer, None)
                .await
                .map_err(|e| e.to_string())?;
            group_opt = Some(g);
            r
        }
        Target::Season(ssid) => video::resolve_season(&client, &ssid, prefer, None)
            .await
            .map_err(|e| e.to_string())?,
        }
    };

    let root = input.output_dir.clone().unwrap_or_else(|| {
        state
            .config_dir_opt()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    });

    let tasks: Vec<DownloadTask> = DownloadTask::from_resolves(&results, mode, &root, group_opt);
    if tasks.is_empty() {
        return Err("解析成功，但未找到可下载的音视频流".to_string());
    }

    state.set_tasks(tasks.clone());
    Ok(tasks)
}

/// 合集预览：识别输入是否为「属于某个合集的视频」，若是则返回合集的分集列表，
/// 供前端先展示勾选界面、用户选择后再调用 `bili_resolve`（传 `bvids`）只解析选中项。
///
/// 非合集视频 / 合集页 / 番剧入口返回 `None`（这些场景无需预览，走原解析流程）。
#[tauri::command]
pub async fn bili_preview(
    input: ResolveInput,
    state: State<'_, BiliState>,
) -> Result<Option<CollectionPreview>, String> {
    let sessdata = login::load_and_check(state.config_dir_opt().as_deref())
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "未登录或登录态已失效，请先扫码登录".to_string())?;
    let _ = crate::biliapi::buvid_cache::ensure_buvid().await;
    let client = BiliClient::new(&sessdata);

    // 仅视频页（BV/AV）需要预览；合集页 / 番剧走原流程
    let target = identify(&input.input);
    let bvid = match &target {
        Target::Bv(b) => b.clone(),
        Target::Av(a) => bv_from_aid(*a),
        _ => return Ok(None),
    };

    let info = video::get_video_info(&client, &bvid)
        .await
        .map_err(|e| e.to_string())?;
    if info.ugc_season.id <= 0 {
        return Ok(None);
    }

    let episodes: Vec<CollectionEpisode> = info
        .ugc_season
        .flatten_episodes()
        .into_iter()
        .map(|(i, bvid, title, cover)| CollectionEpisode {
            index: i,
            bvid,
            title,
            cover,
        })
        .collect();

    // 诊断：确认合集分集封面是否成功取到
    let empty_cover = episodes.iter().filter(|e| e.cover.is_empty()).count();
    println!(
        "[bili-debug] preview episodes={} empty_cover={} first_cover={:?}",
        episodes.len(),
        empty_cover,
        episodes.first().map(|e| e.cover.clone())
    );

    Ok(Some(CollectionPreview {
        id: info.ugc_season.id.to_string(),
        title: if info.ugc_season.title.is_empty() {
            info.title.clone()
        } else {
            info.ugc_season.title.clone()
        },
        episodes,
    }))
}

/// 异步解析（后台运行，区分「解析中 / 下载中」两阶段）
///
/// 立即返回，解析进度经 `download-progress` 事件推送（`phase = "resolve"`），
/// 完成后经 `resolve-finished` 事件推送最终任务列表。前端据此区分两阶段。
#[tauri::command]
pub async fn bili_resolve_async(
    input: ResolveInput,
    app: AppHandle,
    state: State<'_, BiliState>,
) -> Result<(), String> {
    let sessdata = login::load_and_check(state.config_dir_opt().as_deref())
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "未登录或登录态已失效，请先扫码登录".to_string())?;

    let mode = parse_mode(input.mode.as_deref().unwrap_or("audio"));
    let prefer = parse_format(input.prefer_format.as_deref().unwrap_or("1080P"));
    // 预取 buvid3/buvid4 指纹，规避 web-space 接口 -352 风控（失败不致命）
    let _ = crate::biliapi::buvid_cache::ensure_buvid().await;
    let client = BiliClient::new(&sessdata);
    let target = identify(&input.input);
    let root = input.output_dir.clone().unwrap_or_else(|| {
        state
            .config_dir_opt()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    });

    let app_for_cb = app.clone();
    tauri::async_runtime::spawn(async move {
        // 预取条目总数，便于展示「解析 X/Y」
        let mut total = match &target {
            Target::Bv(_) | Target::Av(_) => 1usize,
            Target::Collection(mid, sid) => {
                match video::get_collection_bvids(&client, mid, sid).await {
                    Ok((b, _)) => b.len(),
                    Err(_) => 0,
                }
            }
            Target::Season(ssid) => match video::get_season_info(&client, None, Some(ssid)).await {
                Ok(s) => s
                    .episodes
                    .iter()
                    .filter(|ep| !ep.bvid.is_empty() && ep.cid != 0)
                    .count(),
                Err(_) => 0,
            },
        };

        // 起始进度事件（phase = resolve）
        emit_resolve_progress(&app_for_cb, 0, total.max(1), "开始解析…");

        let app_for_resolve = app_for_cb.clone();
        let cb: Option<Arc<dyn Fn(usize, usize, &str) + Send + Sync>> =
            Some(Arc::new(move |done: usize, tot: usize, title: &str| {
                emit_resolve_progress(&app_for_resolve, done, tot, title);
            }));

        let mut group_opt: Option<TaskGroup> = None;
        // 若前端已传入勾选的分集 bvid 列表，则只解析这些分集（跳过合集自动展开）
        let result = if let Some(sel) = input.bvids.clone().filter(|v| !v.is_empty()) {
            video::resolve_bvids(&client, sel, prefer, cb.clone()).await
        } else {
            match &target {
            Target::Bv(bvid) => {
                // 先取视频详情，判断是否属于合集（ugc_season）
                match video::get_video_info(&client, bvid).await {
                    Ok(info) => {
                        if info.ugc_season.id > 0 {
                            // 直接从 view 返回的 ugc_season 取合集分集（兼容 episodes / sections[].episodes），
                            // 避免再调 seasons_archives_list（该接口被风控拦截，稳定返回 -352）。
                            let mut bvids = ugc_season_bvids(&info.ugc_season);
                            if bvids.is_empty() {
                                bvids.push(bvid.to_string());
                            }
                            total = bvids.len();
                            emit_resolve_progress(
                                &app_for_cb,
                                0,
                                total.max(1),
                                "开始解析合集…",
                            );
                            let group = TaskGroup {
                                id: info.ugc_season.id.to_string(),
                                title: if info.ugc_season.title.is_empty() {
                                    info.title.clone()
                                } else {
                                    info.ugc_season.title.clone()
                                },
                            };
                            group_opt = Some(group);
                            video::resolve_bvids(&client, bvids, prefer, cb).await
                        } else {
                            let r = video::resolve_video(&client, bvid, prefer).await;
                            cb.as_ref().map(|f| f(1, 1, bvid));
                            r.map(|r| vec![r])
                        }
                    }
                    Err(e) => Err(e),
                }
            }
            Target::Av(aid) => match video::get_video_info(&client, &bv_from_aid(*aid)).await {
                Ok(info) => {
                    if info.ugc_season.id > 0 {
                        let mut bvids = ugc_season_bvids(&info.ugc_season);
                        if bvids.is_empty() {
                            bvids.push(info.bvid.clone());
                        }
                        total = bvids.len();
                        emit_resolve_progress(&app_for_cb, 0, total.max(1), "开始解析合集…");
                        let group = TaskGroup {
                            id: info.ugc_season.id.to_string(),
                            title: if info.ugc_season.title.is_empty() {
                                info.title.clone()
                            } else {
                                info.ugc_season.title.clone()
                            },
                        };
                        group_opt = Some(group);
                        video::resolve_bvids(&client, bvids, prefer, cb).await
                    } else {
                        let r = video::resolve_video(&client, &info.bvid, prefer).await;
                        cb.as_ref().map(|f| f(1, 1, &info.bvid));
                        r.map(|r| vec![r])
                    }
                }
                Err(e) => Err(e),
            },
            Target::Collection(mid, sid) => {
                match video::resolve_collection(&client, mid, sid, prefer, cb).await {
                    Ok((r, g)) => {
                        group_opt = Some(g);
                        Ok(r)
                    }
                    Err(e) => Err(e),
                }
            }
            Target::Season(ssid) => video::resolve_season(&client, ssid, prefer, cb).await,
            }
        };

        match result {
            Ok(results) => {
                let tasks: Vec<DownloadTask> =
                    DownloadTask::from_resolves(&results, mode, &root, group_opt.clone());
                if tasks.is_empty() {
                    let _ = app_for_cb.emit(
                        "resolve-finished",
                        ResolveFinished {
                            ok: false,
                            tasks: vec![],
                            error: Some("解析成功，但未找到可下载的音视频流".into()),
                            total: total.max(1),
                            resolved: 0,
                        },
                    );
                    return;
                }
                app_for_cb.state::<BiliState>().set_tasks(tasks.clone());
                let _ = app_for_cb.emit(
                    "resolve-finished",
                    ResolveFinished {
                        ok: true,
                        tasks,
                        error: None,
                        total: total.max(1),
                        resolved: results.len(),
                    },
                );
            }
            Err(e) => {
                let _ = app_for_cb.emit(
                    "resolve-finished",
                    ResolveFinished {
                        ok: false,
                        tasks: vec![],
                        error: Some(e.to_string()),
                        total: total.max(1),
                        resolved: 0,
                    },
                );
            }
        }
    });

    Ok(())
}

/// 构造并发送一条「解析中」进度事件（复用 download-progress，phase = "resolve"）
fn emit_resolve_progress(app: &AppHandle, done: usize, total: usize, title: &str) {
    let percent = if total > 0 {
        done as f64 / total as f64
    } else {
        0.0
    };
    let _ = app.emit(
        "download-progress",
        ProgressEvent {
            phase: "resolve".into(),
            task_id: format!("resolve:{}/{}", done, total),
            title: title.to_string(),
            status: if done >= total {
                "Completed"
            } else {
                "Downloading"
            }
            .into(),
            percent,
            downloaded: 0,
            total: 0,
            speed: 0,
            error: None,
            source_title: title.to_string(),
            output_path: None,
            confidence: None,
            quality_status: None,
            quality_model_id: None,
            quality_output_path: None,
            quality_error: None,
        },
    );
}

/// 发送识别/重命名阶段的任务事件。
fn emit_audio_postprocess(app: &AppHandle, task: &DownloadTask, phase: &str, percent: f64) {
    let _ = app.emit(
        "download-progress",
        ProgressEvent {
            phase: phase.into(),
            task_id: task.id.clone(),
            title: task.title.clone(),
            status: format!("{:?}", task.recognition_status),
            percent,
            downloaded: 0,
            total: 0,
            speed: 0,
            error: task.recognition_error.clone(),
            source_title: if task.source_title.is_empty() {
                task.title.clone()
            } else {
                task.source_title.clone()
            },
            output_path: task.output_path.clone(),
            confidence: task.recognition_result.as_ref().map(|r| r.confidence),
            quality_status: Some(format!("{:?}", task.quality_status)),
            quality_model_id: task.quality_model_id.clone(),
            quality_output_path: task.quality_output_path.clone(),
            quality_error: task.quality_error.clone(),
        },
    );
}

fn emit_quality_postprocess(
    app: &AppHandle,
    task: &DownloadTask,
    percent: f64,
    message: Option<String>,
) {
    let _ = app.emit(
        "download-progress",
        ProgressEvent {
            phase: "enhance".into(),
            task_id: task.id.clone(),
            title: task.title.clone(),
            status: format!("{:?}", task.quality_status),
            percent: percent.clamp(0.0, 1.0),
            downloaded: 0,
            total: 0,
            speed: 0,
            error: task.error.clone(),
            source_title: if task.source_title.is_empty() {
                task.title.clone()
            } else {
                task.source_title.clone()
            },
            output_path: task.output_path.clone(),
            confidence: task.recognition_result.as_ref().map(|r| r.confidence),
            quality_status: Some(format!("{:?}", task.quality_status)),
            quality_model_id: task.quality_model_id.clone(),
            quality_output_path: task.quality_output_path.clone(),
            quality_error: task.quality_error.clone().or(message),
        },
    );
}

fn enhancement_worker_spec() -> Option<WorkerSpec> {
    if let Some(path) = std::env::var_os("AUDIO_AI_WORKER") {
        return Some(WorkerSpec::new(path));
    }
    if std::env::var("AUDIO_AI_USE_FAKE").as_deref() == Ok("1") {
        return WorkerSpec::fake("success");
    }
    WorkerSpec::production()
}

fn quality_output_path(task: &DownloadTask) -> std::path::PathBuf {
    let mut output = audio_rename::staging_path(
        Path::new(&task.output_dir),
        &format!("{}-ai", task.id),
        "flac",
    );
    output.set_extension("flac");
    output
}

const AI_MASTER_DURATION_TOLERANCE_SECONDS: f64 = 0.1;

fn validate_ai_master_output(
    input: &media::AudioProbe,
    output: &media::AudioProbe,
    worker_result: &ai_worker::WorkerResult,
) -> Result<(), WorkerError> {
    if !output.codec_name.eq_ignore_ascii_case("flac") {
        return Err(WorkerError::Protocol(format!(
            "AI master 编码器错误: 期望 flac，实际 {}",
            output.codec_name
        )));
    }
    if worker_result.sample_rate != 48_000 || output.sample_rate != worker_result.sample_rate {
        return Err(WorkerError::Protocol(format!(
            "AI master 采样率不一致: Worker {} Hz，文件 {} Hz",
            worker_result.sample_rate, output.sample_rate
        )));
    }
    if worker_result.channels != input.channels || output.channels != input.channels {
        return Err(WorkerError::Protocol(format!(
            "AI master 声道数不一致: 输入 {}，Worker {}，文件 {}",
            input.channels, worker_result.channels, output.channels
        )));
    }
    if output.duration_seconds <= 0.0
        || (output.duration_seconds - input.duration_seconds).abs()
            > AI_MASTER_DURATION_TOLERANCE_SECONDS
        || (output.duration_seconds - worker_result.duration_seconds).abs()
            > AI_MASTER_DURATION_TOLERANCE_SECONDS
    {
        return Err(WorkerError::Protocol(format!(
            "AI master 时长不一致: 输入 {:.3}s，Worker {:.3}s，文件 {:.3}s",
            input.duration_seconds,
            worker_result.duration_seconds,
            output.duration_seconds
        )));
    }
    Ok(())
}

async fn enhance_audio_task(
    app: &AppHandle,
    task: &mut DownloadTask,
    config: &PythonAiEnhancementConfig,
    staged: &Path,
) -> Result<std::path::PathBuf, WorkerError> {
    let output = quality_output_path(task);
    let input_probe = media::probe_audio(staged).map_err(|error| {
        WorkerError::Protocol(format!("无法探测 AI 输入文件: {error}"))
    })?;
    let Some(spec) = enhancement_worker_spec() else {
        return Err(WorkerError::Spawn(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "找不到 Python AI Worker 或 Python 运行时",
        )));
    };
    let request = AiProcessRequest {
        request_id: format!("{}-ai", task.id),
        input_path: staged.to_string_lossy().into_owned(),
        output_path: output.to_string_lossy().into_owned(),
        model_id: config.model_id.clone(),
        device: config.device.clone(),
        chunk_seconds: config.chunk_seconds,
        overlap_seconds: config.overlap_seconds,
        output_sample_rate: Some(48_000),
        mode: None,
    };
    let task_for_event = task.clone();
    let app_for_event = app.clone();
    let callback: WorkerEventCallback = Arc::new(move |event| {
        let status = match event {
            WorkerEvent::Ready { .. } => "CheckingRuntime",
            WorkerEvent::Progress { phase, .. } if phase == "load_model" => "LoadingModel",
            WorkerEvent::Progress { .. } => "Enhancing",
            WorkerEvent::Completed { .. } => "Completed",
            WorkerEvent::Error { .. } => "Failed",
        };
        let percent = match event {
            WorkerEvent::Progress { percent, .. } => percent / 100.0,
            WorkerEvent::Completed { .. } => 1.0,
            _ => 0.0,
        };
        let mut snapshot = task_for_event.clone();
        snapshot.quality_status = match status {
            "CheckingRuntime" => QualityStatus::CheckingRuntime,
            "LoadingModel" => QualityStatus::LoadingModel,
            "Enhancing" => QualityStatus::Enhancing,
            "Completed" => QualityStatus::Completed,
            _ => QualityStatus::Failed,
        };
        emit_quality_postprocess(&app_for_event, &snapshot, percent, None);
    });

    let run = ai_worker::run_worker_with_callback(&spec, &request, None, Some(callback)).await?;
    if run.result.output_path != output.to_string_lossy()
        || !output.is_file()
        || std::fs::metadata(&output)
            .map(|metadata| metadata.len() == 0)
            .unwrap_or(true)
    {
        return Err(WorkerError::Protocol(
            "AI Worker 返回的输出文件不存在或为空".into(),
        ));
    }
    let output_probe = media::probe_audio(&output).map_err(|error| {
        WorkerError::Protocol(format!("无法探测 AI master: {error}"))
    })?;
    validate_ai_master_output(&input_probe, &output_probe, &run.result)?;
    Ok(output)
}

/// 下载完成后的 AudioOnly 后处理：识别、转码为 MP3、生成最终文件名。
///
/// 识别失败不会让下载任务变成 Failed；任务仍表示下载成功，只通过独立的
/// `recognition_status` 和 `recognition_error` 告知后处理结果。
async fn postprocess_audio_task(
    app: &AppHandle,
    task: &mut DownloadTask,
    fpcalc_path: Option<&Path>,
    config: &AutoRenameConfig,
    quality_config: &PythonAiEnhancementConfig,
) {
    if task.mode != DownloadMode::AudioOnly
        || task.status != crate::biliapi::task::DownloadStatus::Completed
    {
        return;
    }

    let dir = Path::new(&task.output_dir).to_path_buf();
    let source_title = if task.source_title.is_empty() {
        task.title.clone()
    } else {
        task.source_title.clone()
    };
    task.source_title = source_title.clone();
    let staged = task.staged_file();
    let postprocess_plan = audio_postprocess_plan(config.enabled, quality_config.enabled);
    if !staged.exists() {
        task.recognition_status = RecognitionStatus::Failed;
        task.recognition_error = Some(format!("下载完成文件不存在: {}", staged.display()));
        if quality_config.enabled {
            task.quality_status = QualityStatus::Failed;
            task.quality_model_id = Some(quality_config.model_id.clone());
            task.quality_error = Some("AI 增强输入文件不存在".into());
            emit_quality_postprocess(app, task, 1.0, None);
        }
        emit_audio_postprocess(app, task, "recognize", 0.0);
        return;
    }

    let mut stem = audio_rename::fallback_stem(&source_title);
    if postprocess_plan.contains(&AudioPostprocessStage::Recognize) {
        task.recognition_status = RecognitionStatus::Recognizing;
        task.recognition_error = None;
        emit_audio_postprocess(app, task, "recognize", 0.0);
        let identify_result = match fpcalc_path {
            Some(path) => run_identify(&path.to_string_lossy(), &staged.to_string_lossy()).await,
            None => Err(crate::recognizer::AppError::Fingerprint(
                "找不到 fpcalc.exe".into(),
            )),
        };
        match identify_result {
            Ok(info) => {
                let confidence_ok = info.confidence >= config.confidence_threshold;
                let title_ok = !info.title.trim().is_empty();
                task.recognition_result = Some(info.clone());
                if confidence_ok && title_ok {
                    stem = audio_rename::recognized_stem(&info, &source_title, &config.template);
                    // 只有后续 MP3 转码和文件移动都成功后，才标记为 Renamed。
                    task.recognition_status = RecognitionStatus::Recognizing;
                } else {
                    task.recognition_status = if confidence_ok {
                        RecognitionStatus::NoMatch
                    } else {
                        RecognitionStatus::BelowThreshold
                    };
                    task.recognition_error = Some(format!(
                        "识别结果置信度 {:.1}% 未达到 {:.1}% 或缺少曲目标题，已使用原始标题",
                        info.confidence, config.confidence_threshold
                    ));
                }
            }
            Err(e) => {
                task.recognition_status = if e.to_string().contains("未识别到") {
                    RecognitionStatus::NoMatch
                } else {
                    RecognitionStatus::Failed
                };
                task.recognition_error = Some(e.to_string());
            }
        }
    } else {
        task.recognition_status = RecognitionStatus::Disabled;
    }
    emit_audio_postprocess(app, task, "recognize", 1.0);

    let mut source_for_encode = staged.clone();
    if postprocess_plan.contains(&AudioPostprocessStage::Enhance) {
        task.quality_status = QualityStatus::CheckingRuntime;
        task.quality_model_id = Some(quality_config.model_id.clone());
        task.quality_error = None;
        emit_quality_postprocess(app, task, 0.0, Some("正在准备 Python AI 增强".into()));
        match enhance_audio_task(app, task, quality_config, &staged).await {
            Ok(enhanced) => {
                task.quality_status = QualityStatus::Completed;
                task.quality_output_path = Some(enhanced.to_string_lossy().into_owned());
                source_for_encode = enhanced;
                emit_quality_postprocess(app, task, 1.0, Some("AI 增强完成".into()));
            }
            Err(error) => {
                task.quality_status = QualityStatus::Failed;
                task.quality_error = Some(error.to_string());
                let failed_output = quality_output_path(task);
                let _ = std::fs::remove_file(failed_output);
                emit_quality_postprocess(app, task, 1.0, Some("AI 增强失败，已使用原始音频".into()));
            }
        }
    }

    // 识别和 AI 都完成后，最终按用户要求统一转码为 mp3。
    let mp3_part = dir.join(format!(
        ".audio-processor-{}.mp3.part",
        audio_rename::sanitize_component(&task.id).replace(' ', "_")
    ));
    let _ = std::fs::remove_file(&mp3_part);
    if let Err(e) = media::transcode_audio(
        &source_for_encode.to_string_lossy(),
        &mp3_part.to_string_lossy(),
    ) {
        task.recognition_status = RecognitionStatus::RenameFailed;
        task.recognition_error = Some(e.to_string());
        // ffmpeg 不可用时仍保留可播放的原始 m4a，并去掉 .part 后缀。
        match audio_rename::move_to_unique(&staged, &dir, &source_title, "m4a") {
            Ok(path) => {
                task.output_path = Some(path.to_string_lossy().into_owned());
                task.staged_path = None;
            }
            Err(fallback_err) => {
                task.recognition_error = Some(format!("{}；原始文件兜底失败: {}", e, fallback_err));
            }
        }
        emit_audio_postprocess(app, task, "rename", 1.0);
        return;
    }

    emit_audio_postprocess(app, task, "rename", 0.5);
    match audio_rename::move_to_unique(&mp3_part, &dir, &stem, &config.extension) {
        Ok(path) => {
            let _ = std::fs::remove_file(&staged);
            if source_for_encode != staged {
                let _ = std::fs::remove_file(&source_for_encode);
                task.quality_output_path = None;
            }
            task.title = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&stem)
                .to_string();
            task.output_path = Some(path.to_string_lossy().into_owned());
            task.staged_path = None;
            if task.recognition_status == RecognitionStatus::Recognizing {
                task.recognition_status = RecognitionStatus::Renamed;
            }
        }
        Err(e) => {
            task.recognition_status = RecognitionStatus::RenameFailed;
            task.recognition_error = Some(e.clone());
            let _ = std::fs::remove_file(&mp3_part);
            if source_for_encode != staged {
                let _ = std::fs::remove_file(&source_for_encode);
                task.quality_output_path = None;
            }
            if let Ok(path) = audio_rename::move_to_unique(&staged, &dir, &source_title, "m4a") {
                task.output_path = Some(path.to_string_lossy().into_owned());
                task.staged_path = None;
                task.recognition_error = Some(format!("{}；已使用原始 M4A 文件兜底", e));
            }
        }
    }
    emit_audio_postprocess(app, task, "rename", 1.0);
}

/// 输入目标类型
#[derive(Debug)]
enum Target {
    Bv(String),
    Av(i64),
    Collection(String, String), // (mid, season_id)
    Season(String),             // ssid
}

/// 从用户输入字符串识别目标：BV / AV / 链接中的 bvid / 合集 / 番剧
fn identify(input: &str) -> Target {
    let s = input.trim();
    // 纯 BV 号
    if let Some(rest) = s.strip_prefix("BV").or_else(|| s.strip_prefix("bv")) {
        if !rest.is_empty() {
            return Target::Bv(format!("BV{}", rest));
        }
    }
    // 纯 AV 号
    if let Some(rest) = s.strip_prefix("av").or_else(|| s.strip_prefix("AV")) {
        if let Ok(aid) = rest.parse::<i64>() {
            return Target::Av(aid);
        }
    }
    // 链接：提取 query 参数
    if s.contains("bilibili.com") {
        // —— 合集 / 系列页（space.bilibili.com）优先识别 ——
        // 支持 query 形式（?sid=..&mid=..）与路径形式
        // （/channel/collection/detail/{sid} 或 /channel/series/detail/{sid}）
        if let Some(sid) = extract_query(s, "sid").or_else(|| extract_path_token(s, "detail")) {
            let mid = extract_query(s, "mid")
                .or_else(|| extract_mid_from_space(s))
                .unwrap_or_default();
            if !mid.is_empty() {
                return Target::Collection(mid, sid);
            }
        }

        if let Some(ssid) = extract_query(s, "ssid")
            .or_else(|| extract_path_token(s, "ss").or_else(|| extract_prefixed_token(s, "ss")))
        {
            return Target::Season(ssid);
        }
        if let Some(ep) = extract_query(s, "ep_id")
            .or_else(|| extract_path_token(s, "ep").or_else(|| extract_prefixed_token(s, "ep")))
        {
            // ep 也需要 season 信息，但 resolve_season 仅接受 ssid；
            // 简化：ep 直接当作 ss 不可用，这里回退用 bvid 解析
            if let Some(bvid) = extract_query(s, "bvid") {
                return Target::Bv(bvid);
            }
            // ep 单独出现：尝试从链接拿 bvid
            if let Some(bvid) = extract_bvid_from_url(s) {
                return Target::Bv(bvid);
            }
            // 退化为 Season（ssid 缺失时由 video 层报错）
            return Target::Season(ep);
        }
        if let Some(bvid) = extract_query(s, "bvid").or_else(|| extract_bvid_from_url(s)) {
            return Target::Bv(bvid);
        }
        if let Some(mid) = extract_query(s, "mid") {
            if let Some(sid) = extract_query(s, "sid") {
                return Target::Collection(mid, sid);
            }
        }
    }
    // 兜底：当作 BV 号
    Target::Bv(s.to_string())
}

/// 从 space.bilibili.com 路径提取 UP 主 mid：
/// 形如 `space.bilibili.com/123456/...` 中的第一段数字
fn extract_mid_from_space(url: &str) -> Option<String> {
    let idx = url.find("space.bilibili.com")?;
    let after = &url[idx + "space.bilibili.com".len()..];
    let rest = after.trim_start_matches('/');
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let tok = &rest[..end];
    if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
        Some(tok.to_string())
    } else {
        None
    }
}

/// 从 URL 路径提取形如 /ss123/ 或 /ep123/ 的 token
fn extract_path_token(url: &str, prefix: &str) -> Option<String> {
    let pat = format!("/{}/", prefix);
    if let Some(pos) = url.find(&pat) {
        let after = &url[pos + pat.len()..];
        let end = after.find('/').unwrap_or(after.len());
        let tok = &after[..end];
        if !tok.is_empty() {
            return Some(tok.to_string());
        }
    }
    None
}

/// 从 URL 路径提取前缀连写 token，如 `/ss12345`、`/ep678` 或 `/ss12345/`
/// （前缀后紧跟数字，直到下一个分隔符）
fn extract_prefixed_token(url: &str, prefix: &str) -> Option<String> {
    let pat = format!("/{}", prefix);
    let pos = url.find(&pat)?;
    let after = &url[pos + pat.len()..];
    let end = after.find(['/', '?', '#', '&']).unwrap_or(after.len());
    let tok = &after[..end];
    if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
        Some(tok.to_string())
    } else {
        None
    }
}

/// 从 query 提取参数值
fn extract_query(url: &str, key: &str) -> Option<String> {
    let marker = format!("{}=", key);
    let idx = url.find(&marker)?;
    let after = &url[idx + marker.len()..];
    let end = after.find(['&', '#', '/']).unwrap_or(after.len());
    let v = &after[..end];
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

/// 从 URL 提取 BV 号（形如 BV1xx...）
fn extract_bvid_from_url(url: &str) -> Option<String> {
    let idx = url.find("BV")?;
    let after = &url[idx..];
    let end = after.find(['/', '?', '#', '&']).unwrap_or(after.len());
    let cand = &after[..end];
    if cand.len() > 2 {
        Some(cand.to_string())
    } else {
        None
    }
}

/// AV 号转 BV 号（B 站 base58 算法；仅用于链接场景）
fn bv_from_aid(aid: i64) -> String {
    // 简化：直接用 av 号回退；实际解析由 view 接口处理。
    // 这里我们返回空串再让 get_video_info 失败更明确，但为兼容，
    // video 层 get_video_info 仅接受 bvid。故这里做标准转换。
    const TABLE: &[u8] = b"fcCAfED2BxB7H9KIvMNT3nrS8Lp5gG4a1doe0j6mZQuyOhiktJWzlYwVPUX5sR";
    let xor = 23442827791579u64;
    let base = 58u64;
    let mut bv = [b'B', b'V', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut tmp = aid as u64 ^ xor;
    let mut i = 0;
    while tmp > 0 {
        let rem = (tmp % base) as usize;
        bv[11 - i] = TABLE[rem];
        tmp /= base;
        i += 1;
    }
    // 按 B 站换位表重排
    let swap = |arr: &mut [u8], a: usize, b: usize| {
        arr.swap(a, b);
    };
    swap(&mut bv, 0, 9);
    swap(&mut bv, 3, 10);
    swap(&mut bv, 6, 4);
    swap(&mut bv, 7, 2);
    swap(&mut bv, 8, 1);
    String::from_utf8_lossy(&bv).into_owned()
}

/// 启动已解析任务的并发下载（后台运行，进度经事件推送）
#[tauri::command]
pub async fn bili_start_download(
    input: StartDownloadInput,
    app: AppHandle,
    state: State<'_, BiliState>,
) -> Result<Vec<String>, String> {
    let mut tasks = state.snapshot_tasks();
    if tasks.is_empty() {
        return Err("没有可下载的任务，请先调用 bili_resolve".to_string());
    }
    // 仅下载前端勾选的任务（task_ids 指定）；为 None 时下载全部
    if let Some(ids) = &input.task_ids {
        if ids.is_empty() {
            return Err("未勾选任何任务".to_string());
        }
        let id_set: std::collections::HashSet<&String> = ids.iter().collect();
        tasks.retain(|t| id_set.contains(&t.id));
        if tasks.is_empty() {
            return Err("未勾选任何任务".to_string());
        }
    }
    // 下载并发度：用户显式指定优先；否则走统一入口（默认 3，受 BILI_CONCURRENCY 覆盖）
    let concurrency = input
        .concurrency
        .unwrap_or_else(|| video::resolve_concurrency(3))
        .max(1);
    let client = Arc::new(crate::http_client::client::HttpClient::new());
    let rename_config = AutoRenameConfig {
        enabled: input.auto_rename.unwrap_or(true),
        confidence_threshold: input.confidence_threshold.unwrap_or(70.0).clamp(0.0, 100.0),
        ..AutoRenameConfig::default()
    };
    let quality_enabled = input.python_ai_enhancement_enabled.unwrap_or(false);
    let quality_model_id = input
        .python_ai_model_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        .unwrap_or("audiosr-basic")
        .to_string();
    let (default_chunk_seconds, default_overlap_seconds) = if quality_model_id == "audiosr-basic" {
        (10.24, 1.28)
    } else {
        (20.0, 2.0)
    };
    let quality_config = PythonAiEnhancementConfig {
        enabled: quality_enabled,
        model_id: quality_model_id,
        device: input.python_ai_device.unwrap_or_else(|| "auto".into()),
        chunk_seconds: input.python_ai_chunk_seconds.unwrap_or(default_chunk_seconds),
        overlap_seconds: input.python_ai_overlap_seconds.unwrap_or(default_overlap_seconds),
    };
    validate_python_ai_config(&quality_config)?;
    // fpcalc 缺失不阻断下载，后处理会转为 MP3，并保留可读错误状态。
    let fpcalc_path = crate::commands::fpcalc_path(&app).ok();

    // 每次启动下载前重置控制句柄（清空上一次的暂停/停止信号）
    let control = state.reset_download_control();

    let app_for_cb = app.clone();
    let prog_cb: Option<
        Arc<dyn Fn(&DownloadTask, crate::http_client::types::Progress) + Send + Sync>,
    > = Some(Arc::new(
        move |task: &DownloadTask, p: crate::http_client::types::Progress| {
            // 通过 AppHandle 取共享状态（AppHandle 为 'static，闭包内安全）
            let st = app_for_cb.state::<BiliState>();
            st.apply_results(std::slice::from_ref(task));
            let payload = ProgressEvent {
                phase: "download".into(),
                task_id: task.id.clone(),
                title: task.title.clone(),
                // 使用枚举名（如 "Downloading"/"Cancelled"/"Paused"），
                // 便于前端按真实状态实时更新任务展示（与 tasks 列表的 status 字段一致）。
                status: format!("{:?}", task.status),
                percent: p.percent,
                downloaded: p.downloaded,
                total: p.total.unwrap_or(0),
                speed: p.speed,
                error: task.error.clone(),
                source_title: if task.source_title.is_empty() {
                    task.title.clone()
                } else {
                    task.source_title.clone()
                },
                output_path: task.output_path.clone(),
                confidence: task.recognition_result.as_ref().map(|r| r.confidence),
                quality_status: Some(format!("{:?}", task.quality_status)),
                quality_model_id: task.quality_model_id.clone(),
                quality_output_path: task.quality_output_path.clone(),
                quality_error: task.quality_error.clone(),
            };
            let _ = app_for_cb.emit("download-progress", payload);
        },
    ));

    let mut tasks_ref = tasks.clone();
    // 若下载命令显式指定了目录，覆盖各任务的输出目录（优先于解析时设定的值）
    if let Some(dir) = input.output_dir.clone().filter(|d| !d.is_empty()) {
        for t in tasks_ref.iter_mut() {
            t.output_dir = dir.clone();
        }
    }
    let app2 = app.clone();
    tauri::async_runtime::spawn(async move {
        let results = crate::biliapi::task::run_batch(
            client,
            &mut tasks_ref,
            concurrency,
            prog_cb,
            Some(&control),
        )
        .await;
        for task in tasks_ref.iter_mut() {
            if task.mode == DownloadMode::AudioOnly {
                task.quality_status = if quality_config.enabled {
                    QualityStatus::Pending
                } else {
                    QualityStatus::Disabled
                };
                task.quality_model_id = quality_config
                    .enabled
                    .then(|| quality_config.model_id.clone());
                task.quality_output_path = None;
                task.quality_error = None;
            }
            postprocess_audio_task(
                &app2,
                task,
                fpcalc_path.as_deref(),
                &rename_config,
                &quality_config,
            )
            .await;
            app2.state::<BiliState>()
                .apply_results(std::slice::from_ref(task));
        }
        let failed: Vec<_> = results
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_err())
            .collect();
        let renamed = tasks_ref
            .iter()
            .filter(|t| t.recognition_status == RecognitionStatus::Renamed)
            .count();
        let fallback = tasks_ref
            .iter()
            .filter(|t| {
                t.mode == DownloadMode::AudioOnly && {
                    matches!(
                        t.recognition_status,
                        RecognitionStatus::Disabled
                            | RecognitionStatus::NoMatch
                            | RecognitionStatus::BelowThreshold
                            | RecognitionStatus::Failed
                            | RecognitionStatus::RenameFailed
                    )
                }
            })
            .count();
        let recognize_failed = tasks_ref
            .iter()
            .filter(|t| {
                t.mode == DownloadMode::AudioOnly && {
                    matches!(
                        t.recognition_status,
                        RecognitionStatus::NoMatch
                            | RecognitionStatus::BelowThreshold
                            | RecognitionStatus::Failed
                            | RecognitionStatus::RenameFailed
                    )
                }
            })
            .count();
        let enhanced = tasks_ref
            .iter()
            .filter(|t| t.quality_status == QualityStatus::Completed)
            .count();
        let enhance_failed = tasks_ref
            .iter()
            .filter(|t| t.quality_status == QualityStatus::Failed)
            .count();

        // 下载完成后写入通用历史库（每条任务一条，含最终状态/错误）
        let hist_dir = crate::commands::history_dir(&app2);
        if let Ok(conn) = crate::history::open_db(&hist_dir) {
            for t in tasks_ref.iter() {
                let subtitle = format!(
                    "{} · {} · {}",
                    mode_label(t.mode),
                    status_label(t.status),
                    recognition_label(t.recognition_status)
                );
                let payload = serde_json::to_string(t).unwrap_or_default();
                let file_path = t.output_file().to_string_lossy().to_string();
                if let Err(e) = crate::history::insert(
                    &conn,
                    crate::history::HistoryKind::Download,
                    &t.title,
                    &subtitle,
                    &payload,
                    &file_path,
                ) {
                    eprintln!("[history] 写入下载历史失败: {e}");
                }
            }
        }

        let _ = app2.emit(
            "download-finished",
            serde_json::json!({
                "ok": failed.is_empty(),
                "failed": failed.len(),
                "renamed": renamed,
                "recognize_failed": recognize_failed,
                "fallback": fallback,
                "enhanced": enhanced,
                "enhance_failed": enhance_failed
            }),
        );
    });

    Ok(tasks.iter().map(|t| t.id.clone()).collect())
}

/// 暂停下载：置位暂停信号，进行中的任务在完成当前文件后进入 `Paused`（保留已下载部分，可续传）。
#[tauri::command]
pub fn bili_pause_download(state: State<'_, BiliState>) -> Result<(), String> {
    state.pause_download();
    Ok(())
}

/// 停止下载：置位停止信号，立即中断下载并删除已下载部分（状态置 `Cancelled`）。
#[tauri::command]
pub fn bili_stop_download(app: AppHandle, state: State<'_, BiliState>) -> Result<(), String> {
    state.stop_download();
    // 立即回写任务状态快照（被取消的任务在 run_batch 收尾前可能仍是旧状态），
    // 让前端即时感知「取消中」。最终状态由后台任务收尾时经事件推送。
    let _ = app.emit("download-control", serde_json::json!({ "action": "stop" }));
    Ok(())
}

/// 查询当前任务列表与状态
#[tauri::command]
pub fn bili_list_tasks(state: State<'_, BiliState>) -> Result<Vec<DownloadTask>, String> {
    Ok(state.snapshot_tasks())
}

/// 生成登录二维码
#[tauri::command]
pub async fn bili_login_qr() -> Result<LoginQr, String> {
    let info: QrInfo = login::new_qr_info().await.map_err(|e| e.to_string())?;
    let svg = login::generate_qr_svg(&info.url).map_err(|e| e.to_string())?;
    Ok(LoginQr {
        qr_svg: format!("data:image/svg+xml;charset=utf-8,{}", urlencoding(&svg)),
        qr_key: info.qrcode_key,
    })
}

/// 对 SVG 字符串做 URL 编码以嵌入 data URL（避免 `#`/`"` 等破坏属性）
fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", b));
            }
        }
    }
    out
}

/// 轮询登录二维码状态；成功则自动持久化 SESSDATA
#[tauri::command]
pub async fn bili_login_poll(
    qr_key: String,
    state: State<'_, BiliState>,
) -> Result<LoginState, String> {
    let (status, sessdata) = login::get_qr_status(&qr_key)
        .await
        .map_err(|e| e.to_string())?;
    if status.code == crate::biliapi::types::QR_SUCCESS {
        login::persist_login(state.config_dir_opt().as_deref(), sessdata.as_deref());
        Ok(LoginState {
            authed: true,
            message: status.message,
        })
    } else {
        Ok(LoginState {
            authed: false,
            message: status.message,
        })
    }
}

/// 校验当前是否已登录
#[tauri::command]
pub async fn bili_check_login(state: State<'_, BiliState>) -> Result<bool, String> {
    Ok(login::load_and_check(state.config_dir_opt().as_deref())
        .await
        .map_err(|e| e.to_string())?
        .is_some())
}

/// 获取已登录用户的 B 站资料（昵称 + 头像）。未登录时返回 `None`。
#[tauri::command]
pub async fn bili_user_info(state: State<'_, BiliState>) -> Result<Option<BiliUserInfo>, String> {
    // 先按标准流程校验登录态（失效会被清除）。
    let sessdata = match login::load_and_check(state.config_dir_opt().as_deref())
        .await
        .map_err(|e| e.to_string())?
    {
        Some(s) => s,
        // 兜底：直接读落盘 SESSDATA（不触发清除），避免 `refreshLogin` 已校验过、
        // 短时间内二次 `load_and_check` 误判为失效而拿不到资料。
        None => match crate::biliapi::storage::load_sessdata(state.config_dir_opt().as_deref()) {
            Some(s) if !s.is_empty() => s,
            _ => return Ok(None),
        },
    };
    let info = login::fetch_user_info(&sessdata)
        .await
        .map_err(|e| e.to_string())?;
    Ok(Some(BiliUserInfo {
        name: info.name,
        face: info.face,
    }))
}

/// 登出：清除持久化登录态
#[tauri::command]
pub fn bili_logout(state: State<'_, BiliState>) -> Result<(), String> {
    let dir = state.config_dir_opt();
    crate::biliapi::storage::clear_sessdata(dir.as_deref());
    crate::biliapi::storage::clear_wbi(dir.as_deref());
    crate::biliapi::wbi_cache::clear_cache();
    state.set_tasks(Vec::new());
    Ok(())
}

/// 将 `DownloadStatus` 转为可读中文标签
fn status_label(s: crate::biliapi::task::DownloadStatus) -> String {
    use crate::biliapi::task::DownloadStatus;
    match s {
        DownloadStatus::Pending => "等待中",
        DownloadStatus::Downloading => "下载中",
        DownloadStatus::Completed => "已完成",
        DownloadStatus::Failed => "失败",
        DownloadStatus::Paused => "已暂停",
        DownloadStatus::Cancelled => "已停止",
    }
    .to_string()
}

/// 将 `DownloadMode` 转为可读中文标签
fn mode_label(m: crate::biliapi::task::DownloadMode) -> String {
    use crate::biliapi::task::DownloadMode;
    match m {
        DownloadMode::AudioOnly => "仅音频",
        DownloadMode::VideoOnly => "仅视频",
        DownloadMode::Merge => "音视频合并",
    }
    .to_string()
}

/// 将识别状态转为历史记录中的可读标签。
fn recognition_label(s: RecognitionStatus) -> &'static str {
    match s {
        RecognitionStatus::Disabled => "未启用识别",
        RecognitionStatus::Pending => "等待识别",
        RecognitionStatus::Recognizing => "识别中",
        RecognitionStatus::Renamed => "已重命名",
        RecognitionStatus::NoMatch => "未匹配",
        RecognitionStatus::BelowThreshold => "置信度不足",
        RecognitionStatus::Failed => "识别失败",
        RecognitionStatus::RenameFailed => "重命名失败",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_ai_config_defaults_to_disabled_audio_sr_settings() {
        let config = PythonAiEnhancementConfig {
            enabled: false,
            model_id: "audiosr-basic".into(),
            device: "auto".into(),
            chunk_seconds: 10.24,
            overlap_seconds: 1.28,
        };
        assert!(!config.enabled);
        assert!(validate_python_ai_config(&config).is_ok());
    }

    #[test]
    fn audio_postprocess_plan_disables_both_optional_stages() {
        assert_eq!(
            audio_postprocess_plan(false, false),
            vec![AudioPostprocessStage::Encode]
        );
    }

    #[test]
    fn audio_postprocess_plan_runs_only_recognition_when_ai_is_disabled() {
        assert_eq!(
            audio_postprocess_plan(true, false),
            vec![AudioPostprocessStage::Recognize, AudioPostprocessStage::Encode]
        );
    }

    #[test]
    fn audio_postprocess_plan_runs_only_ai_when_recognition_is_disabled() {
        assert_eq!(
            audio_postprocess_plan(false, true),
            vec![AudioPostprocessStage::Enhance, AudioPostprocessStage::Encode]
        );
    }

    #[test]
    fn audio_postprocess_plan_runs_recognition_before_ai() {
        assert_eq!(
            audio_postprocess_plan(true, true),
            vec![
                AudioPostprocessStage::Recognize,
                AudioPostprocessStage::Enhance,
                AudioPostprocessStage::Encode
            ]
        );
    }

    #[test]
    fn python_ai_config_rejects_invalid_device_and_overlap() {
        let invalid_device = PythonAiEnhancementConfig {
            enabled: true,
            model_id: "audiosr-basic".into(),
            device: "metal".into(),
            chunk_seconds: 10.24,
            overlap_seconds: 1.28,
        };
        assert!(validate_python_ai_config(&invalid_device).is_err());

        let invalid_overlap = PythonAiEnhancementConfig {
            device: "cpu".into(),
            overlap_seconds: 10.24,
            ..invalid_device
        };
        assert!(validate_python_ai_config(&invalid_overlap).is_err());
    }

    #[test]
    fn quality_output_path_is_flac_and_is_hidden_from_final_outputs() {
        let task = DownloadTask {
            id: "BV1abc#1".into(),
            title: "source".into(),
            source_title: "source".into(),
            video_url: None,
            audio_url: Some("audio".into()),
            mode: DownloadMode::AudioOnly,
            output_dir: std::env::temp_dir().to_string_lossy().into_owned(),
            status: crate::biliapi::task::DownloadStatus::Completed,
            error: None,
            recognition_status: RecognitionStatus::Disabled,
            recognition_result: None,
            recognition_error: None,
            quality_status: QualityStatus::Pending,
            quality_model_id: Some("audiosr-basic".into()),
            quality_output_path: None,
            quality_error: None,
            staged_path: None,
            output_path: None,
            group: None,
            cover: None,
        };
        let output = quality_output_path(&task);
        assert_eq!(output.extension().and_then(|value| value.to_str()), Some("flac"));
        assert!(output
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.starts_with(".audio-processor-")));
    }

    fn audio_probe(codec_name: &str, sample_rate: u32, channels: u16, duration: f64) -> media::AudioProbe {
        media::AudioProbe {
            codec_name: codec_name.into(),
            sample_rate,
            channels,
            duration_seconds: duration,
        }
    }

    fn ai_worker_result(sample_rate: u32, channels: u16, duration: f64) -> ai_worker::WorkerResult {
        ai_worker::WorkerResult {
            output_path: "master.flac".into(),
            model_id: "audiosr-basic".into(),
            model_version: "test".into(),
            sample_rate,
            channels,
            duration_seconds: duration,
            peak_db: -6.0,
        }
    }

    #[test]
    fn ai_master_validation_accepts_matching_flac() {
        let input = audio_probe("aac", 44_100, 2, 10.24);
        let output = audio_probe("flac", 48_000, 2, 10.25);
        let result = ai_worker_result(48_000, 2, 10.24);
        assert!(validate_ai_master_output(&input, &output, &result).is_ok());
    }

    #[test]
    fn ai_master_validation_rejects_invalid_output_parameters() {
        let input = audio_probe("aac", 44_100, 2, 10.24);
        let result = ai_worker_result(48_000, 2, 10.24);
        for output in [
            audio_probe("wav", 48_000, 2, 10.24),
            audio_probe("flac", 44_100, 2, 10.24),
            audio_probe("flac", 48_000, 1, 10.24),
            audio_probe("flac", 48_000, 2, 9.0),
        ] {
            assert!(validate_ai_master_output(&input, &output, &result).is_err());
        }
    }

    #[test]
    fn test_identify_collection_query() {
        let t = identify(
            "https://space.bilibili.com/123456/channel/collection/detail?sid=789&mid=123456",
        );
        match t {
            Target::Collection(mid, sid) => {
                assert_eq!(mid, "123456");
                assert_eq!(sid, "789");
            }
            _ => panic!("期望 Collection，实际 {:?}", t),
        }
    }

    #[test]
    fn test_identify_collection_path() {
        let t = identify("https://space.bilibili.com/123456/channel/series/detail/789");
        match t {
            Target::Collection(mid, sid) => {
                assert_eq!(mid, "123456");
                assert_eq!(sid, "789");
            }
            _ => panic!("期望 Collection，实际 {:?}", t),
        }
    }

    #[test]
    fn test_identify_collection_mid_query_only() {
        // mid 来自 query，sid 来自 query
        let t = identify("https://space.bilibili.com/channel/collection/detail?sid=789&mid=555");
        match t {
            Target::Collection(mid, sid) => {
                assert_eq!(mid, "555");
                assert_eq!(sid, "789");
            }
            _ => panic!("期望 Collection，实际 {:?}", t),
        }
    }

    #[test]
    fn test_identify_bv_still_works() {
        let t = identify("BV16w4m1277k");
        match t {
            Target::Bv(b) => assert_eq!(b, "BV16w4m1277k"),
            _ => panic!("期望 Bv，实际 {:?}", t),
        }
    }

    #[test]
    fn test_identify_video_page_not_collection() {
        // 普通视频页（无 sid/mid）不误判为合集
        let t = identify("https://www.bilibili.com/video/BV16w4m1277k?vd_source=abc");
        match t {
            Target::Bv(b) => assert_eq!(b, "BV16w4m1277k"),
            _ => panic!("期望 Bv，实际 {:?}", t),
        }
    }

    #[test]
    fn test_identify_season_ssid() {
        let t = identify("https://www.bilibili.com/bangumi/play/?ssid=12345");
        match t {
            Target::Season(ss) => assert_eq!(ss, "12345"),
            _ => panic!("期望 Season，实际 {:?}", t),
        }
    }

    #[test]
    fn test_identify_season_ss_prefixed_path() {
        // 番剧路径连写形式 /bangumi/play/ss12345
        let t = identify("https://www.bilibili.com/bangumi/play/ss12345");
        match t {
            Target::Season(ss) => assert_eq!(ss, "12345"),
            _ => panic!("期望 Season，实际 {:?}", t),
        }
    }
}
