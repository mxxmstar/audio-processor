//! 音频质量处理相关 Tauri 命令。
//!
//! 当前只提供 AI Worker 运行时握手检查。真实模型和音频处理命令在
//! Worker 协议稳定后继续实现。

use crate::audio_quality::ai_worker::{
    self, AiProcessRequest, WorkerError, WorkerEvent, WorkerEventCallback, WorkerSpec,
    PROTOCOL_VERSION,
};
use crate::audio_quality::backend::{
    self, default_output_sample_rate, Backend, BackendDefaults, FLASHSR_SAMPLE_RATE, ModelInfo,
    VOICEFIXER_SAMPLE_RATE,
};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioQualityTask {
    pub id: String,
    pub input_path: String,
    pub output_path: String,
    pub status: String,
    pub phase: String,
    pub percent: f64,
    pub model_id: String,
    pub device: String,
    pub worker_kind: String,
    pub message: Option<String>,
    pub error: Option<String>,
    pub output_sample_rate: Option<u32>,
    pub output_channels: Option<u16>,
    pub output_duration_seconds: Option<f64>,
}

#[derive(Default, Clone)]
pub struct AudioQualityState {
    inner: Arc<Mutex<AudioQualityStore>>,
}

#[derive(Default)]
struct AudioQualityStore {
    tasks: HashMap<String, AudioQualityTask>,
    cancel_flags: HashMap<String, Arc<AtomicBool>>,
}

impl AudioQualityState {
    fn insert(&self, task: AudioQualityTask, cancel: Arc<AtomicBool>) {
        let mut store = self.inner.lock().expect("audio quality state poisoned");
        store.cancel_flags.insert(task.id.clone(), cancel);
        store.tasks.insert(task.id.clone(), task);
    }

    fn update<F>(&self, id: &str, update: F) -> Option<AudioQualityTask>
    where
        F: FnOnce(&mut AudioQualityTask),
    {
        let mut store = self.inner.lock().expect("audio quality state poisoned");
        let task = store.tasks.get_mut(id)?;
        update(task);
        Some(task.clone())
    }

    fn cancel(&self, id: &str) -> bool {
        let store = self.inner.lock().expect("audio quality state poisoned");
        let Some(flag) = store.cancel_flags.get(id) else {
            return false;
        };
        flag.store(true, Ordering::Relaxed);
        true
    }

    fn list(&self) -> Vec<AudioQualityTask> {
        let store = self.inner.lock().expect("audio quality state poisoned");
        let mut tasks: Vec<_> = store.tasks.values().cloned().collect();
        tasks.sort_by(|a, b| a.id.cmp(&b.id));
        tasks
    }

    fn remove_cancel(&self, id: &str) {
        self.inner
            .lock()
            .expect("audio quality state poisoned")
            .cancel_flags
            .remove(id);
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiRuntimeCheck {
    pub available: bool,
    pub production_ready: bool,
    pub protocol_version: u32,
    pub worker_kind: String,
    pub program: Option<String>,
    pub worker_version: Option<String>,
    pub models: Vec<String>,
    pub model_errors: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioQualityModelInstallResult {
    pub model_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioQualityStartInput {
    pub input_path: String,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub chunk_seconds: Option<f64>,
    #[serde(default)]
    pub overlap_seconds: Option<f64>,
    #[serde(default)]
    pub output_sample_rate: Option<u32>,
    /// VoiceFixer 修复模式（0 原始 / 1 预处理去高频 / 2 训练模式）；其它后端忽略。
    #[serde(default)]
    pub mode: Option<u32>,
}

/// 检查 Python AI Worker 是否能启动并完成 JSONL ready 握手。
///
/// `model_id` 选填，决定**主探测**哪一个后端（FlashSR 用 `.venv-flashsr` 的
/// 独立 Worker，AudioSR / DeepFilterNet 用 `.venv` 的 Worker），用于
/// `worker_kind` / 版本展示。但不同后端运行在**不同的** Python 运行时，各自
/// 探测。若只探测当前选中的那一个，选 FlashSR 时 AudioSR / DeepFilterNet 会被
/// 误判为「不可用」。因此这里额外探测另一后端并把所有可用模型取**并集**返回，
/// UI 才能正确标注每个模型的可用性。
///
/// `AUDIO_AI_WORKER` 可指定未来打包后的 Worker 可执行文件；未配置且
/// `AUDIO_AI_USE_FAKE=1` 时使用仓库内 fake Worker，仅用于开发协议验证，并通过
/// `productionReady=false` 明确标记其不是实际 AI 模型运行时。
#[tauri::command]
pub async fn audio_quality_check_ai_runtime(
    model_id: Option<String>,
) -> Result<AiRuntimeCheck, String> {
    let model_id = model_id.unwrap_or_else(|| "audiosr-basic".into());

    // 主探测：针对当前选中的 model_id，用于 worker_kind / 版本 / 主可用性展示。
    // 连对应的 Worker 规格都解析不出来（通常发生在找不到 Python 运行时），直接
    // 返回「不可用」，无需再探测其它后端。
    let primary_spec = match backend::select_worker_spec(&model_id) {
        Ok(spec) => spec,
        Err(_) => return Ok(unavailable_runtime("找不到 Python AI Worker 或 Python 运行时")),
    };
    let primary_ready = ai_worker::probe_worker(&primary_spec.0).await.ok();
    let worker_kind = primary_spec.1.clone();
    let program = primary_spec.0.program.to_string_lossy().into_owned();

    // 合并全部后端 Worker 的可用模型，避免「仅探测当前选中 worker 时其它后端
    // 被误判为不可用」。某一后端进程起不来时跳过它的模型，不影响其它后端。
    let mut models: Vec<String> = Vec::new();
    let mut model_errors: Vec<String> = Vec::new();
    if let Some(ready) = &primary_ready {
        merge_worker_models(&mut models, &mut model_errors, ready);
    }
    for spec in [
        WorkerSpec::flashsr(),
        WorkerSpec::production(),
        WorkerSpec::hifigan(),
        WorkerSpec::voicefixer(),
    ]
    .into_iter()
    .flatten()
    {
        if let Ok(ready) = ai_worker::probe_worker(&spec).await {
            merge_worker_models(&mut models, &mut model_errors, &ready);
        }
    }

    let available = primary_ready.is_some() || !models.is_empty();
    let production_ready = !models.is_empty();
    Ok(AiRuntimeCheck {
        available,
        production_ready,
        protocol_version: PROTOCOL_VERSION,
        worker_kind,
        program: Some(program),
        worker_version: primary_ready.as_ref().map(|r| r.worker_version.clone()),
        models,
        model_errors,
        error: if primary_ready.is_none() {
            Some("主模型运行时探测失败（其它后端仍可能可用）".into())
        } else {
            None
        },
    })
}

/// 把某个 Worker 探测到的可用模型合并进累计集合（去重）。
fn merge_worker_models(
    models: &mut Vec<String>,
    errors: &mut Vec<String>,
    ready: &ai_worker::WorkerReady,
) {
    for m in &ready.models {
        if !models.contains(m) {
            models.push(m.clone());
        }
    }
    for e in &ready.model_errors {
        if !errors.contains(e) {
            errors.push(e.clone());
        }
    }
}

/// 列出 manifest.json 中全部可用模型及其后端，供前端渲染可选模型。
#[tauri::command]
pub fn audio_quality_list_models() -> Result<Vec<ModelInfo>, String> {
    Ok(backend::list_models())
}

/// 显式下载并校验模型。模型安装不会由音频处理任务自动触发。
///
/// `workers` 控制断点 Range 下载并发数，限制在 1..=32；模型路径和
/// SHA-256 由 Python 安装器从 manifest.json 读取，Rust 不接受外部 URL。
#[tauri::command(rename_all = "snake_case")]
pub async fn audio_quality_download_model(
    model_id: String,
    workers: Option<u32>,
) -> Result<AudioQualityModelInstallResult, String> {
    let model_id = model_id.trim().to_string();
    if model_id.is_empty() {
        return Err("modelId 不能为空".into());
    }
    let workers = workers.unwrap_or(8).clamp(1, 32);
    let Some(spec) = WorkerSpec::model_manager() else {
        return Err("找不到 Python AI 模型安装器或 Python 运行时".into());
    };
    let progress: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(|line| {
        eprintln!("[audio-ai-model] {line}");
    });
    ai_worker::install_model(&spec, &model_id, workers, Some(progress))
        .await
        .map_err(|error| format!("模型安装失败: {error}"))?;
    Ok(AudioQualityModelInstallResult {
        model_id,
        status: "installed".into(),
        message: "模型已下载并通过 SHA-256 校验".into(),
    })
}

fn unavailable_runtime(error: &str) -> AiRuntimeCheck {
    AiRuntimeCheck {
        available: false,
        production_ready: false,
        protocol_version: PROTOCOL_VERSION,
        worker_kind: "unavailable".into(),
        program: None,
        worker_version: None,
        models: Vec::new(),
        model_errors: Vec::new(),
        error: Some(error.into()),
    }
}

/// 启动一个后台 AI 处理任务。真实 Worker 通过 `AUDIO_AI_WORKER` 配置；fake
/// Worker 只能通过 `AUDIO_AI_USE_FAKE=1` 显式启用，避免生成不可播放的测试文件。
#[tauri::command]
pub fn audio_quality_start(
    input: AudioQualityStartInput,
    app: AppHandle,
    state: State<'_, AudioQualityState>,
) -> Result<AudioQualityTask, String> {
    let input_path = PathBuf::from(&input.input_path);
    if !input_path.is_file() {
        return Err(format!("输入音频不存在: {}", input_path.display()));
    }

    let output_path =
        allocate_output_path(&input_path, input.output_path.as_deref().map(Path::new))?;
    if output_path == input_path {
        return Err("输出路径不能覆盖输入音频".into());
    }
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("创建输出目录失败: {}: {e}", parent.display()))?;
    }

    let model_id = input.model_id.unwrap_or_else(|| "audiosr-basic".into());
    let backend = backend::resolve_backend(&model_id);
    // 采样率约束按后端区分：FlashSR 只能 48 kHz、VoiceFixer 原生 44.1 kHz（R2）。
    validate_output_sample_rate(backend, input.output_sample_rate)?;
    let (spec, worker_kind) = backend::select_worker_spec(&model_id)?;
    let id = format!(
        "aq-{}-{}",
        chrono_like_timestamp(),
        NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed)
    );
    let defaults = BackendDefaults::for_backend(backend);
    let request = AiProcessRequest {
        request_id: id.clone(),
        input_path: input_path.to_string_lossy().into_owned(),
        output_path: output_path.to_string_lossy().into_owned(),
        model_id,
        device: input.device.unwrap_or_else(|| "auto".into()),
        chunk_seconds: input.chunk_seconds.unwrap_or(defaults.chunk_seconds),
        overlap_seconds: input.overlap_seconds.unwrap_or(defaults.overlap_seconds),
        // 不能一律填 48 kHz：VoiceFixer 原生 44.1 kHz，填 48k 会被 Python 侧拒绝
        output_sample_rate: input
            .output_sample_rate
            .or(Some(default_output_sample_rate(backend))),
        mode: input.mode,
    };
    validate_request(&request)?;

    let cancel = Arc::new(AtomicBool::new(false));
    let task = AudioQualityTask {
        id: id.clone(),
        input_path: request.input_path.clone(),
        output_path: request.output_path.clone(),
        status: "queued".into(),
        phase: "queued".into(),
        percent: 0.0,
        model_id: request.model_id.clone(),
        device: request.device.clone(),
        worker_kind,
        message: None,
        error: None,
        output_sample_rate: None,
        output_channels: None,
        output_duration_seconds: None,
    };
    state.insert(task.clone(), cancel.clone());

    let state_for_task = (*state).clone();
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        let _ = state_for_task.update(&id, |task| {
            task.status = "processing".into();
            task.phase = "starting".into();
            task.message = Some("正在启动 Python AI Worker".into());
        });
        emit_task_progress(&app_for_task, &state_for_task, &id);

        let state_for_event = state_for_task.clone();
        let app_for_event = app_for_task.clone();
        let id_for_event = id.clone();
        let callback: WorkerEventCallback = Arc::new(move |event| {
            apply_worker_event(&state_for_event, &id_for_event, event);
            emit_task_progress(&app_for_event, &state_for_event, &id_for_event);
        });
        let result =
            ai_worker::run_worker_with_callback(&spec, &request, Some(cancel), Some(callback))
                .await;

        match result {
            Ok(run) => {
                let output = Path::new(&run.result.output_path);
                let valid = output.is_file()
                    && std::fs::metadata(output)
                        .map(|metadata| metadata.len() > 0)
                        .unwrap_or(false);
                if valid {
                    let _ = state_for_task.update(&id, |task| {
                        task.status = "completed".into();
                        task.phase = "completed".into();
                        task.percent = 100.0;
                        task.message = Some("AI 输出已生成".into());
                        task.output_sample_rate = Some(run.result.sample_rate);
                        task.output_channels = Some(run.result.channels);
                        task.output_duration_seconds = Some(run.result.duration_seconds);
                    });
                } else {
                    let _ = state_for_task.update(&id, |task| {
                        task.status = "failed".into();
                        task.phase = "verifying".into();
                        task.error = Some("Worker 返回成功，但输出文件不存在或为空".into());
                    });
                }
            }
            Err(error) => {
                let status = if matches!(error, WorkerError::Cancelled) {
                    "cancelled"
                } else {
                    "failed"
                };
                let _ = state_for_task.update(&id, |task| {
                    task.status = status.into();
                    task.phase = "finished".into();
                    task.error = Some(error.to_string());
                });
            }
        }
        // 终态写入历史库（含成功 / 失败 / 取消）
        if let Some(task) = state_for_task.list().into_iter().find(|t| t.id == id) {
            record_history(&app_for_task, &task);
        }
        emit_task_progress(&app_for_task, &state_for_task, &id);
        state_for_task.remove_cancel(&id);
    });

    Ok(task)
}

#[tauri::command(rename_all = "snake_case")]
pub fn audio_quality_cancel(
    task_id: String,
    app: AppHandle,
    state: State<'_, AudioQualityState>,
) -> Result<(), String> {
    if state.cancel(&task_id) {
        let _ = state.update(&task_id, |task| {
            task.phase = "cancelling".into();
            task.message = Some("正在取消 Python AI Worker".into());
        });
        // 必须立即回推一次状态：否则前端要一直等到 Worker 真正退出才更新，
        // 期间进度条仍停在「上一次的进度」，看起来像点了取消没反应。
        emit_task_progress(&app, &state, &task_id);
        Ok(())
    } else {
        Err(format!("找不到可取消的音频处理任务: {task_id}"))
    }
}

#[tauri::command]
pub fn audio_quality_list_tasks(
    state: State<'_, AudioQualityState>,
) -> Result<Vec<AudioQualityTask>, String> {
    Ok(state.list())
}

/// 输出采样率约束（按后端区分，避免张冠李戴）：
/// - FlashSR 硬约束 48 kHz（模型约束见 FlashSR 实施计划 §4.4）；
/// - VoiceFixer 原生 **44.1 kHz**，强制 48 kHz 会把修复结果重采样（R2）。
fn validate_output_sample_rate(
    backend: Backend,
    output_sample_rate: Option<u32>,
) -> Result<(), String> {
    let expected = match backend {
        Backend::FlashSr => Some(FLASHSR_SAMPLE_RATE),
        Backend::VoiceFixer => Some(VOICEFIXER_SAMPLE_RATE),
        _ => None,
    };
    let Some(expected) = expected else {
        return Ok(());
    };
    if let Some(rate) = output_sample_rate {
        if rate != expected {
            return Err(match backend {
                Backend::FlashSr => format!("FlashSR 仅支持 {expected} Hz 输出采样率"),
                Backend::VoiceFixer => {
                    format!("语音修复（VoiceFixer）输出为 {expected} Hz，不支持其它采样率")
                }
                _ => format!("该后端仅支持 {expected} Hz 输出采样率"),
            });
        }
    }
    Ok(())
}

fn validate_request(request: &AiProcessRequest) -> Result<(), String> {
    if !(request.chunk_seconds.is_finite() && request.chunk_seconds > 0.0) {
        return Err("chunkSeconds 必须是正数".into());
    }
    if !(request.overlap_seconds.is_finite()
        && request.overlap_seconds >= 0.0
        && request.overlap_seconds < request.chunk_seconds)
    {
        return Err("overlapSeconds 必须大于等于 0 且小于 chunkSeconds".into());
    }
    if request.model_id.trim().is_empty() {
        return Err("modelId 不能为空".into());
    }
    Ok(())
}

fn allocate_output_path(input: &Path, requested: Option<&Path>) -> Result<PathBuf, String> {
    let candidate = requested
        .map(Path::to_path_buf)
        .unwrap_or_else(|| input.with_extension("ai.flac"));
    let parent = candidate
        .parent()
        .ok_or_else(|| "输出路径缺少父目录".to_string())?;
    let stem = candidate
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("enhanced");
    let extension = candidate
        .extension()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("flac");
    for index in 0..10_000usize {
        let name = if index == 0 {
            format!("{stem}.{extension}")
        } else {
            format!("{stem} ({index}).{extension}")
        };
        let path = parent.join(name);
        if !path.exists() {
            return Ok(path);
        }
    }
    Err(format!(
        "无法为输出文件分配不冲突的路径: {}",
        candidate.display()
    ))
}

fn apply_worker_event(state: &AudioQualityState, task_id: &str, event: &WorkerEvent) {
    let _ = state.update(task_id, |task| match event {
        WorkerEvent::Ready { worker_version, .. } => {
            task.phase = "ready".into();
            task.message = Some(format!("Worker 已就绪: {worker_version}"));
        }
        WorkerEvent::Progress {
            phase,
            percent,
            message,
            ..
        } => {
            task.status = "processing".into();
            task.phase = phase.clone();
            task.percent = percent.clamp(0.0, 100.0);
            task.message = message.clone();
        }
        WorkerEvent::Completed { .. } => {
            task.phase = "verifying".into();
            task.percent = 100.0;
        }
        WorkerEvent::Error { code, message, .. } => {
            task.phase = "error".into();
            task.error = Some(format!("[{code}] {message}"));
        }
    });
}

fn emit_task_progress(app: &AppHandle, state: &AudioQualityState, task_id: &str) {
    let Some(task) = state.list().into_iter().find(|task| task.id == task_id) else {
        return;
    };
    let _ = app.emit("audio-quality-progress", task);
}

/// 任务进入终态（completed / failed / cancelled）时写入通用历史库。
/// 失败仅打印日志，不影响任务本身的结果。
fn record_history(app: &AppHandle, task: &AudioQualityTask) {
    if !matches!(task.status.as_str(), "completed" | "failed" | "cancelled") {
        return;
    }
    let Ok(conn) = crate::history::open_db(&crate::commands::history_dir(app)) else {
        return;
    };
    let title = Path::new(&task.input_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| task.input_path.clone());
    let status_label = match task.status.as_str() {
        "completed" => "已完成",
        "failed" => "失败",
        _ => "已取消",
    };
    let subtitle = format!("{} · {}", task.model_id, status_label);
    let payload = serde_json::to_string(task).unwrap_or_default();
    // 仅成功时才存在有效输出文件，失败/取消留空以免「打开目录」指向不存在的文件
    let file_path = if task.status == "completed" {
        task.output_path.clone()
    } else {
        String::new()
    };
    if let Err(e) = crate::history::insert(
        &conn,
        crate::history::HistoryKind::Enhance,
        &title,
        &subtitle,
        &payload,
        &file_path,
    ) {
        eprintln!("[history] 写入音质提升历史失败: {e}");
    }
}

fn chrono_like_timestamp() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("audio-processor-quality-command-{suffix}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 回归：当前选中 FlashSR 时，运行时检查仍应把 AudioSR / DeepFilterNet
    /// 一并报告为可用（跨后端模型并集），而非把它们误判为「不可用」。
    #[tokio::test]
    async fn runtime_check_unions_models_across_backends() {
        if ai_worker::WorkerSpec::flashsr().is_none()
            || ai_worker::WorkerSpec::production().is_none()
        {
            eprintln!("skip: Python AI 运行时不可用");
            return;
        }
        let check = audio_quality_check_ai_runtime(Some("flashsr".into()))
            .await
            .unwrap();
        assert!(
            check.models.contains(&"flashsr".to_string()),
            "应包含 flashsr: {:?}",
            check.models
        );
        assert!(
            check.models.contains(&"audiosr-basic".to_string()),
            "选 FlashSR 时不应把 AudioSR 误判为不可用: {:?}",
            check.models
        );
        assert!(check.available);
        assert!(check.production_ready);
    }

    #[test]
    fn validate_request_rejects_invalid_overlap() {
        let request = AiProcessRequest {
            request_id: "test".into(),
            input_path: "input".into(),
            output_path: "output".into(),
            model_id: "model".into(),
            device: "cpu".into(),
            chunk_seconds: 2.0,
            overlap_seconds: 2.0,
            output_sample_rate: Some(48_000),
            mode: None,
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn flashsr_rejects_non_48k_output_sample_rate() {
        assert!(validate_output_sample_rate(Backend::FlashSr, Some(44_100)).is_err());
        assert!(validate_output_sample_rate(Backend::FlashSr, Some(48_000)).is_ok());
        // 未指定采样率时交给 Python 侧兜底为 48 kHz
        assert!(validate_output_sample_rate(Backend::FlashSr, None).is_ok());
        // 其它后端不受该限制
        assert!(validate_output_sample_rate(Backend::AudioSr, Some(44_100)).is_ok());
    }

    #[test]
    fn voicefixer_rejects_48k_output_sample_rate() {
        // R2：不得把 VoiceFixer 的 44.1 kHz 结果重采样成 48 kHz
        assert!(validate_output_sample_rate(Backend::VoiceFixer, Some(48_000)).is_err());
        assert!(validate_output_sample_rate(Backend::VoiceFixer, Some(44_100)).is_ok());
        assert!(validate_output_sample_rate(Backend::VoiceFixer, None).is_ok());
    }

    #[test]
    fn resolve_backend_routes_flashsr_via_manifest() {
        // 仓库 manifest.json 含 backend=flashsr 的条目
        assert_eq!(
            backend::resolve_backend("flashsr"),
            Backend::FlashSr
        );
        // 未命中时按 id 前缀回退
        assert_eq!(
            backend::resolve_backend("flashsr-custom"),
            Backend::FlashSr
        );
        assert_eq!(
            backend::resolve_backend("audiosr-basic"),
            Backend::AudioSr
        );
    }

    #[test]
    fn allocate_output_path_does_not_overwrite_existing_file() {
        let dir = temp_dir();
        let input = dir.join("source.m4a");
        let output = dir.join("enhanced.flac");
        std::fs::write(&input, b"input").unwrap();
        std::fs::write(&output, b"existing").unwrap();

        let allocated = allocate_output_path(&input, Some(&output)).unwrap();
        assert_eq!(allocated, dir.join("enhanced (1).flac"));
        assert_eq!(std::fs::read(&output).unwrap(), b"existing");
        let _ = std::fs::remove_dir_all(dir);
    }
}
