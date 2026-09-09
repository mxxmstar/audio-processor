//! 图像质量处理相关 Tauri 命令（Real-ESRGAN 超分）。
//!
//! 镜像 `commands::audio_quality`：运行时检查、模型列举、启动/取消任务、历史记录。
//! 差异仅在任务载荷——音频用采样率/声道/时长，图像用宽/高/倍数/格式。

use crate::commands::history_dir;
use crate::history::{self, HistoryKind};
use crate::image_quality::ai_worker::{
    self, AiProcessRequest, WorkerError, WorkerEvent, WorkerEventCallback, WorkerSpec,
    PROTOCOL_VERSION,
};
use crate::image_quality::backend::{self, ModelInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageQualityTask {
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
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub scale: Option<u32>,
    pub format: Option<String>,
}

#[derive(Default, Clone)]
pub struct ImageQualityState {
    inner: Arc<Mutex<ImageQualityStore>>,
}

#[derive(Default)]
struct ImageQualityStore {
    tasks: HashMap<String, ImageQualityTask>,
    cancel_flags: HashMap<String, Arc<AtomicBool>>,
}

impl ImageQualityState {
    fn insert(&self, task: ImageQualityTask, cancel: Arc<AtomicBool>) {
        let mut store = self.inner.lock().expect("image quality state poisoned");
        store.cancel_flags.insert(task.id.clone(), cancel);
        store.tasks.insert(task.id.clone(), task);
    }

    fn update<F>(&self, id: &str, update: F) -> Option<ImageQualityTask>
    where
        F: FnOnce(&mut ImageQualityTask),
    {
        let mut store = self.inner.lock().expect("image quality state poisoned");
        let task = store.tasks.get_mut(id)?;
        update(task);
        Some(task.clone())
    }

    fn cancel(&self, id: &str) -> bool {
        let store = self.inner.lock().expect("image quality state poisoned");
        let Some(flag) = store.cancel_flags.get(id) else {
            return false;
        };
        flag.store(true, Ordering::Relaxed);
        true
    }

    fn list(&self) -> Vec<ImageQualityTask> {
        let store = self.inner.lock().expect("image quality state poisoned");
        let mut tasks: Vec<_> = store.tasks.values().cloned().collect();
        tasks.sort_by(|a, b| a.id.cmp(&b.id));
        tasks
    }

    fn remove_cancel(&self, id: &str) {
        self.inner
            .lock()
            .expect("image quality state poisoned")
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
pub struct ImageQualityModelInstallResult {
    pub model_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnhanceImageInput {
    pub input_path: String,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub jpg_quality: Option<u32>,
    #[serde(default)]
    pub outscale: Option<f64>,
    #[serde(default)]
    pub tile: Option<u32>,
}

/// 检查 Real-ESRGAN Python Worker 是否能启动并完成 JSONL ready 握手。
///
/// `AUDIO_AI_REALESRGAN_WORKER` 可指定打包后的 Worker 可执行；未配置且
/// `AUDIO_AI_USE_FAKE=1` 时使用 fake Worker（仅协议验证，`productionReady=false`）。
#[tauri::command]
pub async fn enhance_image_check_ai_runtime(
    model_id: Option<String>,
) -> Result<AiRuntimeCheck, String> {
    let model_id = model_id.unwrap_or_else(|| backend::REALESRGAN_MODEL_ID.into());

    let (spec, worker_kind) = match backend::select_worker_spec(&model_id) {
        Ok(spec) => spec,
        Err(_) => return Ok(unavailable_runtime("找不到 Real-ESRGAN Worker 或 Python 运行时")),
    };
    let ready = match ai_worker::probe_worker(&spec).await {
        Ok(ready) => ready,
        Err(_) => {
            return Ok(AiRuntimeCheck {
                available: false,
                production_ready: false,
                protocol_version: PROTOCOL_VERSION,
                worker_kind,
                program: Some(spec.program.to_string_lossy().into_owned()),
                worker_version: None,
                models: Vec::new(),
                model_errors: Vec::new(),
                error: Some("Real-ESRGAN 运行时探测失败".into()),
            });
        }
    };
    let program = spec.program.to_string_lossy().into_owned();
    Ok(AiRuntimeCheck {
        available: true,
        production_ready: !ready.models.is_empty(),
        protocol_version: PROTOCOL_VERSION,
        worker_kind,
        program: Some(program),
        worker_version: Some(ready.worker_version),
        models: ready.models,
        model_errors: ready.model_errors,
        error: None,
    })
}

/// 列出 manifest.json 中全部可用模型，供前端渲染可选模型。
#[tauri::command]
pub fn enhance_image_list_models() -> Result<Vec<ModelInfo>, String> {
    Ok(backend::list_models())
}

/// 显式下载并校验模型（镜像音频侧）。Real-ESRGAN 权重已在阶段 0 落地，
/// 此命令主要用于环境初始化或权重缺失时的补齐。
#[tauri::command(rename_all = "snake_case")]
pub async fn enhance_image_download_model(
    model_id: String,
    workers: Option<u32>,
) -> Result<ImageQualityModelInstallResult, String> {
    let model_id = model_id.trim().to_string();
    if model_id.is_empty() {
        return Err("modelId 不能为空".into());
    }
    let workers = workers.unwrap_or(8).clamp(1, 32);
    let Some(spec) = WorkerSpec::model_manager() else {
        return Err("找不到 Python AI 模型安装器或 Python 运行时".into());
    };
    let progress: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(|line| {
        eprintln!("[image-ai-model] {line}");
    });
    ai_worker::install_model(&spec, &model_id, workers, Some(progress))
        .await
        .map_err(|error| format!("模型安装失败: {error}"))?;
    Ok(ImageQualityModelInstallResult {
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

/// 启动一个后台图像超分任务。真实 Worker 通过 `AUDIO_AI_REALESRGAN_WORKER`
/// 配置；fake Worker 只能通过 `AUDIO_AI_USE_FAKE=1` 显式启用。
#[tauri::command]
pub fn enhance_image(
    input: EnhanceImageInput,
    app: AppHandle,
    state: State<'_, ImageQualityState>,
) -> Result<ImageQualityTask, String> {
    let input_path = PathBuf::from(&input.input_path);
    if !input_path.is_file() {
        return Err(format!("输入图片不存在: {}", input_path.display()));
    }

    let output_path =
        allocate_output_path(&input_path, input.output_path.as_deref().map(Path::new))?;
    if output_path == input_path {
        return Err("输出路径不能覆盖输入图片".into());
    }
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("创建输出目录失败: {}: {e}", parent.display()))?;
    }

    let model_id = input
        .model_id
        .unwrap_or_else(|| backend::REALESRGAN_MODEL_ID.into());
    let (spec, worker_kind) = backend::select_worker_spec(&model_id)?;
    let id = format!(
        "iq-{}-{}",
        chrono_like_timestamp(),
        NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed)
    );
    let request = AiProcessRequest {
        request_id: id.clone(),
        input_path: input_path.to_string_lossy().into_owned(),
        output_path: output_path.to_string_lossy().into_owned(),
        model_id,
        device: input.device.unwrap_or_else(|| "cpu".into()),
        outscale: input.outscale,
        jpg_quality: input.jpg_quality,
        tile: input.tile,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let task = ImageQualityTask {
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
        width: None,
        height: None,
        scale: None,
        format: None,
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
                        task.width = Some(run.result.width);
                        task.height = Some(run.result.height);
                        task.scale = Some(run.result.scale);
                        task.format = Some(run.result.format);
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
        // 终态写入历史库（成功 / 失败 / 取消）
        if let Some(task) = state_for_task.list().into_iter().find(|t| t.id == id) {
            record_history(&app_for_task, &task);
        }
        emit_task_progress(&app_for_task, &state_for_task, &id);
        state_for_task.remove_cancel(&id);
    });

    Ok(task)
}

#[tauri::command(rename_all = "snake_case")]
pub fn enhance_image_cancel(
    task_id: String,
    app: AppHandle,
    state: State<'_, ImageQualityState>,
) -> Result<(), String> {
    if state.cancel(&task_id) {
        let _ = state.update(&task_id, |task| {
            task.phase = "cancelling".into();
            task.message = Some("正在取消 Python AI Worker".into());
        });
        emit_task_progress(&app, &state, &task_id);
        Ok(())
    } else {
        Err(format!("找不到可取消的图像处理任务: {task_id}"))
    }
}

#[tauri::command]
pub fn enhance_image_list_tasks(
    state: State<'_, ImageQualityState>,
) -> Result<Vec<ImageQualityTask>, String> {
    Ok(state.list())
}

fn allocate_output_path(input: &Path, requested: Option<&Path>) -> Result<PathBuf, String> {
    let candidate = requested
        .map(Path::to_path_buf)
        .unwrap_or_else(|| input.with_extension("png"));
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
        .unwrap_or("png");
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

fn apply_worker_event(state: &ImageQualityState, task_id: &str, event: &WorkerEvent) {
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
        WorkerEvent::Completed {
            width,
            height,
            scale,
            format,
            ..
        } => {
            task.phase = "verifying".into();
            task.percent = 100.0;
            task.width = Some(*width);
            task.height = Some(*height);
            task.scale = Some(*scale);
            task.format = Some(format.clone());
        }
        WorkerEvent::Error { code, message, .. } => {
            task.phase = "error".into();
            task.error = Some(format!("[{code}] {message}"));
        }
    });
}

fn emit_task_progress(app: &AppHandle, state: &ImageQualityState, task_id: &str) {
    let Some(task) = state.list().into_iter().find(|task| task.id == task_id) else {
        return;
    };
    let _ = app.emit("image-quality-progress", task);
}

/// 任务进入终态（completed / failed / cancelled）时写入通用历史库。
fn record_history(app: &AppHandle, task: &ImageQualityTask) {
    if !matches!(task.status.as_str(), "completed" | "failed" | "cancelled") {
        return;
    }
    let Ok(conn) = history::open_db(&history_dir(app)) else {
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
    let file_path = if task.status == "completed" {
        task.output_path.clone()
    } else {
        String::new()
    };
    if let Err(e) = history::insert(
        &conn,
        HistoryKind::ImageEnhance,
        &title,
        &subtitle,
        &payload,
        &file_path,
    ) {
        eprintln!("[history] 写入图像增强历史失败: {e}");
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

    #[test]
    fn resolve_backend_routes_realesrgan() {
        assert_eq!(
            backend::resolve_backend("realesrgan-x4plus"),
            backend::Backend::RealEsrGan
        );
        assert_eq!(
            backend::resolve_backend("realesrgan"),
            backend::Backend::RealEsrGan
        );
    }

    #[test]
    fn allocate_output_path_does_not_overwrite_existing_file() {
        let dir = std::env::temp_dir().join("image-quality-cmd-test");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("source.png");
        let output = dir.join("enhanced.png");
        std::fs::write(&input, b"input").unwrap();
        std::fs::write(&output, b"existing").unwrap();

        let allocated = allocate_output_path(&input, Some(&output)).unwrap();
        assert_eq!(allocated, dir.join("enhanced (1).png"));
        assert_eq!(std::fs::read(&output).unwrap(), b"existing");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
