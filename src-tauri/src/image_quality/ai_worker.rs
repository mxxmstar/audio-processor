//! Rust/Python AI Worker 的 JSONL 子进程协议和生命周期管理（图像超分版）。
//!
//! 与 `audio_quality::ai_worker` 同构：进程协议、stdout/stderr 双向读取、取消、
//! ready 握手、错误分类、进度条噪声过滤完全复用；**仅 result 载荷不同**——
//! 音频用 `sample_rate/channels/duration_seconds/peak_db`，图像用
//! `width/height/scale/format`。因此这里是独立模块的镜像实现，而非共享泛型。

use serde::{Deserialize, Serialize};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

pub const PROTOCOL_VERSION: u32 = 1;
const MAX_PROTOCOL_LINE_BYTES: usize = 1024 * 1024;
const MAX_EVENTS: usize = 4096;
const MAX_ERROR_STDERR_BYTES: usize = 8 * 1024;
/// 收到 result / ready 后给子进程自行退出的宽限期。
///
/// 跑完 Real-ESRGAN 这类 ~64 MB 权重的推理后，Python 解释器关闭要释放张量、
/// join 线程，实测需要数秒。旧值（2 秒）会让 `finish_child` 强杀子进程
/// （Windows 下 `TerminateProcess` 退出码 1），把已经成功并写好输出的任务误判
/// 为失败（音频侧阶段 6 实测，图像侧同样适用）。
const CHILD_EXIT_TIMEOUT: Duration = Duration::from_secs(15);
const READY_TIMEOUT: Duration = Duration::from_secs(3);

/// 一个 AI 处理请求。路径由 Rust 分配，Python 只执行结构化请求中的路径。
#[derive(Debug, Clone, Serialize)]
pub struct AiProcessRequest {
    pub request_id: String,
    pub input_path: String,
    pub output_path: String,
    pub model_id: String,
    pub device: String,
    /// 输出倍数覆盖；缺省时使用模型的固定倍数（RealESRGAN_x4plus 为 4）。
    pub outscale: Option<f64>,
    /// jpg 输出质量（仅 format=jpg 时生效）；缺省为 95。
    pub jpg_quality: Option<u32>,
    /// 分块边长（像素）；缺省由 pipeline 自适应（小图整图直推、大图 512）。
    pub tile: Option<u32>,
}

impl AiProcessRequest {
    pub fn new(request_id: impl Into<String>, input: &Path, output: &Path) -> Self {
        Self {
            request_id: request_id.into(),
            input_path: input.to_string_lossy().into_owned(),
            output_path: output.to_string_lossy().into_owned(),
            model_id: "realesrgan-x4plus".into(),
            device: "cpu".into(),
            outscale: None,
            jpg_quality: None,
            tile: None,
        }
    }

    fn process_message(&self) -> ProcessMessage<'_> {
        ProcessMessage {
            protocol_version: PROTOCOL_VERSION,
            request_id: &self.request_id,
            command: "process",
            input_path: &self.input_path,
            output_path: &self.output_path,
            model_id: &self.model_id,
            device: &self.device,
            outscale: self.outscale,
            jpg_quality: self.jpg_quality,
            tile: self.tile,
        }
    }
}

#[derive(Debug, Serialize)]
struct ProcessMessage<'a> {
    protocol_version: u32,
    request_id: &'a str,
    command: &'static str,
    input_path: &'a str,
    output_path: &'a str,
    model_id: &'a str,
    device: &'a str,
    outscale: Option<f64>,
    jpg_quality: Option<u32>,
    tile: Option<u32>,
}

#[derive(Debug, Serialize)]
struct CancelMessage<'a> {
    protocol_version: u32,
    request_id: &'a str,
    command: &'static str,
}

#[derive(Debug, Serialize)]
struct ShutdownMessage<'a> {
    protocol_version: u32,
    request_id: &'a str,
    command: &'static str,
}

/// Python Worker 发出的事件。
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerEvent {
    Ready {
        protocol_version: u32,
        request_id: String,
        worker_version: String,
        models: Vec<String>,
        #[serde(default)]
        model_errors: Vec<String>,
    },
    Progress {
        protocol_version: u32,
        request_id: String,
        phase: String,
        percent: f64,
        #[serde(default)]
        processed_tiles: Option<u32>,
        #[serde(default)]
        total_tiles: Option<u32>,
        #[serde(default)]
        message: Option<String>,
    },
    #[serde(rename = "result")]
    Completed {
        protocol_version: u32,
        request_id: String,
        status: String,
        output_path: String,
        model_id: String,
        model_version: String,
        width: u32,
        height: u32,
        scale: u32,
        format: String,
    },
    #[serde(rename = "error")]
    Error {
        protocol_version: u32,
        request_id: String,
        code: String,
        message: String,
        #[serde(default)]
        retryable: bool,
    },
}

impl WorkerEvent {
    fn protocol_version(&self) -> u32 {
        match self {
            Self::Ready {
                protocol_version, ..
            }
            | Self::Progress {
                protocol_version, ..
            }
            | Self::Completed {
                protocol_version, ..
            }
            | Self::Error {
                protocol_version, ..
            } => *protocol_version,
        }
    }

    fn request_id(&self) -> &str {
        match self {
            Self::Ready { request_id, .. }
            | Self::Progress { request_id, .. }
            | Self::Completed { request_id, .. }
            | Self::Error { request_id, .. } => request_id,
        }
    }
}

/// Worker 的启动方式。生产环境可以替换为打包后的 `image-ai-worker.exe`。
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
}

impl WorkerSpec {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
        }
    }

    pub fn arg(mut self, arg: impl AsRef<OsStr>) -> Self {
        self.args.push(arg.as_ref().to_os_string());
        self
    }

    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Real-ESRGAN 处理 Worker（运行在专用 `.venv-realesrgan` 下）。
    pub fn realesrgan() -> Option<Self> {
        if let Some(path) = std::env::var_os("AUDIO_AI_REALESRGAN_WORKER") {
            return Some(Self::new(path));
        }
        let python = find_python_realesrgan()?;
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python")
            .join("image_ai_realesrgan")
            .join("worker.py");
        if !script.is_file() {
            return None;
        }
        Some(Self::new(python).arg("-u").arg(script))
    }

    /// Real-ESRGAN 协议自检用的 fake Worker（与音频的 `fake_worker` 同构）。
    pub fn realesrgan_fake(mode: &str) -> Option<Self> {
        let python = find_python_realesrgan()?;
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python")
            .join("image_ai_realesrgan")
            .join("fake_worker.py");
        if !script.is_file() {
            return None;
        }
        Some(
            Self::new(python)
                .arg("-u")
                .arg(script)
                .arg("--mode")
                .arg(mode),
        )
    }

    /// 显式模型安装器。与音频共用 Python 运行时，但安装器只负责下载/校验模型。
    pub fn model_manager() -> Option<Self> {
        if let Some(path) = std::env::var_os("AUDIO_AI_MODEL_MANAGER") {
            return Some(Self::new(path));
        }
        let python = find_python_realesrgan()?;
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python")
            .join("audio_ai_flashsr")
            .join("model_manager.py");
        if !script.is_file() {
            return None;
        }
        Some(Self::new(python).arg("-u").arg(script))
    }
}

/// 已完成的 Worker 运行结果和有限事件记录。
#[derive(Debug)]
pub struct WorkerRunOutput {
    pub result: WorkerResult,
    pub events: Vec<WorkerEvent>,
    pub stderr: String,
}

pub type WorkerEventCallback = Arc<dyn Fn(&WorkerEvent) + Send + Sync>;

#[derive(Debug, Clone)]
pub struct WorkerReady {
    pub worker_version: String,
    pub models: Vec<String>,
    pub model_errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct WorkerResult {
    pub output_path: String,
    pub model_id: String,
    pub model_version: String,
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub format: String,
}

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("启动 AI Worker 失败: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("写入 AI Worker stdin 失败: {0}")]
    Stdin(#[source] std::io::Error),
    #[error("读取 AI Worker stdout 失败: {0}")]
    Stdout(#[source] std::io::Error),
    #[error("读取 AI Worker stderr 失败: {0}")]
    Stderr(#[source] std::io::Error),
    #[error("AI Worker 协议错误: {0}")]
    Protocol(String),
    #[error("AI Worker 返回错误 [{code}]: {message}")]
    Remote {
        code: String,
        message: String,
        retryable: bool,
    },
    #[error("AI Worker 已取消")]
    Cancelled,
    #[error("AI Worker 未发送 ready 事件")]
    NotReady,
    #[error("等待 AI Worker ready 超时")]
    ReadyTimeout,
    #[error("AI Worker 未发送 result 事件")]
    NoResult,
    #[error("AI Worker 输出路径与 Rust 分配路径不一致")]
    OutputPathMismatch,
    #[error("AI Worker 进程异常退出，退出码: {code:?}{stderr}")]
    Exited {
        code: Option<i32>,
        stderr: String,
    },
}

/// 启动一个 Worker，发送一次 process 请求并读取到 result/error。
///
/// `cancel` 由上层任务控制；置位后 Rust 会先发送 cancel 协议消息，超时
/// 未退出再终止子进程。stdout/stderr 同时读取，避免管道互相阻塞。
pub async fn run_worker(
    spec: &WorkerSpec,
    request: &AiProcessRequest,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<WorkerRunOutput, WorkerError> {
    run_worker_with_callback(spec, request, cancel, None).await
}

/// 与 [`run_worker`] 相同，但会在读取每条合法事件后同步调用回调。
pub async fn run_worker_with_callback(
    spec: &WorkerSpec,
    request: &AiProcessRequest,
    cancel: Option<Arc<AtomicBool>>,
    on_event: Option<WorkerEventCallback>,
) -> Result<WorkerRunOutput, WorkerError> {
    let mut command = Command::new(&spec.program);
    command
        .args(&spec.args)
        .envs(spec.env.iter().map(|(key, value)| (key, value)))
        .env("PYTHONUNBUFFERED", "1")
        // 强制 Python stdio 使用 UTF-8：Windows 默认按区域编码（cp936/GBK）
        // 解码 stdin，会使含非 ASCII 字符（如日文）的路径被破坏。
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    hide_console_window(&mut command);

    let mut child = command.spawn().map_err(WorkerError::Spawn)?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| WorkerError::Protocol("AI Worker stdin 未按协议打开".into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| WorkerError::Protocol("AI Worker stdout 未按协议打开".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| WorkerError::Protocol("AI Worker stderr 未按协议打开".into()))?;

    write_json_line(&mut stdin, &request.process_message()).await?;
    let stderr_task = tokio::spawn(read_stderr(stderr));
    let mut lines = BufReader::new(stdout).lines();
    let cancel_flag = cancel.clone();
    let cancel_wait = wait_for_cancel(cancel_flag);
    tokio::pin!(cancel_wait);
    let mut cancel_sent = false;
    let mut events = Vec::new();
    let mut ready = false;
    let mut completed: Option<WorkerResult> = None;
    let mut terminal_error: Option<WorkerError> = None;

    loop {
        tokio::select! {
            line = lines.next_line() => {
                let line = line.map_err(WorkerError::Stdout)?;
                let Some(line) = line else { break };
                if line.len() > MAX_PROTOCOL_LINE_BYTES {
                    terminal_error = Some(WorkerError::Protocol("JSONL 消息超过 1 MiB 限制".into()));
                    break;
                }
                let event: WorkerEvent = serde_json::from_str(&line)
                    .map_err(|e| WorkerError::Protocol(format!("解析 JSONL 失败: {e}")))?;
                validate_event(&event, request)?;
                if events.len() < MAX_EVENTS {
                    events.push(event.clone());
                }
                if let Some(callback) = &on_event {
                    callback(&event);
                }
                match event {
                    WorkerEvent::Ready { .. } => ready = true,
                    WorkerEvent::Progress { .. } => {
                        if !ready {
                            terminal_error = Some(WorkerError::NotReady);
                            break;
                        }
                    }
                    WorkerEvent::Completed {
                        output_path,
                        model_id,
                        model_version,
                        width,
                        height,
                        scale,
                        format,
                        ..
                    } => {
                        if !ready {
                            terminal_error = Some(WorkerError::NotReady);
                            break;
                        }
                        if output_path != request.output_path {
                            terminal_error = Some(WorkerError::OutputPathMismatch);
                            break;
                        }
                        completed = Some(WorkerResult {
                            output_path,
                            model_id,
                            model_version,
                            width,
                            height,
                            scale,
                            format,
                        });
                        break;
                    }
                    WorkerEvent::Error { code, message, retryable, .. } => {
                        terminal_error = Some(if code == "CANCELLED" || cancel_sent {
                            WorkerError::Cancelled
                        } else {
                            WorkerError::Remote { code, message, retryable }
                        });
                        break;
                    }
                }
            }
            _ = &mut cancel_wait, if !cancel_sent => {
                write_json_line(
                    &mut stdin,
                    &CancelMessage {
                        protocol_version: PROTOCOL_VERSION,
                        request_id: &request.request_id,
                        command: "cancel",
                    },
                ).await?;
                cancel_sent = true;
            }
        }
    }

    drop(stdin);
    if terminal_error.is_some() || completed.is_some() || cancel_sent {
        finish_child(&mut child, CHILD_EXIT_TIMEOUT).await;
    }
    let status = child.wait().await.map_err(WorkerError::Stdout)?;
    let stderr = match stderr_task.await {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => return Err(WorkerError::Stderr(error)),
        Err(error) => {
            return Err(WorkerError::Protocol(format!(
                "读取 AI Worker stderr 任务失败: {error}"
            )))
        }
    };

    if let Some(error) = terminal_error {
        return Err(error);
    }
    let Some(result) = completed else {
        if cancel_sent {
            return Err(WorkerError::Cancelled);
        }
        if !status.success() {
            return Err(worker_exit_error(status.code(), &stderr));
        }
        return Err(if ready {
            WorkerError::NoResult
        } else {
            WorkerError::NotReady
        });
    };
    // 已拿到 result 时任务实际已完成，且输出文件已由 Python 侧校验存在。
    // 此时的退出码只反映"收尾 / 解释器关闭"阶段（例如被宽限期强杀），
    // 不应据此丢弃一个已经成功的任务。
    if !status.success() {
        eprintln!(
            "AI Worker 已完成但退出码非 0（{}），仅记录不判失败",
            status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".into())
        );
    }

    Ok(WorkerRunOutput {
        result,
        events,
        stderr,
    })
}

/// 只启动 Worker 并完成 ready 握手，用于运行时检查，不执行图像超分。
pub async fn probe_worker(spec: &WorkerSpec) -> Result<WorkerReady, WorkerError> {
    let mut command = Command::new(&spec.program);
    command
        .args(&spec.args)
        .envs(spec.env.iter().map(|(key, value)| (key, value)))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    hide_console_window(&mut command);

    let mut child = command.spawn().map_err(WorkerError::Spawn)?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| WorkerError::Protocol("AI Worker stdin 未按协议打开".into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| WorkerError::Protocol("AI Worker stdout 未按协议打开".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| WorkerError::Protocol("AI Worker stderr 未按协议打开".into()))?;
    let stderr_task = tokio::spawn(read_stderr(stderr));
    let mut lines = BufReader::new(stdout).lines();
    let line = timeout(READY_TIMEOUT, lines.next_line())
        .await
        .map_err(|_| WorkerError::ReadyTimeout)?
        .map_err(WorkerError::Stdout)?
        .ok_or(WorkerError::NotReady)?;
    if line.len() > MAX_PROTOCOL_LINE_BYTES {
        return Err(WorkerError::Protocol("ready 消息超过 1 MiB 限制".into()));
    }
    let event: WorkerEvent = serde_json::from_str(&line)
        .map_err(|e| WorkerError::Protocol(format!("解析 ready JSONL 失败: {e}")))?;
    if event.protocol_version() != PROTOCOL_VERSION {
        return Err(WorkerError::Protocol(format!(
            "协议版本不匹配: {} != {}",
            event.protocol_version(),
            PROTOCOL_VERSION
        )));
    }
    let WorkerEvent::Ready {
        worker_version,
        models,
        model_errors,
        ..
    } = event
    else {
        return Err(WorkerError::NotReady);
    };

    write_json_line(
        &mut stdin,
        &ShutdownMessage {
            protocol_version: PROTOCOL_VERSION,
            request_id: "runtime-check",
            command: "shutdown",
        },
    )
    .await?;
    drop(stdin);
    finish_child(&mut child, CHILD_EXIT_TIMEOUT).await;
    let status = child.wait().await.map_err(WorkerError::Stdout)?;
    let stderr = match stderr_task.await {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => return Err(WorkerError::Stderr(error)),
        Err(error) => {
            return Err(WorkerError::Protocol(format!(
                "读取 AI Worker stderr 任务失败: {error}"
            )))
        }
    };
    if !status.success() {
        return Err(worker_exit_error(status.code(), &stderr));
    }
    Ok(WorkerReady {
        worker_version,
        models,
        model_errors,
    })
}

/// 运行一次显式模型安装命令，并把安装器 stderr 中的进度行交给调用方。
pub async fn install_model(
    spec: &WorkerSpec,
    model_id: &str,
    workers: u32,
    on_progress: Option<Arc<dyn Fn(&str) + Send + Sync>>,
) -> Result<(), WorkerError> {
    let mut command = Command::new(&spec.program);
    command
        .args(&spec.args)
        .arg("--model-id")
        .arg(model_id)
        .arg("--workers")
        .arg(workers.to_string())
        .envs(spec.env.iter().map(|(key, value)| (key, value)))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    hide_console_window(&mut command);

    let mut child = command.spawn().map_err(WorkerError::Spawn)?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| WorkerError::Protocol("模型安装器 stderr 未按协议打开".into()))?;
    let mut lines = BufReader::new(stderr).lines();
    let mut stderr_text = String::new();
    while let Some(line) = lines.next_line().await.map_err(WorkerError::Stderr)? {
        if !stderr_text.is_empty() {
            stderr_text.push('\n');
        }
        stderr_text.push_str(&line);
        if let Some(callback) = &on_progress {
            callback(&line);
        }
    }
    let status = child.wait().await.map_err(WorkerError::Spawn)?;
    if !status.success() {
        return Err(worker_exit_error(status.code(), &stderr_text));
    }
    Ok(())
}

fn validate_event(event: &WorkerEvent, request: &AiProcessRequest) -> Result<(), WorkerError> {
    if event.protocol_version() != PROTOCOL_VERSION {
        return Err(WorkerError::Protocol(format!(
            "协议版本不匹配: {} != {}",
            event.protocol_version(),
            PROTOCOL_VERSION
        )));
    }
    // ready 事件可以在 process request_id 之前发送空 request_id。
    if !event.request_id().is_empty() && event.request_id() != request.request_id {
        return Err(WorkerError::Protocol(format!(
            "request_id 不匹配: {} != {}",
            event.request_id(),
            request.request_id
        )));
    }
    Ok(())
}

async fn write_json_line<W: AsyncWriteExt + Unpin, T: Serialize>(
    writer: &mut W,
    value: &T,
) -> Result<(), WorkerError> {
    let payload = serde_json::to_vec(value)
        .map_err(|e| WorkerError::Protocol(format!("编码 JSONL 失败: {e}")))?;
    writer
        .write_all(&payload)
        .await
        .map_err(WorkerError::Stdin)?;
    writer.write_all(b"\n").await.map_err(WorkerError::Stdin)?;
    writer.flush().await.map_err(WorkerError::Stdin)
}

async fn wait_for_cancel(cancel: Option<Arc<AtomicBool>>) {
    loop {
        if cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
}

async fn read_stderr<R: AsyncRead + Unpin>(mut reader: R) -> Result<String, std::io::Error> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await?;
    const MAX_STDERR_BYTES: usize = 256 * 1024;
    if bytes.len() > MAX_STDERR_BYTES {
        bytes.drain(..bytes.len() - MAX_STDERR_BYTES);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn worker_exit_error(code: Option<i32>, stderr: &str) -> WorkerError {
    let stderr = stderr.trim();
    let detail = if stderr.is_empty() {
        String::new()
    } else {
        let cleaned = strip_progress_noise(stderr);
        if cleaned.is_empty() {
            "，stderr: (stderr 仅含进度条输出，已过滤；未捕获到异常信息)".to_string()
        } else {
            format!("，stderr: {}", stderr_tail(&cleaned, MAX_ERROR_STDERR_BYTES))
        }
    };
    WorkerError::Exited { code, stderr: detail }
}

/// 剔除 tqdm 等进度条片段。
fn strip_progress_noise(stderr: &str) -> String {
    stderr
        .split(['\n', '\r'])
        .filter(|part| !is_progress_noise(part))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 判断一个片段是否为进度条输出（tqdm / 上游 enhance 的 Tile 进度）。
fn is_progress_noise(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.contains("DDIM Sampler") || trimmed.contains("Running DDIM Sampling") {
        return true;
    }
    if trimmed.contains("it/s") || trimmed.contains("s/it") || trimmed.contains("%|") {
        return true;
    }
    // Real-ESRGAN 上游 enhance 的 `Tile X/Y` 进度打印
    if trimmed.contains("Tile ") && trimmed.contains("/") {
        return true;
    }
    false
}

fn stderr_tail(stderr: &str, max_bytes: usize) -> String {
    if stderr.len() <= max_bytes {
        return stderr.to_string();
    }
    let start = stderr
        .char_indices()
        .find(|(index, _)| *index >= stderr.len() - max_bytes)
        .map(|(index, _)| index)
        .unwrap_or(0);
    format!("...(stderr 已截断，显示末尾)...\n{}", &stderr[start..])
}

async fn finish_child(child: &mut Child, wait_for: Duration) {
    if timeout(wait_for, child.wait()).await.is_err() {
        let _ = child.kill().await;
    }
}

fn hide_console_window(command: &mut Command) {
    #[cfg(windows)]
    {
        command.creation_flags(0x0800_0000);
    }
}

/// 查找 Real-ESRGAN 专用 Python 运行时（`.venv-realesrgan`）。
///
/// 优先级：`AUDIO_AI_REALESRGAN_PYTHON` → `AUDIO_AI_PYTHON` →
/// `.venv-realesrgan/Scripts/python.exe` → `.venv/...`（回退）→ `python` / `python3`。
fn find_python_realesrgan() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("AUDIO_AI_REALESRGAN_PYTHON") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("AUDIO_AI_PYTHON") {
        candidates.push(PathBuf::from(path));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    candidates.push(root.join(".venv-realesrgan").join("Scripts").join("python.exe"));
    candidates.push(root.join(".venv").join("Scripts").join("python.exe"));
    candidates.push(PathBuf::from("python"));
    candidates.push(PathBuf::from("python3"));
    candidates.into_iter().find(|candidate| {
        std::process::Command::new(candidate)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    })
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
        let dir = std::env::temp_dir().join(format!("image-processor-ai-{suffix}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn realesrgan_fake_worker_ready_probe() {
        let Some(spec) = WorkerSpec::realesrgan_fake("success") else {
            eprintln!("skip: Real-ESRGAN python runtime unavailable");
            return;
        };
        let ready = probe_worker(&spec).await.unwrap();
        assert_eq!(ready.worker_version, "fake-realesrgan-0.1.0");
        assert_eq!(ready.models, vec!["realesrgan-x4plus"]);
    }

    #[tokio::test]
    async fn realesrgan_fake_worker_success_round_trip() {
        let Some(spec) = WorkerSpec::realesrgan_fake("success") else {
            eprintln!("skip: Real-ESRGAN python runtime unavailable");
            return;
        };
        let dir = temp_dir();
        let output = dir.join("output.png");
        let request = AiProcessRequest::new("test-success", Path::new("input.png"), &output);
        let run = run_worker(&spec, &request, None).await.unwrap();

        assert_eq!(run.result.output_path, output.to_string_lossy());
        assert!(run
            .events
            .iter()
            .any(|event| matches!(event, WorkerEvent::Ready { .. })));
        assert!(run
            .events
            .iter()
            .any(|event| matches!(event, WorkerEvent::Progress { .. })));
        assert_eq!(run.result.width, 64);
        assert_eq!(run.result.height, 48);
        assert_eq!(run.result.scale, 4);
        assert_eq!(run.result.format, "png");
        assert!(output.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn realesrgan_fake_worker_error_is_returned() {
        let Some(spec) = WorkerSpec::realesrgan_fake("error") else {
            eprintln!("skip: Real-ESRGAN python runtime unavailable");
            return;
        };
        let dir = temp_dir();
        let request =
            AiProcessRequest::new("test-error", Path::new("input.png"), &dir.join("out.png"));
        let error = run_worker(&spec, &request, None).await.unwrap_err();
        assert!(
            matches!(error, WorkerError::Remote { ref code, .. } if code == "INFERENCE_FAILED")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn realesrgan_fake_worker_can_be_cancelled() {
        let Some(spec) = WorkerSpec::realesrgan_fake("slow") else {
            eprintln!("skip: Real-ESRGAN python runtime unavailable");
            return;
        };
        let dir = temp_dir();
        let request =
            AiProcessRequest::new("test-cancel", Path::new("input.png"), &dir.join("out.png"));
        let cancel = Arc::new(AtomicBool::new(false));
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(120)).await;
            trigger.store(true, Ordering::Relaxed);
        });

        let error = run_worker(&spec, &request, Some(cancel)).await.unwrap_err();
        assert!(matches!(error, WorkerError::Cancelled));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn realesrgan_fake_worker_exit_includes_stderr() {
        let Some(spec) = WorkerSpec::realesrgan_fake("crash") else {
            eprintln!("skip: Real-ESRGAN python runtime unavailable");
            return;
        };
        let dir = temp_dir();
        let request =
            AiProcessRequest::new("test-crash", Path::new("input.png"), &dir.join("out.png"));
        let error = run_worker(&spec, &request, None).await.unwrap_err();
        match error {
            WorkerError::Exited { code, stderr } => {
                assert_eq!(code, Some(1));
                assert!(stderr.contains("fake realesrgan worker process crash"));
            }
            other => panic!("expected worker exit error, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn result_accepted_even_when_worker_exits_non_zero() {
        // 生产缺陷回归：任务已完成、result 已发出，但收尾阶段退出码非 0。
        // 已拿到 result 时不应把成功的任务判为失败。
        let Some(spec) = WorkerSpec::realesrgan_fake("success-bad-exit") else {
            eprintln!("skip: Real-ESRGAN python runtime unavailable");
            return;
        };
        let dir = temp_dir();
        let output = dir.join("output.png");
        let request = AiProcessRequest::new("test-bad-exit", Path::new("input.png"), &output);
        let run = run_worker(&spec, &request, None).await.unwrap();
        assert_eq!(run.result.model_id, "realesrgan-x4plus");
        assert!(output.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn realesrgan_real_worker_forward_pass() {
        let Some(spec) = WorkerSpec::realesrgan() else {
            eprintln!("skip: Real-ESRGAN python runtime unavailable");
            return;
        };
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let weight = root
            .join("models")
            .join("cache")
            .join("realesrgan")
            .join("RealESRGAN_x4plus.pth");
        if !weight.is_file() {
            eprintln!("skip: missing weight {}", weight.display());
            return;
        }
        let dir = temp_dir();
        // 用 Pillow 生成一张小测试图（开发环境已具备）
        let input = dir.join("input.png");
        let python_code = "from PIL import Image; Image.new('RGB',(160,120),(128,64,32)).save(r'INPUT')"
            .replace("INPUT", &input.to_string_lossy());
        let status = std::process::Command::new(find_python_realesrgan().unwrap())
            .args(["-c", &python_code])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if !status.map(|s| s.success()).unwrap_or(false) {
            eprintln!("skip: Pillow unavailable");
            let _ = std::fs::remove_dir_all(dir);
            return;
        }
        let output = dir.join("output.png");
        let mut request = AiProcessRequest::new("test-real", &input, &output);
        request.model_id = "realesrgan-x4plus".into();
        request.device = "cpu".into();

        let run = run_worker(&spec, &request, None).await.unwrap();
        assert_eq!(run.result.model_id, "realesrgan-x4plus");
        assert_eq!(run.result.width, 640);
        assert_eq!(run.result.height, 480);
        assert_eq!(run.result.scale, 4);
        assert!(output.exists(), "真实前向应产出文件");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn strip_progress_noise_keeps_traceback() {
        let mut stderr = String::new();
        for i in 0..200 {
            stderr.push_str(&format!(
                "Tile {}/6 [00:0{}<00:0{}, 1.2it/s]\r",
                i % 7,
                i / 7,
                6 - i / 7
            ));
        }
        stderr.push_str("Traceback (most recent call last):\n");
        stderr.push_str("  File \"worker.py\", line 932, in enhance\n");
        stderr.push_str("RuntimeError: CUDA out of memory\n");

        let cleaned = strip_progress_noise(&stderr);
        assert!(!cleaned.contains("Tile "), "进度条应被过滤: {cleaned}");
        assert!(!cleaned.contains("it/s"), "进度条速率应被过滤: {cleaned}");
        assert!(cleaned.contains("RuntimeError"), "真实异常应保留: {cleaned}");
        assert!(cleaned.contains("Traceback"), "traceback 应保留: {cleaned}");
    }

    #[test]
    fn strip_progress_noise_all_noise_becomes_empty() {
        let stderr = "Tile 0/6 [00:00<?, ?it/s]\rTile 6/6 [00:01, 1.3it/s]";
        assert_eq!(strip_progress_noise(stderr), "");
    }
}
