//! 音频质量处理相关 Tauri 命令。
//!
//! 当前只提供 AI Worker 运行时握手检查。真实模型和音频处理命令在
//! Worker 协议稳定后继续实现。

use crate::audio_quality::ai_worker::{self, WorkerSpec, PROTOCOL_VERSION};
use serde::Serialize;

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
    pub error: Option<String>,
}

/// 检查 Python AI Worker 是否能启动并完成 JSONL ready 握手。
///
/// `AUDIO_AI_WORKER` 可指定未来打包后的 Worker 可执行文件；未配置时使用
/// 仓库内 fake Worker，仅用于开发协议验证，并通过 `productionReady=false`
/// 明确标记其不是实际 AI 模型运行时。
#[tauri::command]
pub async fn audio_quality_check_ai_runtime() -> Result<AiRuntimeCheck, String> {
    let (spec, worker_kind, production_ready) =
        if let Some(path) = std::env::var_os("AUDIO_AI_WORKER") {
            (WorkerSpec::new(path), "configured".to_string(), true)
        } else {
            let Some(spec) = WorkerSpec::fake("success") else {
                return Ok(AiRuntimeCheck {
                    available: false,
                    production_ready: false,
                    protocol_version: PROTOCOL_VERSION,
                    worker_kind: "unavailable".into(),
                    program: None,
                    worker_version: None,
                    models: Vec::new(),
                    error: Some("找不到 Python 运行时，请设置 AUDIO_AI_PYTHON".into()),
                });
            };
            (spec, "fake".to_string(), false)
        };

    let program = spec.program.to_string_lossy().into_owned();
    match ai_worker::probe_worker(&spec).await {
        Ok(ready) => Ok(AiRuntimeCheck {
            available: true,
            production_ready,
            protocol_version: PROTOCOL_VERSION,
            worker_kind,
            program: Some(program),
            worker_version: Some(ready.worker_version),
            models: ready.models,
            error: None,
        }),
        Err(error) => Ok(AiRuntimeCheck {
            available: false,
            production_ready: false,
            protocol_version: PROTOCOL_VERSION,
            worker_kind,
            program: Some(program),
            worker_version: None,
            models: Vec::new(),
            error: Some(error.to_string()),
        }),
    }
}
