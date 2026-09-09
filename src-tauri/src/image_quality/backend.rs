//! 图像超分后端的路由与默认参数解析。
//!
//! 图像域目前只有一个后端 RealESRGAN（运行在专用 `.venv-realesrgan`），
//! 路由优先级：环境变量 `AUDIO_AI_REALESRGAN_WORKER` → 按 manifest 选具体
//! Worker → fake Worker（仅 `AUDIO_AI_USE_FAKE=1`）。

use crate::image_quality::ai_worker::WorkerSpec;
use std::path::PathBuf;

pub const REALESRGAN_MODEL_ID: &str = "realesrgan-x4plus";
pub const REALESRGAN_SCALE: u32 = 4;

/// 一个模型的处理后端类型（图像域）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    RealEsrGan,
    Unknown,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::RealEsrGan => "realesrgan",
            Backend::Unknown => "unknown",
        }
    }

    /// 把 manifest / id 字符串解析为后端类型；未知值按 id 前缀回退。
    pub fn parse(value: &str) -> Backend {
        match value {
            "realesrgan" => Backend::RealEsrGan,
            other => {
                if other.contains("realesrgan") {
                    Backend::RealEsrGan
                } else {
                    Backend::Unknown
                }
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub backend: String,
}

/// 解析 `model_id` 对应的后端。优先从 manifest 读取 `backend` 字段；
/// 解析失败 / 未命中时按 id 是否包含 "realesrgan" 回退。
pub fn resolve_backend(model_id: &str) -> Backend {
    if let Some(backend) = backend_from_manifest(model_id) {
        return backend;
    }
    Backend::parse(model_id)
}

/// 根据 `model_id` 选择 Worker 规格。`AUDIO_AI_REALESRGAN_WORKER` 优先；
/// fake 模式选 realesrgan 的 fake Worker；否则选真实 realesrgan Worker。
pub fn select_worker_spec(_model_id: &str) -> Result<(WorkerSpec, String), String> {
    if let Some(path) = std::env::var_os("AUDIO_AI_REALESRGAN_WORKER") {
        return Ok((WorkerSpec::new(path), "configured".into()));
    }
    if std::env::var("AUDIO_AI_USE_FAKE").as_deref() == Ok("1") {
        return WorkerSpec::realesrgan_fake("success")
            .map(|spec| (spec, "fake".into()))
            .ok_or_else(|| "找不到 Python 运行时，无法启动 fake Worker".into());
    }
    WorkerSpec::realesrgan()
        .map(|spec| (spec, "realesrgan".into()))
        .ok_or_else(|| "找不到 Real-ESRGAN Python Worker 或 Python 运行时".into())
}

/// 列出 manifest 中图像域模型（仅 RealESRGAN 后端），供前端渲染可选模型。
/// 共享 manifest 含音频模型，这里按 backend 过滤，避免把音频模型混入图像界面。
pub fn list_models() -> Vec<ModelInfo> {
    match read_manifest() {
        Some(manifest) => manifest
            .models
            .into_iter()
            .filter(|entry| Backend::parse(&entry.backend) == Backend::RealEsrGan)
            .map(|entry| ModelInfo {
                id: entry.id,
                backend: Backend::parse(&entry.backend).as_str().to_string(),
            })
            .collect(),
        None => Vec::new(),
    }
}

fn manifest_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("AUDIO_AI_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
}

#[derive(Debug, serde::Deserialize)]
struct Manifest {
    #[serde(default)]
    models: Vec<ManifestEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct ManifestEntry {
    id: String,
    backend: String,
}

fn read_manifest() -> Option<Manifest> {
    let path = manifest_dir().join("manifest.json");
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn backend_from_manifest(model_id: &str) -> Option<Backend> {
    let manifest = read_manifest()?;
    let entry = manifest.models.iter().find(|model| model.id == model_id)?;
    Some(Backend::parse(&entry.backend))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_backend_routes_realesrgan() {
        assert_eq!(Backend::parse("realesrgan-x4plus"), Backend::RealEsrGan);
        assert_eq!(Backend::parse("realesrgan"), Backend::RealEsrGan);
        assert_eq!(Backend::parse("totally-unknown"), Backend::Unknown);
    }

    #[test]
    fn list_models_reads_backend_field() {
        // 不混入音频后端模型；真实仓库 manifest 含 realesrgan-x4plus，过滤后只返回它。
        let models = list_models();
        for model in &models {
            assert_eq!(model.backend, "realesrgan");
        }
    }
}
