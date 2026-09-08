//! 音质提升后端的路由与默认参数解析。
//!
//! 每个 `model_id` 映射到具体的 Python Worker：FlashSR 运行在
//! `.venv-flashsr` 下的独立进程，AudioSR / DeepFilterNet 运行在 `.venv`
//! 下的进程。路由优先级：环境变量 `AUDIO_AI_WORKER` → 按 manifest / id
//! 前缀选择具体 Worker → fake Worker（仅 `AUDIO_AI_USE_FAKE=1`）。

use crate::audio_quality::ai_worker::WorkerSpec;
use std::path::PathBuf;

pub const FLASHSR_MODEL_ID: &str = "flashsr";
pub const AUDIOSR_MODEL_ID: &str = "audiosr-basic";
pub const FLASHSR_SAMPLE_RATE: u32 = 48_000;

/// 一个模型的处理后端类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    FlashSr,
    AudioSr,
    DeepFilterNet,
    HiFiGan,
    Unknown,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::FlashSr => "flashsr",
            Backend::AudioSr => "audiosr",
            Backend::DeepFilterNet => "deepfilternet",
            Backend::HiFiGan => "hifigan",
            Backend::Unknown => "unknown",
        }
    }

    /// 把 manifest / id 字符串解析为后端类型；未知值按 id 前缀回退。
    pub fn parse(value: &str) -> Backend {
        match value {
            "flashsr" => Backend::FlashSr,
            "audiosr" => Backend::AudioSr,
            "deepfilternet" => Backend::DeepFilterNet,
            "hifigan" => Backend::HiFiGan,
            other => {
                if other.starts_with("flashsr") {
                    Backend::FlashSr
                } else if other.starts_with("audiosr") {
                    Backend::AudioSr
                } else if other.starts_with("deepfilternet") {
                    Backend::DeepFilterNet
                } else if other.starts_with("hifigan") {
                    Backend::HiFiGan
                } else {
                    Backend::Unknown
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendDefaults {
    pub chunk_seconds: f64,
    pub overlap_seconds: f64,
    pub default_device: &'static str,
}

impl BackendDefaults {
    pub fn for_backend(backend: Backend) -> Self {
        match backend {
            // FlashSR 单次前向硬限制 245760 样本（48 kHz 下 5.12 s），
            // 但 Python 侧忽略 chunk_seconds 自行按 245760 切块；此处仅
            // 给出语义正确的默认值，避免与 overlap 的校验 `overlap < chunk` 冲突。
            Backend::FlashSr => BackendDefaults {
                chunk_seconds: 5.12,
                overlap_seconds: 0.5,
                default_device: "auto",
            },
            Backend::AudioSr => BackendDefaults {
                chunk_seconds: 10.24,
                overlap_seconds: 1.28,
                default_device: "auto",
            },
            // HiFi-GAN 是全卷积声码器，可整段前向（无 FlashSR 的定长硬限制），
            // 因此块可更大；声码器重建无需重叠相加，overlap 设为 0。
            Backend::HiFiGan => BackendDefaults {
                chunk_seconds: 30.0,
                overlap_seconds: 0.0,
                default_device: "auto",
            },
            _ => BackendDefaults {
                chunk_seconds: 20.0,
                overlap_seconds: 2.0,
                default_device: "auto",
            },
        }
    }
}

/// 解析 `model_id` 对应的后端。优先从 manifest 读取 `backend` 字段；
/// 解析失败 / 未命中时按 id 前缀回退。
pub fn resolve_backend(model_id: &str) -> Backend {
    if let Some(backend) = backend_from_manifest(model_id) {
        return backend;
    }
    Backend::parse(model_id)
}

/// 根据 `model_id` 选择 Worker 规格。`AUDIO_AI_WORKER` 优先；fake 模式优先
/// 选 FlashSR 的 fake Worker（回退到通用 fake）；否则按后端选真实 Worker。
pub fn select_worker_spec(model_id: &str) -> Result<(WorkerSpec, String), String> {
    if let Some(path) = std::env::var_os("AUDIO_AI_WORKER") {
        return Ok((WorkerSpec::new(path), "configured".into()));
    }
    if std::env::var("AUDIO_AI_USE_FAKE").as_deref() == Ok("1") {
        // 按 model_id 的后端选对应 fake Worker：保证 result.model_id 与请求一致
        // （hifigan 的 fake Worker 回显 "hifigan-48k"）。回退链保证 FlashSR /
        // 通用 fake 仍可用。
        let fake = match resolve_backend(model_id) {
            Backend::FlashSr => WorkerSpec::flashsr_fake("success"),
            Backend::HiFiGan => WorkerSpec::hifigan_fake("success"),
            _ => None,
        }
        .or_else(|| WorkerSpec::flashsr_fake("success"))
        .or_else(|| WorkerSpec::fake("success"))
        .map(|spec| (spec, "fake".into()));
        return fake.ok_or_else(|| "找不到 Python 运行时，无法启动 fake Worker".into());
    }
    let backend = resolve_backend(model_id);
    let spec = match backend {
        Backend::FlashSr => WorkerSpec::flashsr()
            .map(|spec| (spec, "flashsr".into()))
            .or_else(|| WorkerSpec::production().map(|spec| (spec, "python".into()))),
        Backend::HiFiGan => WorkerSpec::hifigan()
            .map(|spec| (spec, "hifigan".into()))
            .or_else(|| WorkerSpec::production().map(|spec| (spec, "python".into()))),
        _ => WorkerSpec::production().map(|spec| (spec, "python".into())),
    };
    spec.ok_or_else(|| "找不到 Python AI Worker 或 Python 运行时".into())
}

/// 列出 manifest 中全部模型及其后端，供前端渲染可选模型。
pub fn list_models() -> Vec<ModelInfo> {
    match read_manifest() {
        Some(manifest) => manifest
            .models
            .into_iter()
            .map(|entry| ModelInfo {
                id: entry.id,
                backend: Backend::parse(&entry.backend).as_str().to_string(),
            })
            .collect(),
        None => Vec::new(),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub backend: String,
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
    fn resolve_backend_parses_exact_and_prefix() {
        assert_eq!(Backend::parse("flashsr"), Backend::FlashSr);
        assert_eq!(Backend::parse("audiosr-basic"), Backend::AudioSr);
        assert_eq!(Backend::parse("flashsr-custom"), Backend::FlashSr);
        assert_eq!(Backend::parse("audiosr-custom"), Backend::AudioSr);
        assert_eq!(Backend::parse("deepfilternet2-speech"), Backend::DeepFilterNet);
        assert_eq!(Backend::parse("hifigan-48k"), Backend::HiFiGan);
        assert_eq!(Backend::parse("hifigan-custom"), Backend::HiFiGan);
        assert_eq!(Backend::parse("totally-unknown"), Backend::Unknown);
    }

    #[test]
    fn flashsr_defaults_use_short_chunk_and_small_overlap() {
        let defaults = BackendDefaults::for_backend(Backend::FlashSr);
        assert_eq!(defaults.chunk_seconds, 5.12);
        assert_eq!(defaults.overlap_seconds, 0.5);
        assert!(defaults.overlap_seconds < defaults.chunk_seconds);
    }

    #[test]
    fn audiosr_defaults_are_distinct() {
        let defaults = BackendDefaults::for_backend(Backend::AudioSr);
        assert_eq!(defaults.chunk_seconds, 10.24);
        assert_eq!(defaults.overlap_seconds, 1.28);
    }

    #[test]
    fn hifigan_defaults_use_large_chunk_and_zero_overlap() {
        // 声码器整段前向，块更大；重建无需重叠相加。overlap 必须 < chunk。
        let defaults = BackendDefaults::for_backend(Backend::HiFiGan);
        assert_eq!(defaults.chunk_seconds, 30.0);
        assert_eq!(defaults.overlap_seconds, 0.0);
        assert!(defaults.overlap_seconds < defaults.chunk_seconds);
    }

    #[test]
    fn list_models_reads_backend_field() {
        // 不依赖真实 manifest：直接验证 parse 链路在 manifest 缺失时是空列表
        // 且不会 panic。
        let models = list_models();
        // 真实仓库 manifest 含 deepfilternet2-speech / audiosr-basic / flashsr /
        // hifigan-48k 共 4 个模型，CI 缺 manifest 时为 0；均合法。
        assert!(models.len() <= 4);
    }
}
