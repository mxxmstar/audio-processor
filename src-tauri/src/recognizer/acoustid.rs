use serde::Deserialize;

use crate::recognizer::error::{AppError, Result};

/// AcoustID 开放 API 的客户端密钥（client key）。
/// 注意：生产环境建议改为从配置文件读取，避免硬编码。
const ACOUSTID_KEY: &str = "rheYMWAd4x";

/// AcoustID 接口返回的最外层结构。
#[derive(Deserialize)]
struct AcoustidResponse {
    status: String,                       // 固定为 "ok" 表示成功
    #[serde(default)]
    results: Vec<AcoustidResult>,        // 匹配结果列表（可能为空）
    #[serde(default)]
    error: Option<AcoustidError>,        // 失败时的错误信息
}

/// AcoustID 失败时的错误体。
#[derive(Deserialize)]
struct AcoustidError {
    message: String,
}

/// 单个匹配结果（一个指纹可能对应多个录音）。
#[derive(Deserialize)]
struct AcoustidResult {
    #[serde(default)]
    id: Option<String>,                  // 录音 ID（未请求 recordings meta 时直接在此返回）
    #[serde(default)]
    score: f64,                          // 该结果的匹配分数（0~1）
    #[serde(default)]
    recordings: Vec<Recording>,          // 该分数下对应的录音列表（请求 recordings meta 时返回）
}

/// AcoustID 中的录音（曲目）信息。
#[derive(Deserialize)]
struct Recording {
    id: String,                           // 录音 ID（用于后续查 MusicBrainz）
    #[serde(default)]
    title: Option<String>,                // 标题（可能缺失）
    #[serde(default)]
    artists: Vec<Artist>,                 // 艺术家列表
}

/// 艺术家信息。
#[derive(Deserialize)]
struct Artist {
    name: String,
}

/// 经过整理后返回给 `mod.rs` 的匹配数据。
pub struct AcoustidMatch {
    pub recording_id: String,   // 录音 ID
    pub title: String,          // 标题
    pub artist: String,         // 艺术家
    pub confidence: f64,        // 置信度（百分制）
}

/// 调用 AcoustID 的 lookup 接口。
///
/// 参数：
/// - `duration`：音频时长（秒），AcoustID 用于辅助匹配
/// - `fingerprint`：Chromaprint 指纹字符串
pub async fn lookup(duration: &f64, fingerprint: &str) -> Result<AcoustidMatch> {
    // 构造 HTTP 客户端
    let client = reqwest::Client::new();

    // 使用 POST 表单提交。AcoustID 支持 POST，且表单会自动对指纹中的
    // '+' '/' '=' 做正确 URL 编码，避免 GET 拼接时因特殊字符导致参数解析异常。
    let resp = client
        .post("https://api.acoustid.org/v2/lookup")
        .form(&[
            ("client", ACOUSTID_KEY),
            // AcoustID 要求 duration 为整数秒，带小数会被服务端拒绝（误报缺参数）
            ("duration", &format!("{:.0}", duration)),
            ("fingerprint", fingerprint),
            // meta 用空格分隔多个字段；form 编码后服务端能正确解析为 recordings + artists
            ("meta", "recordings artists"),
        ])
        .send()
        .await
        .map_err(|e| AppError::AcoustidRequest(e.to_string()))?;

    // 读取响应体文本（便于解析与错误提示）
    let text = resp
        .text()
        .await
        .map_err(|e| AppError::AcoustidRequest(e.to_string()))?;

    // 解析 JSON 响应
    let body: AcoustidResponse = serde_json::from_str(&text)
        .map_err(|e| AppError::AcoustidRequest(format!("{e}；原始: {text}")))?;

    // 检查状态字段
    if body.status != "ok" {
        let msg = body
            .error
            .map(|e| e.message)
            .unwrap_or_else(|| "未知错误".into());
        return Err(AppError::AcoustidRequest(msg));
    }

    // 取第一个结果（按返回顺序即最高分优先）
    let result = body.results.into_iter().next().ok_or(AppError::NoMatch)?;

    // 录音 ID 优先取 recordings 内第一个，否则用 result 顶层的 id
    let (recording_id, title, artist) = match result.recordings.into_iter().next() {
        Some(rec) => (
            rec.id,
            rec.title.unwrap_or_else(|| "未知标题".into()),
            rec.artists
                .first()
                .map(|a| a.name.clone())
                .unwrap_or_else(|| "未知艺术家".into()),
        ),
        None => (
            result.id.ok_or(AppError::NoMatch)?,
            "未知标题".into(),
            "未知艺术家".into(),
        ),
    };

    Ok(AcoustidMatch {
        recording_id,
        title,
        artist,
        confidence: result.score * 100.0, // 分数 0~1 转为百分比
    })
}
