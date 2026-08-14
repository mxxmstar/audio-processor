use serde::Deserialize;

use crate::recognizer::error::{AppError, Result};

/// MusicBrainz 录音接口返回结构。
#[derive(Deserialize)]
struct RecordingResponse {
    // release-list 是发行（专辑）列表
    #[serde(default)]
    #[serde(rename = "release-list")]
    release_list: Vec<Release>,
}

/// 发行（专辑）信息。
#[derive(Deserialize)]
struct Release {
    title: String,             // 专辑名
    #[serde(default)]
    date: Option<String>,       // 发行日期（可能缺失）
}

/// 整理后返回给 `mod.rs` 的专辑信息。
pub struct RecordingInfo {
    pub album: Option<String>,
    pub album_date: Option<String>,
}

/// 根据 AcoustID 提供的录音 ID 查询 MusicBrainz，补充专辑信息。
///
/// MusicBrainz 对 User-Agent 有强制要求，因此客户端需设置 UA。
pub async fn get_recording(id: &str) -> Result<RecordingInfo> {
    // 构造带 User-Agent 的客户端（MusicBrainz API 规范要求）
    let client = reqwest::Client::builder()
        .user_agent("AudioProcessor/0.1 (https://github.com/mxxmstar/audio-processor)")
        .build()
        .map_err(|e| AppError::MusicBrainzRequest(e.to_string()))?;

    // 请求录音详情，要求返回 artists 与 releases 关系
    let resp = client
        .get(format!(
            "https://musicbrainz.org/ws/2/recording/{id}?fmt=json&inc=artists+releases"
        ))
        .send()
        .await
        .map_err(|e| AppError::MusicBrainzRequest(e.to_string()))?;

    let body: RecordingResponse = resp
        .json()
        .await
        .map_err(|e| AppError::MusicBrainzRequest(e.to_string()))?;

    // 取第一个发行作为专辑（通常为最早发行）
    let (album, album_date) = match body.release_list.into_iter().next() {
        Some(rel) => (Some(rel.title), rel.date),
        None => (None, None),
    };

    Ok(RecordingInfo { album, album_date })
}
