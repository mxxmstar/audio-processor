//! 音视频直链挑选
//!
//! 对应 Go 版 `bilidownload/server/task/task.go` 的 `GetVideoURL` / `GetAudioURL`。
//! 入参来自 `get_play_info` 返回的 `PlayInfo`（DASH 流），按 Codecid / 清晰度优先级挑选直链。

use crate::biliapi::types::{Dash, Media, MediaFormat};

/// 视频编码 Codecid 含义（对齐 B 站 DASH 协议）
pub mod codecid {
    /// AV1
    pub const AV1: i64 = 12;
    /// H.264 / AVC
    pub const H264: i64 = 7;
    /// HEVC / H.265
    pub const HEVC: i64 = 13;
}

/// 视频流的 Codecid 优选顺序：AV1 → H.264 → HEVC
/// 顺序靠前的编码优先（体积更小 / 兼容性更好随场景取舍，沿用 Go 版优先级）。
const VIDEO_CODEC_PRIORITY: [i64; 3] = [codecid::AV1, codecid::H264, codecid::HEVC];

/// 在给定清晰度 `format` 下，按 Codecid 优先级挑选视频直链。
///
/// 对应 Go 版 `GetVideoURL(medias, format)`：
/// ```go
/// for _, code := range []int{12, 7, 13} {
///     for _, item := range medias {
///         if item.ID == format && item.Codecid == code {
///             return item.BaseURL, nil
///         }
///     }
/// }
/// ```
///
/// 返回 `None` 表示在目标清晰度下没有可用流（可能需降低 `format` 重试）。
pub fn select_video_url(medias: &[Media], format: MediaFormat) -> Option<String> {
    for &codec in VIDEO_CODEC_PRIORITY.iter() {
        for item in medias {
            if item.id == format.0 && item.codecid == codec {
                return Some(item.base_url.clone());
            }
        }
    }
    None
}

/// 挑选最佳音频直链。
///
/// 对应 Go 版 `GetAudioURL(dash)`：
/// - 优先返回无损 FLAC 流；
/// - 否则按 `ID`（码率等级）选最大的音频流。
///
/// 返回 `None` 表示 DASH 中没有任何音轨。
pub fn select_audio_url(dash: &Dash) -> Option<String> {
    if let Some(flac) = &dash.flac {
        if !flac.audio.base_url.is_empty() {
            return Some(flac.audio.base_url.clone());
        }
    }
    let mut best: Option<&Media> = None;
    for item in dash.audio.iter() {
        match best {
            Some(b) if b.id >= item.id => {}
            _ => best = Some(item),
        }
    }
    best.and_then(|m| {
        if m.base_url.is_empty() {
            None
        } else {
            Some(m.base_url.clone())
        }
    })
}

/// 一次性从 `PlayInfo` 中挑选视频 + 音频直链。
///
/// `format` 为目标视频清晰度；当该清晰度下无可用视频流时，`video_url` 为 `None`，
/// 调用方可降级清晰度后重试 `select_video_url`。
pub fn select_streams(
    dash: &Dash,
    format: MediaFormat,
) -> StreamSelection {
    StreamSelection {
        video_url: select_video_url(&dash.video, format),
        audio_url: select_audio_url(dash),
    }
}

/// `select_streams` 的返回结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamSelection {
    /// 选中的视频直链（按清晰度+Codecid 优先级），无可用流时为 `None`。
    pub video_url: Option<String>,
    /// 选中的音频直链（FLAC 优先，否则最高码率），无音轨时为 `None`。
    pub audio_url: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biliapi::types::{Flac, Media};

    fn media(id: i64, codecid: i64, url: &str) -> Media {
        Media {
            id,
            base_url: url.to_string(),
            backup_url: vec![],
            bandwidth: 0,
            mime_type: String::new(),
            codecs: String::new(),
            width: 0,
            height: 0,
            codecid,
        }
    }

    #[test]
    fn test_select_video_url_priority() {
        // 同一清晰度下同时存在 AV1/H264/HEVC，应优先 AV1
        let medias = vec![
            media(MediaFormat::Q_1080P.0, codecid::HEVC, "hevc"),
            media(MediaFormat::Q_1080P.0, codecid::H264, "h264"),
            media(MediaFormat::Q_1080P.0, codecid::AV1, "av1"),
        ];
        assert_eq!(
            select_video_url(&medias, MediaFormat::Q_1080P),
            Some("av1".to_string())
        );
    }

    #[test]
    fn test_select_video_url_codec_fallback() {
        // 目标清晰度只有 HEVC，回退到 HEVC
        let medias = vec![media(MediaFormat::Q_1080P.0, codecid::HEVC, "hevc")];
        assert_eq!(
            select_video_url(&medias, MediaFormat::Q_1080P),
            Some("hevc".to_string())
        );
    }

    #[test]
    fn test_select_video_url_no_format() {
        // 没有目标清晰度，返回 None
        let medias = vec![media(MediaFormat::Q_720P.0, codecid::H264, "h264")];
        assert_eq!(select_video_url(&medias, MediaFormat::Q_1080P), None);
    }

    #[test]
    fn test_select_audio_url_flac_priority() {
        let dash = Dash {
            duration: 0,
            video: vec![],
            audio: vec![media(30280, 0, "aac")],
            flac: Some(Flac {
                audio: media(30251, 0, "flac"),
            }),
        };
        assert_eq!(select_audio_url(&dash), Some("flac".to_string()));
    }

    #[test]
    fn test_select_audio_url_highest_id() {
        // 无 FLAC，选 ID 最大的音轨
        let dash = Dash {
            duration: 0,
            video: vec![],
            audio: vec![
                media(30216, 0, "low"),
                media(30280, 0, "high"),
            ],
            flac: None,
        };
        assert_eq!(select_audio_url(&dash), Some("high".to_string()));
    }

    #[test]
    fn test_select_audio_url_empty() {
        let dash = Dash {
            duration: 0,
            video: vec![],
            audio: vec![],
            flac: None,
        };
        assert_eq!(select_audio_url(&dash), None);
    }
}
