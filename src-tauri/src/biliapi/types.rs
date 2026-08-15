//! B 站 API 响应及业务数据结构
//!
//! 字段与 `bilidownload/server/bilibili/type.go` 对齐。
//! 仅保留本模块封装接口所需字段（按需裁剪，未使用的嵌套字段省略）。

use serde::Deserialize;

/// 清晰度枚举（对齐 Go 版 common.MediaFormat）
/// B 站 DASH 质量代码
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct MediaFormat(pub i64);

impl MediaFormat {
    pub const Q_360P: MediaFormat = MediaFormat(16);
    pub const Q_720P: MediaFormat = MediaFormat(64);
    pub const Q_1080P: MediaFormat = MediaFormat(80);
    pub const Q_1080P_PLUS: MediaFormat = MediaFormat(112);
    pub const Q_4K: MediaFormat = MediaFormat(120);
    pub const Q_DOLBY: MediaFormat = MediaFormat(126);
    pub const Q_HDR: MediaFormat = MediaFormat(125);
    pub const Q_8K: MediaFormat = MediaFormat(127);
}

/// B 站统一响应外壳：`code` + `message/msg` + `data`
#[derive(Debug, Deserialize)]
pub struct BaseRes<T> {
    pub code: i64,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub msg: String,
    pub data: Option<T>,
}

impl<T> BaseRes<T> {
    /// 校验 code==0，否则返回 Api 错误
    pub fn into_result(self) -> Result<T, crate::biliapi::error::BiliApiError> {
        if self.code != 0 {
            return Err(crate::biliapi::error::BiliApiError::Api {
                code: self.code,
                message: if !self.message.is_empty() {
                    self.message
                } else {
                    self.msg.clone()
                },
            });
        }
        self.data.ok_or_else(|| {
            crate::biliapi::error::BiliApiError::Other("响应缺少 data 字段".into())
        })
    }
}

/// 二维码生成响应 data
#[derive(Debug, Deserialize, Default)]
pub struct QrInfo {
    pub url: String,
    pub qrcode_key: String,
}

/// 扫码状态常量
pub const QR_NO_SCAN: i64 = 86101; // 未扫码
pub const QR_NO_CLICK: i64 = 86090; // 已扫码未确认
pub const QR_EXPIRES: i64 = 86038; // 已过期
pub const QR_SUCCESS: i64 = 0; // 已确认登录

/// 二维码轮询响应 data
#[derive(Debug, Deserialize, Default)]
pub struct QrStatus {
    pub url: String,
    #[serde(default)]
    pub refresh_token: String,
    pub code: i64,
    #[serde(default)]
    pub message: String,
}

/// nav 接口 wbi_img 子结构
#[derive(Debug, Deserialize, Default)]
pub struct NavWbiImg {
    pub img_url: String,
    pub sub_url: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct NavData {
    #[serde(rename = "wbi_img")]
    pub wbi_img: NavWbiImg,
}

/// 视频信息（getVideoInfo）
#[derive(Debug, Deserialize, Default)]
pub struct VideoInfo {
    pub bvid: String,
    #[serde(default)]
    pub aid: i64,
    #[serde(default)]
    pub pic: String,
    pub title: String,
    #[serde(default)]
    pub pubdate: i64,
    #[serde(default)]
    pub desc: String,
    #[serde(default)]
    pub owner: Owner,
    #[serde(default)]
    pub dimension: Dimension,
    #[serde(default)]
    pub pages: Vec<Page>,
    #[serde(default)]
    pub duration: i64,
    #[serde(default)]
    pub stat: Stat,
}

#[derive(Debug, Deserialize, Default)]
pub struct Owner {
    #[serde(default)]
    pub mid: i64,
    pub name: String,
    #[serde(default)]
    pub face: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct Dimension {
    #[serde(default)]
    pub width: i64,
    #[serde(default)]
    pub height: i64,
    #[serde(default)]
    pub rotate: i64,
}

#[derive(Debug, Deserialize, Default)]
pub struct Page {
    pub cid: i64,
    #[serde(default)]
    pub page: i64,
    #[serde(default)]
    pub part: String,
    #[serde(default)]
    pub duration: i64,
}

#[derive(Debug, Deserialize, Default)]
pub struct Stat {
    #[serde(default)]
    pub view: i64,
    #[serde(default)]
    pub danmaku: i64,
    #[serde(default)]
    pub reply: i64,
    #[serde(default)]
    pub favorite: i64,
    #[serde(default)]
    pub coin: i64,
    #[serde(default)]
    pub share: i64,
    #[serde(default)]
    pub like: i64,
}

/// 播放直链（getPlayInfo）
#[derive(Debug, Deserialize, Default)]
pub struct PlayInfo {
    #[serde(default)]
    pub accept_description: Vec<String>,
    pub dash: Option<Dash>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Dash {
    #[serde(default)]
    pub duration: i64,
    #[serde(default)]
    pub video: Vec<Media>,
    #[serde(default)]
    pub audio: Vec<Media>,
    #[serde(default)]
    pub flac: Option<Flac>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Flac {
    pub audio: Media,
}

#[derive(Debug, Deserialize, Default)]
pub struct Media {
    pub id: i64,
    pub base_url: String,
    #[serde(default)]
    pub backup_url: Vec<String>,
    #[serde(default)]
    pub bandwidth: i64,
    #[serde(default)]
    pub mime_type: String,
    #[serde(default)]
    pub codecs: String,
    #[serde(default)]
    pub width: i64,
    #[serde(default)]
    pub height: i64,
    #[serde(default)]
    pub codecid: i64,
}

/// 番剧信息（getSeasonInfo）
#[derive(Debug, Deserialize, Default)]
pub struct SeasonInfo {
    #[serde(default)]
    pub season_id: i64,
    #[serde(default)]
    pub season_title: String,
    pub title: String,
    #[serde(default)]
    pub cover: String,
    #[serde(default)]
    pub total: i64,
    #[serde(default)]
    pub evaluate: String,
    #[serde(default)]
    pub episodes: Vec<Episode>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Episode {
    #[serde(default)]
    pub aid: i64,
    #[serde(default)]
    pub bvid: String,
    #[serde(default)]
    pub cid: i64,
    #[serde(default)]
    pub cover: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub long_title: String,
    #[serde(default)]
    pub duration: i64,
}

/// 收藏夹条目（getFavList）
#[derive(Debug, Deserialize, Default)]
pub struct FavItem {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub cover: String,
    #[serde(default)]
    pub bvid: String,
    #[serde(default)]
    pub pubtime: i64,
    #[serde(default)]
    pub upper: FavUpper,
    #[serde(default)]
    pub ugc: FavUgc,
}

#[derive(Debug, Deserialize, Default)]
pub struct FavUpper {
    #[serde(default)]
    pub mid: i64,
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct FavUgc {
    #[serde(default)]
    pub first_cid: i64,
}

/// 合集首个视频（getSeasonsArchivesListFirstBvid）
#[derive(Debug, Deserialize, Default)]
pub struct SeasonsArchives {
    pub episodes: Vec<SeasonArchiveEpisode>,
}

#[derive(Debug, Deserialize, Default)]
pub struct SeasonArchiveEpisode {
    pub bvid: String,
    #[serde(default)]
    pub cid: i64,
}

/// 热门视频（getPopularVideos）
#[derive(Debug, Deserialize, Default)]
pub struct PopularVideo {
    pub bvid: String,
    #[serde(default)]
    pub aid: i64,
    pub title: String,
    #[serde(default)]
    pub pic: String,
    #[serde(default)]
    pub owner: Owner,
    #[serde(default)]
    pub stat: Stat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_res_ok() {
        let json = r#"{"code":0,"message":"ok","data":{"bvid":"BV1xx","aid":123,"title":"测试"}}"#;
        let base: BaseRes<VideoInfo> = serde_json::from_str(json).unwrap();
        let info = base.into_result().unwrap();
        assert_eq!(info.bvid, "BV1xx");
        assert_eq!(info.title, "测试");
    }

    #[test]
    fn test_base_res_error() {
        let json = r#"{"code":-404,"message":"啥都木有","data":null}"#;
        let base: BaseRes<VideoInfo> = serde_json::from_str(json).unwrap();
        let err = base.into_result().unwrap_err();
        match err {
            crate::biliapi::error::BiliApiError::Api { code, .. } => assert_eq!(code, -404),
            _ => panic!("期望 Api 错误"),
        }
    }

    #[test]
    fn test_qr_status_constants() {
        assert_eq!(QR_SUCCESS, 0);
        assert_eq!(QR_NO_SCAN, 86101);
        assert_eq!(QR_EXPIRES, 86038);
    }
}
