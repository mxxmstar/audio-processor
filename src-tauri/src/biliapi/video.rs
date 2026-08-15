//! 视频/番剧/收藏夹相关 B 站 API 封装
//!
//! 对应 Go 版 `bilidownload/server/bilibili/video.go` 的解析类接口。
//! 所有需要 WBI 签名的接口（view / playurl）都通过 `BiliClient::with_wbi` 自动签名。

use crate::biliapi::client::{BiliClient, BASE_API};
use crate::biliapi::error::Result;
use crate::biliapi::types;

/// 获取单个视频信息（需 WBI 签名）
/// 对应 `x/web-interface/wbi/view`
pub async fn get_video_info(client: &BiliClient, bvid: &str) -> Result<types::VideoInfo> {
    let cfg = client.get(&format!("{}/x/web-interface/wbi/view", BASE_API));
    let cfg = cfg.query("bvid", bvid);
    let cfg = client.with_wbi(cfg).await?;
    client.send_json::<types::VideoInfo>(cfg).await
}

/// 获取番剧/影视信息（无需 WBI 签名）
/// 对应 `pgc/view/web/season`
pub async fn get_season_info(
    client: &BiliClient,
    epid: Option<&str>,
    ssid: Option<&str>,
) -> Result<types::SeasonInfo> {
    let cfg = client.get(&format!("{}/pgc/view/web/season", BASE_API));
    let cfg = match (epid, ssid) {
        (Some(ep), _) => cfg.query("ep_id", ep.to_string()),
        (_, Some(ss)) => cfg.query("season_id", ss.to_string()),
        _ => return Err(crate::biliapi::error::BiliApiError::Other(
            "get_season_info 需要 epid 或 ssid".into(),
        )),
    };
    client.send_json::<types::SeasonInfo>(cfg).await
}

/// 获取播放直链（需 WBI 签名）
/// 对应 `x/player/playurl`，fnval=4048(DASH) fourk=1(支持4K/8K)
pub async fn get_play_info(
    client: &BiliClient,
    bvid: &str,
    cid: i64,
    format: i64,
) -> Result<types::PlayInfo> {
    let cfg = client.get(&format!("{}/x/player/playurl", BASE_API));
    let cfg = cfg
        .query("bvid", bvid)
        .query("cid", cid.to_string())
        .query("qn", format.to_string())
        .query("fnval", "4048")
        .query("fnver", "0")
        .query("fourk", "1");
    let cfg = client.with_wbi(cfg).await?;
    client.send_json::<types::PlayInfo>(cfg).await
}

/// 从播放信息中挑选音视频直链
///
/// 封装 `get_play_info` + `stream::select_streams`：`format` 为目标视频清晰度，
/// 音频按 FLAC 优先 / 否则最高码率自动挑选。返回 `None` 视频直链表示目标清晰度不可用，
/// 调用方可降低 `format` 后重试 `select_video_url`。
pub async fn get_streams(
    client: &BiliClient,
    bvid: &str,
    cid: i64,
    format: i64,
) -> Result<crate::biliapi::stream::StreamSelection> {
    let play = get_play_info(client, bvid, cid, format).await?;
    let dash = play.dash.ok_or_else(|| {
        crate::biliapi::error::BiliApiError::Other("播放信息缺少 DASH 流（可能无权限或需登录）".into())
    })?;
    Ok(crate::biliapi::stream::select_streams(
        &dash,
        types::MediaFormat(format),
    ))
}

/// 获取收藏夹列表（无需 WBI 签名，分页）
/// 对应 `x/v3/fav/resource/list`
pub async fn get_fav_list(
    client: &BiliClient,
    media_id: &str,
    pn: i64,
    ps: i64,
) -> Result<Vec<types::FavItem>> {
    let cfg = client
        .get(&format!("{}/x/v3/fav/resource/list", BASE_API))
        .query("media_id", media_id.to_string())
        .query("pn", pn.to_string())
        .query("ps", ps.to_string())
        .query("platform", "web");
    // fav 接口用 BaseRes，data 为 { medias: [...] }
    #[derive(serde::Deserialize)]
    struct FavData {
        medias: Option<Vec<types::FavItem>>,
    }
    let data = client.send_json::<FavData>(cfg).await?;
    Ok(data.medias.unwrap_or_default())
}

/// 获取合集首个视频 bvid（无需 WBI 签名）
/// 对应 `x/polymer/web-space/seasons_archives_list`
pub async fn get_seasons_archives_first_bvid(
    client: &BiliClient,
    mid: &str,
    season_id: &str,
) -> Result<String> {
    let cfg = client
        .get(&format!("{}/x/polymer/web-space/seasons_archives_list", BASE_API))
        .query("mid", mid.to_string())
        .query("season_id", season_id.to_string())
        .query("web_location", "333.1007");
    let data = client.send_json::<types::SeasonsArchives>(cfg).await?;
    data.episodes
        .first()
        .map(|e| e.bvid.clone())
        .ok_or_else(|| crate::biliapi::error::BiliApiError::Other("合集无视频".into()))
}

/// 获取热门视频（无需 WBI 签名）
/// 对应 `x/web-interface/popular`
pub async fn get_popular_videos(client: &BiliClient, pn: i64, ps: i64) -> Result<Vec<types::PopularVideo>> {
    let cfg = client
        .get(&format!("{}/x/web-interface/popular", BASE_API))
        .query("pn", pn.to_string())
        .query("ps", ps.to_string());
    #[derive(serde::Deserialize)]
    struct PopularData {
        list: Option<Vec<types::PopularVideo>>,
    }
    let data = client.send_json::<PopularData>(cfg).await?;
    Ok(data.list.unwrap_or_default())
}

/// 合集全部分集 bvid（无需 WBI 签名，自动翻页）
/// 对应 `x/polymer/web-space/seasons_archives_list`
pub async fn get_collection_bvids(
    client: &BiliClient,
    mid: &str,
    season_id: &str,
) -> Result<Vec<String>> {
    let mut bvids = Vec::new();
    let mut page_num: i64 = 1;
    const PAGE_SIZE: i64 = 30;
    loop {
        let cfg = client
            .get(&format!("{}/x/polymer/web-space/seasons_archives_list", BASE_API))
            .query("mid", mid.to_string())
            .query("season_id", season_id.to_string())
            .query("page_num", page_num.to_string())
            .query("page_size", PAGE_SIZE.to_string())
            .query("web_location", "333.1007");
        #[derive(serde::Deserialize)]
        struct Archive {
            bvid: String,
        }
        #[derive(serde::Deserialize)]
        struct CollData {
            #[serde(default)]
            archives: Vec<Archive>,
            #[serde(default)]
            page: PageMeta,
        }
        #[derive(serde::Deserialize, Default)]
        struct PageMeta {
            #[serde(default)]
            total: i64,
        }
        let data = client.send_json::<CollData>(cfg).await?;
        let count = data.archives.len() as i64;
        for a in data.archives {
            if !a.bvid.is_empty() {
                bvids.push(a.bvid);
            }
        }
        if count == 0 {
            break;
        }
        let fetched = (page_num * PAGE_SIZE) as i64;
        if data.page.total > 0 && fetched >= data.page.total {
            break;
        }
        if count < PAGE_SIZE {
            break;
        }
        page_num += 1;
    }
    Ok(bvids)
}

// ===================== 解析编排（阶段 1） =====================

/// 单个分 P 的解析结果
#[derive(Debug, Clone)]
pub struct PageStream {
    /// 分 P 序号（从 1 开始）
    pub page: i64,
    /// 分 P 标题（part）
    pub part: String,
    /// 选中的视频直链（按清晰度 + Codecid 优先级）
    pub video_url: Option<String>,
    /// 选中的音频直链（FLAC 优先，否则最高码率）
    pub audio_url: Option<String>,
    /// 实际选用的视频清晰度（降级后）
    pub actual_format: i64,
}

/// 单个视频（可能是多 P）的完整解析结果
#[derive(Debug, Clone)]
pub struct ResolveResult {
    pub bvid: String,
    pub title: String,
    pub pages: Vec<PageStream>,
}

/// 解析单个视频（多 P 遍历 + 清晰度降级）
///
/// `prefer_format` 为期望清晰度；当该清晰度在某分 P 无可用流时，
/// 按 `MediaFormat::fallback_chain` 依次降级，记录实际选用清晰度。
pub async fn resolve_video(
    client: &BiliClient,
    bvid: &str,
    prefer_format: i64,
) -> Result<ResolveResult> {
    let info = get_video_info(client, bvid).await?;
    let prefer = types::MediaFormat(prefer_format);
    let mut pages = Vec::new();
    for p in info.pages.iter() {
        let mut actual = prefer;
        let mut selection = None;
        for fmt in prefer.fallback_chain() {
            let play = get_play_info(client, bvid, p.cid, fmt.0).await?;
            if let Some(dash) = &play.dash {
                let sel = crate::biliapi::stream::select_streams(dash, fmt);
                if sel.video_url.is_some() {
                    actual = fmt;
                    selection = Some(sel);
                    break;
                }
            }
        }
        let sel = selection.unwrap_or_else(|| crate::biliapi::stream::StreamSelection {
            video_url: None,
            audio_url: None,
        });
        pages.push(PageStream {
            page: p.page,
            part: p.part.clone(),
            video_url: sel.video_url,
            audio_url: sel.audio_url,
            actual_format: actual.0,
        });
    }
    Ok(ResolveResult {
        bvid: info.bvid,
        title: info.title,
        pages,
    })
}

/// 解析合集：先拉全部分集 bvid，再逐个 `resolve_video`
pub async fn resolve_collection(
    client: &BiliClient,
    mid: &str,
    season_id: &str,
    prefer_format: i64,
) -> Result<Vec<ResolveResult>> {
    let bvids = get_collection_bvids(client, mid, season_id).await?;
    let mut results = Vec::new();
    for bvid in bvids {
        results.push(resolve_video(client, &bvid, prefer_format).await?);
    }
    Ok(results)
}

/// 解析番剧/影视：遍历 `episodes`，每集直接取播放直链
pub async fn resolve_season(
    client: &BiliClient,
    ssid: &str,
    prefer_format: i64,
) -> Result<Vec<ResolveResult>> {
    let season = get_season_info(client, None, Some(ssid)).await?;
    let prefer = types::MediaFormat(prefer_format);
    let mut results = Vec::new();
    for ep in season.episodes.iter() {
        if ep.bvid.is_empty() || ep.cid == 0 {
            continue;
        }
        let mut actual = prefer;
        let mut selection = None;
        for fmt in prefer.fallback_chain() {
            let play = get_play_info(client, &ep.bvid, ep.cid, fmt.0).await?;
            if let Some(dash) = &play.dash {
                let sel = crate::biliapi::stream::select_streams(dash, fmt);
                if sel.video_url.is_some() {
                    actual = fmt;
                    selection = Some(sel);
                    break;
                }
            }
        }
        let sel = selection.unwrap_or_else(|| crate::biliapi::stream::StreamSelection {
            video_url: None,
            audio_url: None,
        });
        results.push(ResolveResult {
            bvid: ep.bvid.clone(),
            title: if ep.long_title.is_empty() {
                ep.title.clone()
            } else {
                ep.long_title.clone()
            },
            pages: vec![PageStream {
                page: 1,
                part: ep.title.clone(),
                video_url: sel.video_url,
                audio_url: sel.audio_url,
                actual_format: actual.0,
            }],
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biliapi::types::MediaFormat;

    #[test]
    fn test_page_stream_fields() {
        let ps = PageStream {
            page: 2,
            part: "PV".into(),
            video_url: Some("https://v".into()),
            audio_url: Some("https://a".into()),
            actual_format: MediaFormat::Q_1080P.0,
        };
        assert_eq!(ps.page, 2);
        assert_eq!(ps.part, "PV");
        assert!(ps.video_url.is_some());
        assert!(ps.audio_url.is_some());
        assert_eq!(ps.actual_format, 80);
    }

    #[test]
    fn test_resolve_result_aggregate() {
        let r = ResolveResult {
            bvid: "BV1xx".into(),
            title: "合集标题".into(),
            pages: vec![
                PageStream {
                    page: 1,
                    part: "P1".into(),
                    video_url: Some("v1".into()),
                    audio_url: Some("a1".into()),
                    actual_format: 80,
                },
                PageStream {
                    page: 2,
                    part: "P2".into(),
                    video_url: None,
                    audio_url: Some("a2".into()),
                    actual_format: 64,
                },
            ],
        };
        assert_eq!(r.pages.len(), 2);
        // 第二分 P 视频直链缺失（清晰度降级后仍无视频流），但音频仍在
        assert!(r.pages[1].video_url.is_none());
        assert!(r.pages[1].audio_url.is_some());
    }
}
