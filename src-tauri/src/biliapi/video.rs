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
