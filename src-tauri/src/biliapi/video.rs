//! 视频/番剧/收藏夹相关 B 站 API 封装
//!
//! 对应 Go 版 `bilidownload/server/bilibili/video.go` 的解析类接口。
//! 所有需要 WBI 签名的接口（view / playurl）都通过 `BiliClient::with_wbi` 自动签名。

use crate::biliapi::client::{BiliClient, BASE_API};
use crate::biliapi::error::Result;
use crate::biliapi::task::TaskGroup;
use crate::biliapi::types;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// 统一并发度读取入口。
///
/// 优先读取环境变量 `BILI_CONCURRENCY`（同时控制解析与下载阶段的并发）；
/// 否则回退到 `default_val`。值非法或 ≤ 0 时回退默认。
pub fn resolve_concurrency(default_val: usize) -> usize {
    std::env::var("BILI_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default_val)
}

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

/// 获取合集首个视频 bvid
/// 对应 `x/polymer/web-space/seasons_archives_list`（已被 B 站强制要求 WBI 签名）
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
    // 该接口已被 B 站强制要求 WBI 签名，未签名会返回 -352（请求被拦截）
    let cfg = client.with_wbi(cfg).await?;
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

/// 合集分页中单个视频条目（仅取 bvid）
#[derive(serde::Deserialize)]
struct Archive {
    bvid: String,
}

/// 合集全部分集 bvid（无需 WBI 签名，并发翻页）
/// 对应 `x/polymer/web-space/seasons_archives_list`
///
/// 先取第 1 页拿到 `total` 计算总页数，再并发拉取剩余页（受
/// `BILI_CONCURRENCY` 限制），最后按页号排序合并，保持原顺序。
pub async fn get_collection_bvids(
    client: &BiliClient,
    mid: &str,
    season_id: &str,
) -> Result<(Vec<String>, Option<String>)> {
    const PAGE_SIZE: i64 = 30;

    // 本地分页结果结构
    #[derive(serde::Deserialize, Default)]
    struct PageMeta {
        #[serde(default)]
        total: i64,
    }
    #[derive(serde::Deserialize, Default)]
    struct Meta {
        #[serde(default)]
        name: String,
    }
    #[derive(serde::Deserialize, Default)]
    struct CollData {
        #[serde(default)]
        archives: Vec<Archive>,
        #[serde(default)]
        page: PageMeta,
        #[serde(default)]
        meta: Meta,
    }

    // 单页请求（避免 reqwest builder 跨 await 借用问题，内联构造）
    async fn fetch_page(
        client: &BiliClient,
        mid: &str,
        season_id: &str,
        page_num: i64,
    ) -> Result<CollData> {
        let cfg = client
            .get(&format!("{}/x/polymer/web-space/seasons_archives_list", BASE_API))
            .query("mid", mid.to_string())
            .query("season_id", season_id.to_string())
            .query("page_num", page_num.to_string())
            .query("page_size", PAGE_SIZE.to_string())
            .query("web_location", "333.1007");
        // 该接口已被 B 站强制要求 WBI 签名，未签名会返回 -352（请求被拦截）
        let cfg = client.with_wbi(cfg).await?;
        client.send_json::<CollData>(cfg).await
    }

    // 第 1 页：确定总量与总页数
    let first = fetch_page(client, mid, season_id, 1).await?;
    let total = first.page.total.max(0);
    let first_count = first.archives.len() as i64;
    let total_pages = if total > 0 {
        ((total + PAGE_SIZE - 1) / PAGE_SIZE).max(1)
    } else {
        // 无 total 字段时，首页不满页即只有 1 页
        if first_count < PAGE_SIZE { 1 } else { 1 }
    };

    // 首页结果已拿到，先收集
    let mut pages: Vec<(i64, Vec<String>)> = Vec::new();
    pages.push((1, collect_bvids(&first.archives)));

    if total_pages > 1 {
        let concurrency = resolve_concurrency(4);
        let sem = Arc::new(Semaphore::new(concurrency));
        let client = Arc::new(client.clone());
        let mid = mid.to_string();
        let season_id = season_id.to_string();

        let mut handles = Vec::new();
        for pn in 2..=total_pages {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let client = client.clone();
            let mid = mid.clone();
            let sid = season_id.clone();
            handles.push(tokio::spawn(async move {
                let res = fetch_page(&client, &mid, &sid, pn).await;
                drop(permit);
                (pn, res)
            }));
        }
        for h in handles {
            let (pn, res) = h.await.map_err(|e| {
                crate::biliapi::error::BiliApiError::Other(format!("合集翻页任务被取消: {e}"))
            })?;
            let data = res?;
            pages.push((pn, collect_bvids(&data.archives)));
        }
    }

    // 按页号排序后合并（同页内已有序）
    pages.sort_by_key(|(pn, _)| *pn);
    let mut bvids = Vec::new();
    for (_, mut items) in pages {
        bvids.append(&mut items);
    }
    let title = if first.meta.name.is_empty() {
        None
    } else {
        Some(first.meta.name.clone())
    };
    Ok((bvids, title))
}

/// 从单页 archives 过滤出非空 bvid
fn collect_bvids(archives: &[Archive]) -> Vec<String> {
    archives
        .iter()
        .map(|a| a.bvid.clone())
        .filter(|b| !b.is_empty())
        .collect()
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
    /// 视频封面 URL（用于前端展示），无则空串
    pub cover: String,
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
        cover: info.pic.clone(),
    })
}

/// 解析合集：先拉全部分集 bvid，再并发 `resolve_video`
///
/// `concurrency` 控制同时解析的视频数（默认 4）；单个视频解析失败立即返回错误。
/// `on_resolve` 每完成一个视频调用一次（已完成数, 总数, 标题），用于推送解析进度。
///
/// 返回解析结果与合集分组信息（`id=season_id`, `title=合集名`），供前端折叠展示。
pub async fn resolve_collection(
    client: &BiliClient,
    mid: &str,
    season_id: &str,
    prefer_format: i64,
    on_resolve: Option<Arc<dyn Fn(usize, usize, &str) + Send + Sync>>,
) -> Result<(Vec<ResolveResult>, TaskGroup)> {
    let (bvids, title) = get_collection_bvids(client, mid, season_id).await?;
    let group = TaskGroup {
        id: season_id.to_string(),
        title: title.unwrap_or_else(|| format!("合集 {}", season_id)),
    };
    let results = resolve_videos_parallel(client, bvids, prefer_format, on_resolve).await?;
    Ok((results, group))
}

/// 并发解析一组 bvid（保持入参顺序返回）
///
/// 通过 `Semaphore` 限制并发数；任一视频解析失败则整体返回该错误。
/// `on_resolve` 每完成一个视频调用一次（已完成数, 总数, 标题）。
///
/// 供命令层在已知 bvid 列表时直接调用（如从 `view` 的 `ugc_season.episodes`
/// 提取合集分集，从而避开 `seasons_archives_list` 的 -352 风控）。
pub async fn resolve_bvids(
    client: &BiliClient,
    bvids: Vec<String>,
    prefer_format: i64,
    on_resolve: Option<Arc<dyn Fn(usize, usize, &str) + Send + Sync>>,
) -> Result<Vec<ResolveResult>> {
    resolve_videos_parallel(client, bvids, prefer_format, on_resolve).await
}

/// 并发解析一组 bvid（保持入参顺序返回）
///
/// 通过 `Semaphore` 限制并发数；单个视频解析失败仅跳过该项（不再整体中断）。
/// `on_resolve` 每完成一个视频调用一次（已完成数, 总数, 标题）。
async fn resolve_videos_parallel(
    client: &BiliClient,
    bvids: Vec<String>,
    prefer_format: i64,
    on_resolve: Option<Arc<dyn Fn(usize, usize, &str) + Send + Sync>>,
) -> Result<Vec<ResolveResult>> {
    if bvids.is_empty() {
        return Ok(Vec::new());
    }
    let total = bvids.len();
    let concurrency = resolve_concurrency(4);
    let sem = Arc::new(Semaphore::new(concurrency));
    let client = Arc::new(client.clone());

    let mut handles = Vec::with_capacity(bvids.len());
    for bvid in bvids {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            let title = bvid.clone();
            let res = resolve_video(&client, &bvid, prefer_format).await;
            drop(permit);
            (res, title)
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    let mut done = 0usize;
    let mut skipped = 0usize;
    for h in handles {
        let (res, title) = h.await.map_err(|e| {
            crate::biliapi::error::BiliApiError::Other(format!("解析任务被取消: {e}"))
        })?;
        done += 1;
        match res {
            Ok(r) => results.push(r),
            Err(e) => {
                skipped += 1;
                println!("[bili] 跳过解析失败视频 {}: {}", title, e);
            }
        }
        if let Some(cb) = &on_resolve {
            cb(done, total, &title);
        }
    }
    if results.is_empty() && skipped > 0 {
        return Err(crate::biliapi::error::BiliApiError::Other(format!(
            "合集内全部 {} 个视频均解析失败",
            skipped
        )));
    }
    if skipped > 0 {
        println!(
            "[bili] 合集解析完成：成功 {}，跳过失败 {}",
            results.len(),
            skipped
        );
    }
    Ok(results)
}

/// 解析番剧/影视：并发遍历 `episodes`，每集直接取播放直链
///
/// 并发数由 `resolve_concurrency` 决定（默认 4，可通过环境变量
/// `BILI_CONCURRENCY` 覆盖）；单个分集失败立即返回错误。
/// `on_resolve` 每完成一集调用一次（已完成数, 总数, 标题），用于推送解析进度。
pub async fn resolve_season(
    client: &BiliClient,
    ssid: &str,
    prefer_format: i64,
    on_resolve: Option<Arc<dyn Fn(usize, usize, &str) + Send + Sync>>,
) -> Result<Vec<ResolveResult>> {
    let season = get_season_info(client, None, Some(ssid)).await?;
    let prefer = types::MediaFormat(prefer_format);

    // 仅保留可解析的分集（有 bvid 且 cid 有效）
    let eps: Vec<_> = season
        .episodes
        .iter()
        .filter(|ep| !ep.bvid.is_empty() && ep.cid != 0)
        .collect();
    if eps.is_empty() {
        return Ok(Vec::new());
    }
    let total = eps.len();

    let concurrency = resolve_concurrency(4);
    let sem = Arc::new(Semaphore::new(concurrency));
    let client = Arc::new(client.clone());

    let mut handles = Vec::with_capacity(eps.len());
    for ep in eps {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let ep_bvid = ep.bvid.clone();
        let ep_cid = ep.cid;
        let ep_title = ep.title.clone();
        let ep_long = ep.long_title.clone();
        let ep_cover = ep.cover.clone();
        handles.push(tokio::spawn(async move {
            let title = ep_title.clone();
            let res = resolve_episode(
                &client,
                &ep_bvid,
                ep_cid,
                prefer,
                &ep_title,
                &ep_long,
                &ep_cover,
            )
            .await;
            drop(permit);
            (res, title)
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    let mut done = 0usize;
    for h in handles {
        let (res, title) = h.await.map_err(|e| {
            crate::biliapi::error::BiliApiError::Other(format!("解析任务被取消: {e}"))
        })?;
        let res = res?;
        done += 1;
        if let Some(cb) = &on_resolve {
            cb(done, total, &title);
        }
        results.push(res);
    }
    Ok(results)
}

/// 解析单集番剧/影视（内部辅助，供 `resolve_season` 并发调用）
async fn resolve_episode(
    client: &BiliClient,
    bvid: &str,
    cid: i64,
    prefer: types::MediaFormat,
    ep_title: &str,
    ep_long: &str,
    ep_cover: &str,
) -> Result<ResolveResult> {
    let mut actual = prefer;
    let mut selection = None;
    for fmt in prefer.fallback_chain() {
        let play = get_play_info(client, bvid, cid, fmt.0).await?;
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
    Ok(ResolveResult {
        bvid: bvid.to_string(),
        title: if ep_long.is_empty() {
            ep_title.to_string()
        } else {
            ep_long.to_string()
        },
        pages: vec![PageStream {
            page: 1,
            part: ep_title.to_string(),
            video_url: sel.video_url,
            audio_url: sel.audio_url,
            actual_format: actual.0,
        }],
        cover: ep_cover.to_string(),
    })
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
            cover: String::new(),
        };
        assert_eq!(r.pages.len(), 2);
        // 第二分 P 视频直链缺失（清晰度降级后仍无视频流），但音频仍在
        assert!(r.pages[1].video_url.is_none());
        assert!(r.pages[1].audio_url.is_some());
    }
}
