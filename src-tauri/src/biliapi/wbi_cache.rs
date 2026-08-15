//! WBI key 拉取与内存缓存
//!
//! 对齐 Go 版 `client.go` 的 `getWbiKeyRemote` + 文档第 5 章 5.2-8 的"WBI key 缓存"建议。
//! 通过 `x/web-interface/nav` 取 img/sub key，拼接后作为 mixinKey 来源。
//! 缓存有效期 24 小时（与 Go 版策略一致）。

use crate::biliapi::error::{BiliApiError, Result};
use crate::biliapi::types::{BaseRes, NavData};
use crate::http_client::client::HttpClient;
use crate::http_client::types::{HttpMethod, RequestConfig};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const NAV_URL: &str = "https://api.bilibili.com/x/web-interface/nav";
const CACHE_TTL_SECS: u64 = 24 * 3600;

struct Cache {
    key: String,
    fetched_at: u64, // 秒级时间戳
}

/// 全局 WBI key 缓存（进程内单例）
static WBI_CACHE: Mutex<Option<Cache>> = Mutex::new(None);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 从 nav 接口获取拼接后的 wbi key（`imgKey + subKey`）并算出 mixinKey。
async fn fetch_mixin_key(sessdata: &str) -> Result<String> {
    let client = HttpClient::new();
    let config = RequestConfig::new(NAV_URL)
        .method(HttpMethod::GET)
        .header("Cookie", format!("SESSDATA={}", sessdata))
        .header("Referer", "https://www.bilibili.com");
    let resp: crate::http_client::types::HttpResponse = client.send_expect_success(config).await?;
    let base: BaseRes<NavData> = resp.json::<BaseRes<NavData>>()?;
    let data = base.into_result()?;

    // 正则提取 /bfs/wbi/{key}. 中的 key（对齐 Go 版 regexp `/bfs/wbi/([a-z0-9]+)\.`）
    let re = regex::Regex::new(r"/bfs/wbi/([a-z0-9]+)\.")
        .map_err(|e| BiliApiError::Other(format!("正则编译失败: {}", e)))?;
    let img_key = re
        .captures(&data.wbi_img.img_url)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .ok_or_else(|| BiliApiError::Other("无法从 img_url 提取 wbi key".into()))?;
    let sub_key = re
        .captures(&data.wbi_img.sub_url)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .ok_or_else(|| BiliApiError::Other("无法从 sub_url 提取 wbi key".into()))?;

    Ok(crate::http_client::wbi::get_mixin_key(&img_key, &sub_key))
}

/// 获取 mixinKey，优先使用缓存（24h 内有效）。
pub async fn get_mixin_key(sessdata: &str) -> Result<String> {
    if sessdata.is_empty() {
        return Err(BiliApiError::EmptySessdata);
    }

    // 命中缓存且未过期
    {
        let guard = WBI_CACHE.lock().unwrap();
        if let Some(c) = guard.as_ref() {
            if now_secs().saturating_sub(c.fetched_at) < CACHE_TTL_SECS {
                return Ok(c.key.clone());
            }
        }
    }

    // 未命中或已过期：重新拉取
    let key = fetch_mixin_key(sessdata).await?;
    {
        let mut guard = WBI_CACHE.lock().unwrap();
        *guard = Some(Cache {
            key: key.clone(),
            fetched_at: now_secs(),
        });
    }
    Ok(key)
}

/// 清除缓存（测试或登录态变更时调用）
pub fn clear_cache() {
    *WBI_CACHE.lock().unwrap() = None;
}
