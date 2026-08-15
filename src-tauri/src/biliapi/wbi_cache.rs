//! WBI key 拉取与内存缓存
//!
//! 对齐 Go 版 `client.go` 的 `getWbiKeyRemote` + 文档第 5 章 5.2-8 的"WBI key 缓存"建议。
//! 通过 `x/web-interface/nav` 取 img/sub key，拼接后作为 mixinKey 来源。
//! 缓存有效期 24 小时（与 Go 版策略一致）。

use crate::biliapi::error::{BiliApiError, Result};
use crate::biliapi::storage;
use crate::biliapi::types::{BaseRes, NavData};
use crate::http_client::client::HttpClient;
use crate::http_client::types::{HttpMethod, RequestConfig};
use std::path::Path;
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

/// 落盘目录（可由 Tauri 层在启动时通过 `set_cache_dir` 注入）。
/// 为 `None` 时自动探测（见 `storage::resolve_dir`）。
static CACHE_DIR: Mutex<Option<Option<std::path::PathBuf>>> = Mutex::new(None);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 设置落盘目录（Tauri 启动时调用，传入 `app_config_dir`）。
/// 传 `None` 恢复自动探测。
pub fn set_cache_dir(dir: Option<&Path>) {
    *CACHE_DIR.lock().unwrap() = Some(dir.map(|p| p.to_path_buf()));
}

/// 取当前生效的落盘目录（已显式设置则用之，否则 `None` 走自动探测）。
fn cache_dir() -> Option<Option<std::path::PathBuf>> {
    CACHE_DIR.lock().unwrap().clone()
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

/// 获取 mixinKey，优先内存缓存 → 落盘缓存 → 重新拉取（24h 内有效）。
pub async fn get_mixin_key(sessdata: &str) -> Result<String> {
    if sessdata.is_empty() {
        return Err(BiliApiError::EmptySessdata);
    }

    // 1) 内存缓存命中
    {
        let guard = WBI_CACHE.lock().unwrap();
        if let Some(c) = guard.as_ref() {
            if now_secs().saturating_sub(c.fetched_at) < CACHE_TTL_SECS {
                return Ok(c.key.clone());
            }
        }
    }

    // 2) 落盘缓存命中（阶段 4.2）
    let dir = cache_dir();
    if let Some(key) = storage::load_wbi(dir.as_ref().map(|o| o.as_deref()).flatten()) {
        let mut guard = WBI_CACHE.lock().unwrap();
        *guard = Some(Cache {
            key: key.clone(),
            fetched_at: now_secs(),
        });
        return Ok(key);
    }

    // 3) 未命中或已过期：重新拉取并写回内存 + 落盘
    let key = fetch_mixin_key(sessdata).await?;
    {
        let mut guard = WBI_CACHE.lock().unwrap();
        *guard = Some(Cache {
            key: key.clone(),
            fetched_at: now_secs(),
        });
    }
    storage::save_wbi(dir.as_ref().map(|o| o.as_deref()).flatten(), &key);
    Ok(key)
}

/// 清除缓存（测试或登录态变更时调用）
pub fn clear_cache() {
    *WBI_CACHE.lock().unwrap() = None;
    storage::clear_wbi(cache_dir().as_ref().map(|o| o.as_deref()).flatten());
}
