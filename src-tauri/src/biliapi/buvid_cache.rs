//! buvid3 / buvid4 指纹 Cookie 拉取与缓存
//!
//! B 站部分接口（`web-space` 系，如 `seasons_archives_list`）会做更严格的风控，
//! 请求若不携带 `buvid3`/`buvid4` 指纹 Cookie，会直接返回业务码 -352（请求被拦截）。
//! 普通已登录接口（view / playurl）对此较宽松，因此单视频能下、合集却 -352。
//!
//! 这里在进程内缓存一次指纹（24h），供 `BiliClient::get` 注入到 Cookie 头。

use crate::biliapi::error::Result;
use crate::http_client::client::HttpClient;
use crate::http_client::types::{HttpMethod, RequestConfig};
use serde::Deserialize;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const SPI_URL: &str = "https://api.bilibili.com/x/frontend/finger/spi";
const CACHE_TTL_SECS: u64 = 24 * 3600;

#[derive(Default)]
struct Buvid {
    b3: String,
    b4: String,
    fetched_at: u64,
}

/// 全局 buvid 缓存（进程内单例）
static BUVID_CACHE: Mutex<Option<Buvid>> = Mutex::new(None);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Deserialize, Default)]
struct SpiData {
    #[serde(default)]
    buvid3: String,
    #[serde(default)]
    buvid4: String,
}

#[derive(Debug, Deserialize)]
struct SpiRes {
    #[serde(default)]
    code: i64,
    #[serde(default)]
    data: SpiData,
}

/// 从 `finger/spi` 拉取指纹（无需登录）。
async fn fetch_buvid() -> Result<(String, String)> {
    let client = HttpClient::new();
    let config = RequestConfig::new(SPI_URL)
        .method(HttpMethod::GET)
        .header("Referer", "https://www.bilibili.com");
    let resp = client.send_expect_success(config).await?;
    let base: SpiRes = resp.json::<SpiRes>()?;
    if base.code != 0 {
        return Err(crate::biliapi::error::BiliApiError::Other(format!(
            "finger/spi 返回非零码 {}",
            base.code
        )));
    }
    if base.data.buvid3.is_empty() {
        return Err(crate::biliapi::error::BiliApiError::Other(
            "finger/spi 未返回 buvid3".into(),
        ));
    }
    Ok((base.data.buvid3, base.data.buvid4))
}

/// 确保 buvid 已就绪（内存缓存命中且未过期则跳过；否则重新拉取并缓存）。
/// 失败不致命：调用方忽略结果即可，最坏退化为不带 buvid（与改造前一致）。
pub async fn ensure_buvid() -> Result<()> {
    {
        let guard = BUVID_CACHE.lock().unwrap();
        if let Some(c) = guard.as_ref() {
            if !c.b3.is_empty() && now_secs().saturating_sub(c.fetched_at) < CACHE_TTL_SECS {
                return Ok(());
            }
        }
    }
    let (b3, b4) = fetch_buvid().await?;
    let mut guard = BUVID_CACHE.lock().unwrap();
    *guard = Some(Buvid {
        b3: b3.clone(),
        b4: b4.clone(),
        fetched_at: now_secs(),
    });
    println!(
        "[bili] 已获取指纹 buvid3={}{}",
        &b3.chars().take(8).collect::<String>(),
        if b4.is_empty() { "" } else { " buvid4=*" }
    );
    Ok(())
}

/// 读取已缓存的 buvid Cookie 串（`buvid3=...; buvid4=...`），未就绪时返回空串。
pub fn buvid_cookie() -> String {
    let guard = BUVID_CACHE.lock().unwrap();
    match guard.as_ref() {
        Some(c) if !c.b3.is_empty() => {
            if c.b4.is_empty() {
                format!("buvid3={}", c.b3)
            } else {
                format!("buvid3={}; buvid4={}", c.b3, c.b4)
            }
        }
        _ => String::new(),
    }
}
