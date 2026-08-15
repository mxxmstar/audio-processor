//! 登录相关 B 站 API 封装
//! 对应 Go 版 `bilidownload/server/bilibili/client.go` 的登录逻辑。

use crate::biliapi::client::{BiliClient, BASE_API, BASE_PASSPORT};
use crate::biliapi::error::{BiliApiError, Result};
use crate::biliapi::types;
use crate::http_client::client::HttpClient;
use crate::http_client::types::{HttpMethod, RequestConfig};

/// 获取登录二维码信息
/// 对应 `passport.bilibili.com/x/passport-login/web/qrcode/generate`
pub async fn new_qr_info() -> Result<types::QrInfo> {
    let client = HttpClient::new();
    let cfg = RequestConfig::new(&format!(
        "{}/x/passport-login/web/qrcode/generate",
        BASE_PASSPORT
    ))
    .method(HttpMethod::GET)
    .header("Referer", "https://www.bilibili.com");
    let resp: crate::http_client::types::HttpResponse = client.send_expect_success(cfg).await?;
    let base: types::BaseRes<types::QrInfo> = resp.json::<types::BaseRes<types::QrInfo>>()?;
    base.into_result()
}

/// 轮询二维码状态，登录成功时返回 SESSDATA
/// 对应 `passport.bilibili.com/x/passport-login/web/qrcode/poll`
/// 返回 (状态, 可选 SESSDATA)
pub async fn get_qr_status(qr_key: &str) -> Result<(types::QrStatus, Option<String>)> {
    let client = HttpClient::new();
    let cfg = RequestConfig::new(&format!("{}/x/passport-login/web/qrcode/poll", BASE_PASSPORT))
        .method(HttpMethod::GET)
        .query("qrcode_key", qr_key.to_string())
        .header("Referer", "https://www.bilibili.com");
    let resp: crate::http_client::types::HttpResponse = client.send_expect_success(cfg).await?;
    let base: types::BaseRes<types::QrStatus> = resp.json::<types::BaseRes<types::QrStatus>>()?;
    let status = base.into_result()?;

    let sessdata = if status.code == types::QR_SUCCESS {
        // 从响应头 Set-Cookie 提取 SESSDATA（对齐 Go 版 GetCookieValue）
        resp.headers
            .get("set-cookie")
            .and_then(|v| extract_sessdata(v))
    } else {
        None
    };
    Ok((status, sessdata))
}

/// 从 Set-Cookie 头值中提取 SESSDATA
/// 头可能包含多个 cookie，形如 `SESSDATA=xxx; Expires=...`
fn extract_sessdata(header: &str) -> Option<String> {
    for part in header.split(';') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("SESSDATA=") {
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    None
}

/// 检查是否已登录
/// 对应 `x/space/myinfo`
pub async fn check_login(sessdata: &str) -> Result<bool> {
    if sessdata.is_empty() {
        return Ok(false);
    }
    let client = BiliClient::new(sessdata);
    let cfg = client.get(&format!("{}/x/space/myinfo", BASE_API));
    match client.send_json::<serde_json::Value>(cfg).await {
        Ok(_) => Ok(true),
        Err(BiliApiError::Api { .. }) => Ok(false),
        Err(e) => Err(e),
    }
}
