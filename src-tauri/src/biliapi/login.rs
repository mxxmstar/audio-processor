//! 登录相关 B 站 API 封装
//! 对应 Go 版 `bilidownload/server/bilibili/client.go` 的登录逻辑。

use crate::biliapi::client::{BiliClient, BASE_API, BASE_PASSPORT};
use crate::biliapi::error::{BiliApiError, Result};
use crate::biliapi::storage;
use crate::biliapi::types;
use crate::http_client::client::HttpClient;
use crate::http_client::types::{HttpMethod, RequestConfig};
use std::path::Path;

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

/// 将二维码登录 URL 编码为 SVG 字符串（前端可直接 `<img src="data:image/svg+xml,...">` 渲染）。
/// B 站 `qrcode/generate` 返回的 `url` 是扫码跳转链接，本身不是图片，
/// 必须由本端生成二维码图像。
pub fn generate_qr_svg(url: &str) -> Result<String> {
    use qrcode::render::svg;
    use qrcode::QrCode;
    let code = QrCode::new(url).map_err(|e| BiliApiError::Other(format!("生成二维码失败: {}", e)))?;
    let svg = code
        .render()
        .min_dimensions(240, 240)
        .dark_color(svg::Color("#000000"))
        .light_color(svg::Color("#ffffff"))
        .build();
    Ok(svg)
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
        // 从响应 Cookie（多值）提取 SESSDATA。
        // 注意：单值 `headers["set-cookie"]` 会丢失多值，必须用 `cookies` 列表。
        resp.cookies
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("SESSDATA"))
            .map(|(_, v)| v.clone())
    } else {
        None
    };
    Ok((status, sessdata))
}

/// 扫码成功后持久化 SESSDATA（阶段 4.3）。
/// 传入从 `get_qr_status` 拿到的 SESSDATA；为空则忽略。
/// 同时清掉可能已失效的 WBI 落盘缓存，下次请求强制刷新。
pub fn persist_login(dir: Option<&Path>, sessdata: Option<&str>) {
    match sessdata {
        Some(s) if !s.is_empty() => {
            storage::save_sessdata(dir, Some(s));
            storage::clear_wbi(dir);
            crate::biliapi::wbi_cache::clear_cache();
        }
        _ => {}
    }
}

/// 检查指定 SESSDATA 是否有效（对应 `x/space/myinfo`）。
/// 返回 `Ok(true)` 有效 / `Ok(false)` 失效（业务错误码视为未登录）；
/// 网络等基础设施错误向上抛出。
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

/// 获取已登录用户的昵称与头像（对应 `x/space/myinfo` 的 `data.name` / `data.face`）。
pub struct UserInfo {
    pub name: String,
    pub face: String,
}

pub async fn fetch_user_info(sessdata: &str) -> Result<UserInfo> {
    let client = BiliClient::new(sessdata);
    let cfg = client.get(&format!("{}/x/space/myinfo", BASE_API));
    let v: serde_json::Value = client.send_json::<serde_json::Value>(cfg).await?;
    // 注意：B 站 `x/space/myinfo` 把用户字段直接放在根对象（无 `data` 包裹），
    // 而 `x/web-interface/wbi/view` 等接口才用 `data` 包裹，需区分对待。
    let name = v
        .get("name")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    let face = v
        .get("face")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    // B 站返回的头像多为 http://，桌面 webview 会按混合内容策略拦截，
    // 统一升级为 https:// 以保证正常加载。
    let face = face.replacen("http://", "https://", 1);
    Ok(UserInfo { name, face })
}

/// 启动时自动加载持久化 SESSDATA 并校验（阶段 4.3）。
/// 返回 `Ok(Some(sessdata))` 表示已有有效登录态；`Ok(None)` 表示需重新扫码。
/// 加载或校验过程中任何落盘/网络故障都降级为 `None`（不致命）。
pub async fn load_and_check(dir: Option<&Path>) -> Result<Option<String>> {
    let sessdata = match storage::load_sessdata(dir) {
        Some(s) => s,
        None => return Ok(None),
    };
    if check_login(&sessdata).await? {
        Ok(Some(sessdata))
    } else {
        // 登录态失效：清除落盘，提示重新扫码
        storage::clear_sessdata(dir);
        storage::clear_wbi(dir);
        Ok(None)
    }
}

/// 高层封装：确保存在有效登录态。
/// - 若已有有效持久化 SESSDATA，直接返回它（无需扫码）；
/// - 否则返回 `None`，调用方应发起扫码登录流程，
///   并在 `get_qr_status` 成功后调用 `persist_login` 落盘。
pub async fn ensure_login(dir: Option<&Path>) -> Result<Option<String>> {
    load_and_check(dir).await
}
