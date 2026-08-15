//! BiliClient：封装向 B 站发送请求所需的上下文
//!
//! 持有 `HttpClient` 与登录态 `SESSDATA`，并提供统一的请求构造器。
//! 不直接依赖数据库——SESSDATA 由调用方通过 `new(sessdata)` 注入，
//! 便于与主项目（audio-processor）的存储层解耦。

use crate::biliapi::error::Result;
use crate::biliapi::wbi_cache;
use crate::http_client::client::HttpClient;
use crate::http_client::types::{HttpMethod, RequestConfig};

pub const BASE_API: &str = "https://api.bilibili.com";
pub const BASE_PASSPORT: &str = "https://passport.bilibili.com";

#[derive(Clone)]
pub struct BiliClient {
    pub sessdata: String,
    client: HttpClient,
}

impl BiliClient {
    /// 使用指定登录态创建客户端。SESSDATA 为空时仍可构造，
    /// 但调用需要登录的接口（如 WBI 签名）会返回 `EmptySessdata`。
    pub fn new(sessdata: impl Into<String>) -> Self {
        Self {
            sessdata: sessdata.into(),
            client: HttpClient::new(),
        }
    }

    /// 构造带标准请求头的 GET 配置（含 SESSDATA Cookie + Referer）。
    /// 对齐 Go 版 `MakeHeader`（`Mozilla/5.0` UA 由 http_client 默认提供）。
    pub fn get(&self, url: &str) -> RequestConfig {
        let mut cfg = RequestConfig::new(url).method(HttpMethod::GET);
        if !self.sessdata.is_empty() {
            cfg = cfg.header("Cookie", format!("SESSDATA={}", self.sessdata));
        }
        cfg.header("Referer", "https://www.bilibili.com")
    }

    /// 对请求施加 WBI 签名（自动拉取并缓存 mixinKey）。
    /// 需在已设置好业务 query 参数之后调用。
    pub async fn with_wbi(&self, mut cfg: RequestConfig) -> Result<RequestConfig> {
        let mixin_key = wbi_cache::get_mixin_key(&self.sessdata).await?;
        cfg = cfg.wbi_sign(&mixin_key);
        Ok(cfg)
    }

    /// 发送请求并将响应解析为 `BaseRes<T>`，校验 code==0。
    pub async fn send_json<T: serde::de::DeserializeOwned>(&self, cfg: RequestConfig) -> Result<T> {
        let resp: crate::http_client::types::HttpResponse = self.client.send_expect_success(cfg).await?;
        let base: crate::biliapi::types::BaseRes<T> = resp.json::<crate::biliapi::types::BaseRes<T>>()?;
        base.into_result()
    }
}
