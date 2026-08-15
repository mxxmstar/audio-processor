//! biliapi 模块错误类型

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BiliApiError {
    #[error("HTTP 请求失败: {0}")]
    Http(#[from] crate::http_client::error::HttpClientError),

    #[error("JSON 解析失败: {0}")]
    Json(#[from] serde_json::Error),

    #[error("B 站接口返回错误码 code={code}, message={message}")]
    Api { code: i64, message: String },

    #[error("WBI 签名所需 SESSDATA 为空")]
    EmptySessdata,

    #[error("未找到 Cookie: {0}")]
    CookieNotFound(String),

    #[error("二维码状态: {0}")]
    QrStatus(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, BiliApiError>;
