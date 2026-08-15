use reqwest;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HttpClientError {
    #[error("请求构建失败: {0}")]
    RequestBuildError(String),

    #[error("URL解析失败: {0}")]
    UrlParseError(String),

    #[error("网络请求失败: {0}")]
    RequestError(String),

    /// 服务端回包解析失败
    #[error("响应解析失败: {0}")]
    ResponseParseError(String),

    #[error("HTTP状态码错误: {status}: {body}")]
    HttpStatusError {
        status: u16,
        body: String,
    },

    #[error("超时错误: {0}")]
    TimeoutError(String),


    #[error("TLS错误: {0}")]
    TlsError(String),

    #[error("其他错误: {0}")]
    OtherError(String),

}

impl From<reqwest::Error> for HttpClientError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            HttpClientError::TimeoutError(e.to_string())
        } else if e.is_builder() {
            HttpClientError::RequestBuildError(e.to_string())
        } else if e.is_connect() {
            HttpClientError::RequestError(format!("连接失败: {}", e))
        } else if e.is_decode() {
            HttpClientError::ResponseParseError(e.to_string())
        } else {
            HttpClientError::RequestError(e.to_string())
        }
    }
}