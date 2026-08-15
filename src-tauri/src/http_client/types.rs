use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HTTP 请求方法
///
/// 支持 RESTful API 中常用的几种请求方法。
#[derive(Debug, Clone, PartialEq)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

impl HttpMethod {
    /// 将枚举转为字符串表示，用于底层请求构造
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::GET => "GET",
            HttpMethod::POST => "POST",
            HttpMethod::PUT => "PUT",
            HttpMethod::DELETE => "DELETE",
            HttpMethod::PATCH => "PATCH",
            HttpMethod::HEAD => "HEAD",
            HttpMethod::OPTIONS => "OPTIONS",
        }
    }
}

/// HTTP 请求配置
///
/// 用于构建 HTTP 请求的所有参数，采用建造者模式逐步构造。
///
/// # 示例
/// ```ignore
/// let req = RequestConfig::new("https://api.example.com/users")
///     .method(HttpMethod::POST)
///     .header("Authorization", "Bearer token123")
///     .timeout_secs(30);
/// ```
#[derive(Debug, Clone)]
pub struct RequestConfig {
    /// 请求 URL
    pub url: String,
    /// 请求方法
    pub method: HttpMethod,
    /// 请求头（键值对）
    pub headers: HashMap<String, String>,
    /// 查询参数（URL 后面的 ?key=value）
    pub query: HashMap<String, String>,
    /// 请求体（JSON 格式，可选）
    pub body: Option<serde_json::Value>,
    /// 超时时间（秒）
    pub timeout_secs: u64,
}

impl RequestConfig {
    /// 创建一个新的请求配置
    ///
    /// # 参数
    /// * `url` - 目标 URL
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            method: HttpMethod::GET,
            headers: HashMap::new(),
            query: HashMap::new(),
            body: None,
            timeout_secs: 30,
        }
    }

    /// 设置请求方法
    pub fn method(mut self, method: HttpMethod) -> Self {
        self.method = method;
        self
    }

    /// 添加请求头
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// 批量设置请求头
    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    /// 添加查询参数
    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.insert(key.into(), value.into());
        self
    }

    /// 设置 JSON 请求体
    pub fn json(mut self, body: serde_json::Value) -> Self {
        self.body = Some(body);
        self
    }

    /// 设置超时时间（秒）
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

/// HTTP 响应封装
///
/// 统一封装 HTTP 响应，包含状态码、响应头和响应体。
/// 提供便捷方法用于解析不同类型（文本、JSON）的响应体。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponse {
    /// HTTP 状态码
    pub status: u16,
    /// 响应头
    pub headers: HashMap<String, String>,
    /// 响应体（原始文本）
    pub body: String,
}

impl HttpResponse {
    /// 将响应体解析为指定类型的 JSON 对象
    ///
    /// # 泛型参数
    /// * `T` - 目标类型，必须实现 `serde::de::DeserializeOwned`
    ///
    /// # 示例
    /// ```ignore
    /// #[derive(Deserialize)]
    /// struct User { id: u32, name: String }
    ///
    /// let resp: HttpResponse = client.get("...").await?;
    /// let user: User = resp.json().unwrap();
    /// ```
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, crate::http_client::error::HttpClientError> {
        serde_json::from_str(&self.body)
            .map_err(|e| crate::http_client::error::HttpClientError::ResponseParseError(e.to_string()))
    }

    /// 检查响应是否成功（状态码 200-299）
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// 检查响应是否表示客户端错误（状态码 400-499）
    pub fn is_client_error(&self) -> bool {
        self.status >= 400 && self.status < 500
    }

    /// 检查响应是否表示服务端错误（状态码 500-599）
    pub fn is_server_error(&self) -> bool {
        self.status >= 500 && self.status < 600
    }
}

/// 回调信息
///
/// 在 HTTP 请求的各个生命周期阶段触发回调，用于日志记录、进度追踪等。
#[derive(Debug, Clone)]
pub struct CallbackInfo {
    /// 请求的 URL
    pub url: String,
    /// 请求方法
    pub method: String,
    /// HTTP 状态码
    pub status: u16,
    /// 请求耗时（毫秒）
    pub duration_ms: u64,
    /// 响应体大小（字节）
    pub body_size: usize,
}

