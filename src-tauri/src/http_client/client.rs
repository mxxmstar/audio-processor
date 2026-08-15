use reqwest::Client as ReqwestClient;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::error::HttpClientError;
use super::types::{CallbackInfo, HttpMethod, RequestConfig, HttpResponse};

/// HTTP 客户端
///
/// 对 `reqwest::Client` 的轻量封装，提供简洁的请求接口。
/// 内部维护连接池，支持多个请求复用 TCP 连接，提高性能。
///
/// # 示例
/// ```ignore
/// use crate::http_client::HttpClient;
///
/// let client = HttpClient::new();
/// let resp = client.get("https://api.example.com/users").await?;
/// println!("状态码: {}", resp.status);
/// println!("响应体: {}", resp.body);
/// ```
#[derive(Clone)]
pub struct HttpClient {
    /// 内部的 reqwest 客户端
    inner: ReqwestClient,
    /// 请求完成后的回调函数（可选）
    on_complete: Option<Arc<dyn Fn(CallbackInfo) + Send + Sync>>,
}

impl HttpClient {
    /// 创建新的 HTTP 客户端
    ///
    /// 使用默认配置创建客户端，默认启用：
    /// - 连接池（最大空闲连接数 128）
    /// - 自动重定向（最多 10 次）
    /// - 默认超时 30 秒
    pub fn new() -> Self {
        let inner = ReqwestClient::builder()
            .user_agent("Mozilla/5.0") // 默认 UA，B 站 API 要求（可被 header() 覆盖）
            .no_proxy() // 强制直连，避免 HTTP_PROXY 等环境变量导致走代理
            .timeout(Duration::from_secs(30))
            .build()
            .expect("创建 HTTP 客户端失败");

        Self {
            inner,
            on_complete: None,
        }
    }

    /// 创建支持自定义配置的 HTTP 客户端
    ///
    /// 当需要精细控制客户端行为时使用此方法创建。
    ///
    /// # 参数
    /// * `builder` - 已配置好的 `reqwest::ClientBuilder`
    pub fn with_builder(builder: reqwest::ClientBuilder) -> Result<Self, HttpClientError> {
        let inner = builder
            .build()
            .map_err(|e| HttpClientError::RequestBuildError(e.to_string()))?;

        Ok(Self {
            inner,
            on_complete: None,
        })
    }

    /// 注册请求完成回调
    ///
    /// 每次请求完成后会调用此回调，传入请求的详细信息。
    /// 可用于日志记录、性能监控等场景。
    ///
    /// # 参数
    /// * `callback` - 回调函数，接收 `CallbackInfo`
    pub fn on_complete<F>(mut self, callback: F) -> Self
    where
        F: Fn(CallbackInfo) + Send + Sync + 'static,
    {
        self.on_complete = Some(Arc::new(callback));
        self
    }

    /// 发送 GET 请求
    ///
    /// # 参数
    /// * `url` - 请求地址
    ///
    /// # 返回值
    /// 返回 `HttpResponse` 封装，包含状态码、响应头和响应体
    pub async fn get(&self, url: impl Into<String>) -> Result<HttpResponse, HttpClientError> {
        let config = RequestConfig::new(url).method(HttpMethod::GET);
        self.send(config).await
    }

    /// 发送 POST 请求（JSON 请求体）
    ///
    /// # 参数
    /// * `url` - 请求地址
    /// * `body` - JSON 请求体
    ///
    /// # 示例
    /// ```ignore
    /// let body = serde_json::json!({
    ///     "name": "张三",
    ///     "email": "zhangsan@example.com"
    /// });
    /// let resp = client.post_json("https://api.example.com/users", body).await?;
    /// ```
    pub async fn post_json(
        &self,
        url: impl Into<String>,
        body: serde_json::Value,
    ) -> Result<HttpResponse, HttpClientError> {
        let config = RequestConfig::new(url)
            .method(HttpMethod::POST)
            .header("Content-Type", "application/json")
            .json(body);
        self.send(config).await
    }

    /// 发送 PUT 请求（JSON 请求体）
    ///
    /// # 参数
    /// * `url` - 请求地址
    /// * `body` - JSON 请求体
    pub async fn put_json(
        &self,
        url: impl Into<String>,
        body: serde_json::Value,
    ) -> Result<HttpResponse, HttpClientError> {
        let config = RequestConfig::new(url)
            .method(HttpMethod::PUT)
            .header("Content-Type", "application/json")
            .json(body);
        self.send(config).await
    }

    /// 发送 DELETE 请求
    ///
    /// # 参数
    /// * `url` - 请求地址
    pub async fn delete(&self, url: impl Into<String>) -> Result<HttpResponse, HttpClientError> {
        let config = RequestConfig::new(url).method(HttpMethod::DELETE);
        self.send(config).await
    }

    /// 发送通用 HTTP 请求（核心方法）
    ///
    /// 根据 `RequestConfig` 中的配置构造并发送请求。
    /// 此方法是所有公开请求方法的底层实现。
    ///
    /// # 参数
    /// * `config` - 请求配置，包含 URL、方法、请求头、查询参数和请求体等
    ///
    /// # 处理流程
    /// 1. 解析 URL
    /// 2. 构造请求（设置方法、请求头、查询参数、请求体、超时）
    /// 3. 发送请求并等待响应
    /// 4. 检查状态码，若非成功状态码则返回错误
    /// 5. 读取响应体
    /// 6. 封装响应并触发回调
    pub async fn send(&self, config: RequestConfig) -> Result<HttpResponse, HttpClientError> {
        let start = Instant::now();
        let url_str = config.url.clone();

        // 1. 构造请求
        let mut req = self
            .inner
            .request(
                convert_method(&config.method),
                &url_str,
            )
            .timeout(std::time::Duration::from_secs(config.timeout_secs));

        // 2. 添加请求头
        for (key, value) in &config.headers {
            req = req.header(key.as_str(), value.as_str());
        }

        // 3. 添加查询参数
        if !config.query.is_empty() {
            req = req.query(&config.query);
        }

        // 4. 添加请求体
        if let Some(body) = &config.body {
            req = req.json(body);
        }

        // 5. 发送请求
        println!("HTTP {} -> {}", config.method.as_str(), url_str);

        let response = req.send().await.map_err(|e| {
            let err = HttpClientError::from(e);
            println!("HTTP 请求失败: {}", err);
            err
        })?;

        // 6. 获取状态码
        let status = response.status().as_u16();

        // 7. 获取响应头
        let headers: std::collections::HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    v.to_str().unwrap_or("").to_string(),
                )
            })
            .collect();

        // 8. 读取响应体
        let body = response.text().await.map_err(|e| {
            HttpClientError::ResponseParseError(format!("读取响应体失败: {}", e))
        })?;

        let duration = start.elapsed();
        let duration_ms = duration.as_millis() as u64;

        // 9. 构造统一响应
        let http_response = HttpResponse {
            status,
            headers,
            body: body.clone(),
        };

        // 10. 记录日志
        println!(
            "HTTP {} {} -> {} ({}ms, {} bytes)",
            config.method.as_str(),
            url_str,
            status,
            duration_ms,
            body.len()
        );

        // 11. 触发回调
        if let Some(ref callback) = self.on_complete {
            let info = CallbackInfo {
                url: url_str,
                method: config.method.as_str().to_string(),
                status,
                duration_ms,
                body_size: body.len(),
            };
            (callback)(info);
        }

        Ok(http_response)
    }

    /// 发送请求并断言成功（200-299）
    ///
    /// 如果服务端返回非成功状态码，直接返回错误。
    /// 适合在期望请求一定成功的场景使用。
    ///
    /// # 参数
    /// * `config` - 请求配置
    pub async fn send_expect_success(
        &self,
        config: RequestConfig,
    ) -> Result<HttpResponse, HttpClientError> {
        let resp = self.send(config).await?;
        if !resp.is_success() {
            return Err(HttpClientError::HttpStatusError {
                status: resp.status,
                body: resp.body,
            });
        }
        Ok(resp)
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

/// 将内部 `HttpMethod` 转换为 reqwest 的请求方法
fn convert_method(method: &HttpMethod) -> reqwest::Method {
    match method {
        HttpMethod::GET => reqwest::Method::GET,
        HttpMethod::POST => reqwest::Method::POST,
        HttpMethod::PUT => reqwest::Method::PUT,
        HttpMethod::DELETE => reqwest::Method::DELETE,
        HttpMethod::PATCH => reqwest::Method::PATCH,
        HttpMethod::HEAD => reqwest::Method::HEAD,
        HttpMethod::OPTIONS => reqwest::Method::OPTIONS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_config() {
        let client = HttpClient::new();
        
        let url = "http://192.168.66.83:8080/config";
        println!("\n========== 发送 getConfig 请求 ==========");
        println!("请求地址: {}", url);
        
        match client.get(url).await {
            Ok(response) => {
                println!("\n========== 请求成功 ==========");
                println!("状态码: {}", response.status);
                println!("响应头:");
                for (key, value) in &response.headers {
                    println!("  {}: {}", key, value);
                }
                // println!("\n响应体:");
                // println!("{}", response.body);
                println!("\n======================================\n");
            }
            Err(e) => {
                println!("\n========== 请求失败 ==========");
                println!("错误信息: {}", e);
                println!("\n======================================\n");
            }
        }
    }
}