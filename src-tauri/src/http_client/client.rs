use reqwest::Client as ReqwestClient;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::error::HttpClientError;
use super::types::{CallbackInfo, HttpMethod, Progress, RequestConfig, HttpResponse};

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
    /// 内置重试：当 `config.retry > 0` 且发生**可重试错误**（超时 / 连接失败 / 服务端 5xx）
    /// 时，按指数退避重试。客户端 4xx 业务错误不会重试。
    ///
    /// # 参数
    /// * `config` - 请求配置，包含 URL、方法、请求头、查询参数和请求体等
    pub async fn send(&self, config: RequestConfig) -> Result<HttpResponse, HttpClientError> {
        let max_retry = config.retry;
        let mut attempt: u32 = 0;
        loop {
            match self.send_once(&config).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    let retryable = matches!(
                        e,
                        HttpClientError::TimeoutError(_)
                            | HttpClientError::RequestError(_)
                            | HttpClientError::TlsError(_)
                    ) || matches!(e, HttpClientError::HttpStatusError { status, .. } if status >= 500);
                    if attempt < max_retry && retryable {
                        attempt += 1;
                        // 指数退避：2^attempt 秒（上限 30s）
                        let backoff = std::time::Duration::from_secs(2u64.pow(attempt).min(30));
                        println!(
                            "HTTP 请求可重试失败 (第{}次)，{}s 后重试: {}",
                            attempt,
                            backoff.as_secs(),
                            e
                        );
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
    }

    /// 单次发送（不含重试），供 `send` 调用
    async fn send_once(&self, config: &RequestConfig) -> Result<HttpResponse, HttpClientError> {
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

        // 7.1 提取 Set-Cookie（供登录态提取 SESSDATA）
        let cookies: Vec<(String, String)> = response
            .cookies()
            .map(|c| (c.name().to_string(), c.value().to_string()))
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
            cookies,
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

        if !http_response.is_success() {
            return Err(HttpClientError::HttpStatusError {
                status,
                body: body.clone(),
            });
        }

        Ok(http_response)
    }

    /// 流式下载到本地文件（阶段 2）
    ///
    /// 使用 `reqwest` 的 chunked 流边下边写，避免将整个文件读入内存。
    /// 通过 `on_progress` 回调（节流约 200ms）上报下载进度。
    ///
    /// # 参数
    /// * `url` - 直链地址（B 站 DASH 流需带 `Referer`/`User-Agent`，见 `download_with_config`）
    /// * `path` - 目标文件路径
    /// * `on_progress` - 进度回调（可选）
    ///
    /// # 示例
    /// ```ignore
    /// client.download_to_file(&url, "video.m4s", None).await?;
    /// ```
    pub async fn download_to_file(
        &self,
        url: &str,
        path: &str,
        on_progress: Option<Arc<dyn Fn(Progress) + Send + Sync>>,
    ) -> Result<(), HttpClientError> {
        self.download_with_config(
            RequestConfig::new(url).header("Referer", "https://www.bilibili.com"),
            path,
            on_progress,
        )
        .await
    }

    /// 带自定义请求配置的流式下载（可附加 header / 重试 / Range 断点续传）
    pub async fn download_with_config(
        &self,
        config: RequestConfig,
        path: &str,
        on_progress: Option<Arc<dyn Fn(Progress) + Send + Sync>>,
    ) -> Result<(), HttpClientError> {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        // 若文件已存在，支持 Range 断点续传（从已下载字节之后开始）
        let resume_from = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(resume_from == 0)
            .open(path)
            .await
            .map_err(|e| HttpClientError::OtherError(format!("打开文件失败 {}: {}", path, e)))?;
        if resume_from > 0 {
            use tokio::io::AsyncSeekExt;
            file.seek(std::io::SeekFrom::Start(resume_from))
                .await
                .map_err(|e| HttpClientError::OtherError(format!("seek 失败: {}", e)))?;
        }

        let mut req = self
            .inner
            .request(convert_method(&config.method), &config.url)
            .timeout(std::time::Duration::from_secs(config.timeout_secs));
        for (k, v) in &config.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        if !config.query.is_empty() {
            req = req.query(&config.query);
        }
        if resume_from > 0 {
            req = req.header("Range", format!("bytes={}-", resume_from));
        }

        let resp = req.send().await.map_err(HttpClientError::from)?;
        let status = resp.status().as_u16();
        // 206 Partial Content 或 200 均视为有效
        if status != 200 && status != 206 {
            return Err(HttpClientError::HttpStatusError {
                status,
                body: format!("下载失败，状态码 {}", status),
            });
        }
        let total: Option<u64> = resp
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|len| len + resume_from);

        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = resume_from;
        let mut last_report = Instant::now();
        let mut last_downloaded: u64 = resume_from;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                HttpClientError::ResponseParseError(format!("流读取失败: {}", e))
            })?;
            file.write_all(&chunk)
                .await
                .map_err(|e| HttpClientError::OtherError(format!("写入文件失败: {}", e)))?;
            downloaded += chunk.len() as u64;

            // 节流上报：约 200ms 一次
            let now = Instant::now();
            if let Some(ref cb) = on_progress {
                let elapsed = now.duration_since(last_report).as_millis() as u64;
                if elapsed >= 200 {
                    let speed = if elapsed > 0 {
                        ((downloaded - last_downloaded) as f64 / elapsed as f64 * 1000.0) as u64
                    } else {
                        0
                    };
                    let percent = match total {
                        Some(t) if t > 0 => (downloaded as f64 / t as f64 * 100.0).min(100.0),
                        _ => 0.0,
                    };
                    cb(Progress {
                        downloaded,
                        total,
                        speed,
                        percent,
                    });
                    last_report = now;
                    last_downloaded = downloaded;
                }
            }
        }
        file.flush()
            .await
            .map_err(|e| HttpClientError::OtherError(format!("flush 失败: {}", e)))?;

        if let Some(ref cb) = on_progress {
            cb(Progress {
                downloaded,
                total: Some(downloaded),
                speed: 0,
                percent: 100.0,
            });
        }
        println!("下载完成: {} -> {} ({} bytes)", config.url, path, downloaded);
        Ok(())
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

    /// 启动一个本地 TCP server，返回指定大小的固定内容，用于测试流式下载。
    async fn spawn_file_server(body: Vec<u8>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let _ = sock.read(&mut [0u8; 1024]).await;
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(header.as_bytes()).await;
                let _ = sock.write_all(&body).await;
            }
        });
        format!("http://{}", addr)
    }

    #[tokio::test]
    async fn test_download_to_file_writes_content() {
        let payload: Vec<u8> = (0u8..=255).cycle().take(20_000).collect();
        let url = spawn_file_server(payload.clone()).await;
        let tmp = std::env::temp_dir().join(format!("httpclient_test_{}.bin", std::process::id()));
        let path = tmp.to_string_lossy().to_string();

        let client = HttpClient::new();
        let result = client.download_to_file(&url, &path, None).await;
        assert!(result.is_ok(), "下载应成功: {:?}", result.err());

        let written = std::fs::read(&path).unwrap();
        assert_eq!(written.len(), payload.len());
        assert_eq!(written, payload);

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_download_to_file_progress_called() {
        let payload: Vec<u8> = vec![7u8; 50_000];
        let url = spawn_file_server(payload.clone()).await;
        let tmp = std::env::temp_dir().join(format!("httpclient_test2_{}.bin", std::process::id()));
        let path = tmp.to_string_lossy().to_string();

        let client = HttpClient::new();
        let calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let calls_ref = calls.clone();
        let result = client
            .download_to_file(&url, &path, Some(Arc::new(move |p: Progress| {
                if p.percent > 0.0 {
                    calls_ref.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            })))
            .await;
        assert!(result.is_ok());
        assert!(calls.load(std::sync::atomic::Ordering::SeqCst) >= 1, "进度回调应至少触发一次（结尾 100%）");

        let _ = std::fs::remove_file(&path);
    }
}