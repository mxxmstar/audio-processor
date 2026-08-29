use std::collections::HashMap;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::Client;
use serde_json::json;

use super::types::*;

/// aria2 RPC 客户端
pub struct Aria2Client {
    base_url: String,
    secret: String,
    http_client: Client,
}

impl Aria2Client {
    pub fn new(port: u16, secret: &str) -> Self {
        Aria2Client {
            base_url: format!("http://localhost:{}", port),
            secret: format!("token:{}", secret),
            http_client: Client::new(),
        }
    }

    /// 发送 JSON-RPC 请求
    async fn send_request<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<T, String> {
        let request = JsonRpcRequest::new(method, params);
        let response = self
            .http_client
            .post(&self.base_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("请求失败: {}", e))?;

        let body = response
            .text()
            .await
            .map_err(|e| format!("读取响应失败: {}", e))?;

        let rpc_response: JsonRpcResponse<T> = serde_json::from_str(&body)
            .map_err(|e| format!("解析响应失败: {}, body: {}", e, body))?;

        if let Some(error) = rpc_response.error {
            return Err(format!("aria2 错误: {}", error.message));
        }

        rpc_response.result.ok_or("响应结果为空".to_string())
    }

    /// 获取 aria2 版本
    pub async fn get_version(&self) -> Result<String, String> {
        let version: Aria2Version = self.send_request("aria2.getVersion", vec![]).await?;
        Ok(version.version)
    }

    /// 添加 URI 下载任务
    pub async fn add_uri(&self, urls: Vec<String>, options: Option<HashMap<String, String>>) -> Result<String, String> {
        let mut params: Vec<serde_json::Value> = vec![json!(self.secret), json!(urls)];

        if let Some(opts) = options {
            let opts_json: serde_json::Value = serde_json::to_value(opts)
                .map_err(|e| format!("序列化选项失败: {}", e))?;
            params.push(opts_json);
        }

        let gid: String = self.send_request("aria2.addUri", params).await?;
        Ok(gid)
    }

    /// 暂停任务
    pub async fn pause(&self, gid: &str) -> Result<String, String> {
        let params = vec![json!(self.secret), json!(gid)];
        let result_gid: String = self.send_request("aria2.pause", params).await?;
        Ok(result_gid)
    }

    /// 恢复任务
    pub async fn resume(&self, gid: &str) -> Result<String, String> {
        let params = vec![json!(self.secret), json!(gid)];
        let result_gid: String = self.send_request("aria2.unpause", params).await?;
        Ok(result_gid)
    }

    /// 删除任务
    pub async fn remove(&self, gid: &str) -> Result<String, String> {
        let params = vec![json!(self.secret), json!(gid)];
        let result_gid: String = self.send_request("aria2.remove", params).await?;
        Ok(result_gid)
    }

    /// 获取任务状态
    pub async fn get_status(&self, gid: &str) -> Result<Aria2StatusResponse, String> {
        let params = vec![json!(self.secret), json!(gid)];
        let status: Aria2StatusResponse = self.send_request("aria2.tellStatus", params).await?;
        Ok(status)
    }

    /// 获取活跃任务
    pub async fn tell_active(&self) -> Result<Vec<Aria2StatusResponse>, String> {
        let keys = json!([
            "gid", "status", "totalLength", "completedLength",
            "downloadSpeed", "uploadSpeed", "dir", "files",
            "errorMessage"
        ]);
        let params = vec![json!(self.secret), keys];
        let tasks: Vec<Aria2StatusResponse> = self.send_request("aria2.tellActive", params).await?;
        Ok(tasks)
    }

    /// 获取等待中的任务
    pub async fn tell_waiting(&self, offset: i32, num: i32) -> Result<Vec<Aria2StatusResponse>, String> {
        let keys = json!([
            "gid", "status", "totalLength", "completedLength",
            "downloadSpeed", "uploadSpeed", "dir", "files",
            "errorMessage"
        ]);
        let params = vec![json!(self.secret), json!(offset), json!(num), keys];
        let tasks: Vec<Aria2StatusResponse> = self.send_request("aria2.tellWaiting", params).await?;
        Ok(tasks)
    }

    /// 获取已停止的任务
    pub async fn tell_stopped(&self, num: i32) -> Result<Vec<Aria2StatusResponse>, String> {
        let keys = json!([
            "gid", "status", "totalLength", "completedLength",
            "downloadSpeed", "uploadSpeed", "dir", "files",
            "errorMessage"
        ]);
        let params = vec![json!(self.secret), json!(num), keys];
        let tasks: Vec<Aria2StatusResponse> = self.send_request("aria2.tellStopped", params).await?;
        Ok(tasks)
    }

    /// 获取所有任务（活跃 + 等待 + 已停止）
    pub async fn get_all_tasks(&self) -> Result<Vec<Aria2StatusResponse>, String> {
        let mut all_tasks = Vec::new();

        // 获取活跃任务
        if let Ok(tasks) = self.tell_active().await {
            all_tasks.extend(tasks);
        }

        // 获取等待任务
        if let Ok(tasks) = self.tell_waiting(0, 1000).await {
            all_tasks.extend(tasks);
        }

        // 获取已停止任务（最近 100 个）
        if let Ok(tasks) = self.tell_stopped(100).await {
            all_tasks.extend(tasks);
        }

        Ok(all_tasks)
    }
}

/// aria2 进程管理器
pub struct Aria2ProcessManager {
    pub client: Aria2Client,
    pub process: Arc<Mutex<Option<Child>>>,
    pub running: Arc<AtomicBool>,
    pub port: u16,
    pub secret: String,
}

impl Aria2ProcessManager {
    pub fn new(port: u16, secret: &str) -> Self {
        Aria2ProcessManager {
            client: Aria2Client::new(port, secret),
            process: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            port,
            secret: secret.to_string(),
        }
    }

    /// 启动 aria2 服务
    pub fn start(&mut self, config_path: &str) -> Result<(), String> {
        if self.running.load(Ordering::SeqCst) {
            return Err("aria2 已在运行中".to_string());
        }

        // 检查端口是否已被占用
        if is_port_in_use(self.port) {
            // 端口已被占用，可能 aria2 已经在运行
            self.running.store(true, Ordering::SeqCst);
            return Ok(());
        }

        let child = Command::new(config_path)
            .arg("--conf-path=aria2.conf")
            .arg(format!("--rpc-listen-port={}", self.port))
            .arg(format!("--rpc-secret={}", self.secret))
            .arg("--enable-rpc=true")
            .spawn()
            .map_err(|e| format!("启动 aria2 失败: {}", e))?;

        *self.process.lock().unwrap() = Some(child);
        self.running.store(true, Ordering::SeqCst);

        // 等待 aria2 启动
        std::thread::sleep(Duration::from_millis(1000));

        Ok(())
    }

    /// 停止 aria2 服务
    pub fn stop(&mut self) -> Result<(), String> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        // 尝试优雅关闭
        let _ = self.client.send_request::<String>("aria2.shutdown", vec![json!(format!("token:{}", self.secret))]);

        // 等待进程退出
        std::thread::sleep(Duration::from_millis(500));

        // 强制终止（如果还在运行）
        if let Some(mut child) = self.process.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// 检查 aria2 是否运行
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Drop for Aria2ProcessManager {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// 检查端口是否被占用
fn is_port_in_use(port: u16) -> bool {
    use std::net::TcpListener;
    TcpListener::bind(("127.0.0.1", port)).is_err()
}
