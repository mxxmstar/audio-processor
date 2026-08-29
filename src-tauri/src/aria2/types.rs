use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// aria2 任务状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Aria2TaskStatus {
    Active,
    Waiting,
    Paused,
    Complete,
    Error,
    Removed,
}

impl Aria2TaskStatus {
    pub fn from_aria2_status(status: &str) -> Self {
        match status {
            "active" => Aria2TaskStatus::Active,
            "waiting" => Aria2TaskStatus::Waiting,
            "paused" => Aria2TaskStatus::Paused,
            "complete" => Aria2TaskStatus::Complete,
            "error" => Aria2TaskStatus::Error,
            "removed" => Aria2TaskStatus::Removed,
            _ => Aria2TaskStatus::Error,
        }
    }

    pub fn to_display_text(&self) -> &'static str {
        match self {
            Aria2TaskStatus::Active => "下载中",
            Aria2TaskStatus::Waiting => "等待中",
            Aria2TaskStatus::Paused => "已暂停",
            Aria2TaskStatus::Complete => "已完成",
            Aria2TaskStatus::Error => "错误",
            Aria2TaskStatus::Removed => "已删除",
        }
    }
}

/// aria2 下载任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aria2Task {
    pub gid: String,
    pub name: String,
    pub url: String,
    pub status: Aria2TaskStatus,
    pub total_length: u64,
    pub completed_length: u64,
    pub download_speed: u64,
    pub upload_speed: u64,
    pub progress: f64,
    pub error_message: Option<String>,
    pub created_at: i64,
    pub output_dir: String,
    pub file_path: Option<String>,
}

impl Aria2Task {
    pub fn from_aria2_response(status: &Aria2StatusResponse) -> Self {
        let total_length: u64 = status.total_length.parse().unwrap_or(0);
        let completed_length: u64 = status.completed_length.parse().unwrap_or(0);
        let download_speed: u64 = status.download_speed.parse().unwrap_or(0);
        let upload_speed: u64 = status.upload_speed.parse().unwrap_or(0);

        let progress = if total_length > 0 {
            (completed_length as f64 / total_length as f64) * 100.0
        } else {
            0.0
        };

        let name = if !status.files.is_empty() {
            status.files[0]
                .path
                .split(['/', '\\'])
                .last()
                .unwrap_or("unknown")
                .to_string()
        } else {
            "unknown".to_string()
        };

        let url = if !status.files.is_empty() && !status.files[0].uris.is_empty() {
            status.files[0].uris[0].uri.clone()
        } else {
            "".to_string()
        };

        let file_path = if !status.files.is_empty() {
            Some(status.files[0].path.clone())
        } else {
            None
        };

        Aria2Task {
            gid: status.gid.clone(),
            name,
            url,
            status: Aria2TaskStatus::from_aria2_status(&status.status),
            total_length,
            completed_length,
            download_speed,
            upload_speed,
            progress,
            error_message: status.error_message.clone(),
            created_at: 0, // aria2 不直接提供创建时间，需要自己记录
            output_dir: status.dir.clone(),
            file_path,
        }
    }
}

/// 添加任务请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddTaskRequest {
    pub urls: Vec<String>,
    pub output_dir: Option<String>,
    pub file_name: Option<String>,
    pub options: Option<HashMap<String, String>>,
}

/// aria2 JSON-RPC 请求
#[derive(Debug, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: String,
    pub method: String,
    pub params: Vec<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(method: &str, params: Vec<serde_json::Value>) -> Self {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: "1".to_string(),
            method: method.to_string(),
            params,
        }
    }
}

/// aria2 JSON-RPC 响应
#[derive(Debug, Deserialize)]
pub struct JsonRpcResponse<T> {
    pub jsonrpc: String,
    pub id: String,
    pub result: Option<T>,
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

/// aria2 状态响应
#[derive(Debug, Deserialize)]
pub struct Aria2StatusResponse {
    pub gid: String,
    pub status: String,
    pub total_length: String,
    pub completed_length: String,
    pub download_speed: String,
    pub upload_speed: String,
    pub dir: String,
    pub files: Vec<Aria2File>,
    pub error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Aria2File {
    pub path: String,
    pub uris: Vec<Aria2Uri>,
}

#[derive(Debug, Deserialize)]
pub struct Aria2Uri {
    pub uri: String,
}

/// aria2 版本信息
#[derive(Debug, Deserialize)]
pub struct Aria2Version {
    pub version: String,
}

/// 下载历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadHistory {
    pub gid: String,
    pub name: String,
    pub url: String,
    pub status: Aria2TaskStatus,
    pub total_length: u64,
    pub completed_length: u64,
    pub output_dir: String,
    pub file_path: Option<String>,
    pub created_at: i64,
    pub completed_at: Option<i64>,
}

impl DownloadHistory {
    pub fn from_task(task: &Aria2Task) -> Self {
        DownloadHistory {
            gid: task.gid.clone(),
            name: task.name.clone(),
            url: task.url.clone(),
            status: task.status.clone(),
            total_length: task.total_length,
            completed_length: task.completed_length,
            output_dir: task.output_dir.clone(),
            file_path: task.file_path.clone(),
            created_at: task.created_at,
            completed_at: if task.status == Aria2TaskStatus::Complete {
                Some(chrono::Utc::now().timestamp())
            } else {
                None
            },
        }
    }
}
