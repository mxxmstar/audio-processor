use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter};
use tokio::sync::RwLock;

use super::client::Aria2ProcessManager;
use super::types::*;

/// aria2 任务管理器
pub struct Aria2Manager {
    pub process_manager: Arc<RwLock<Aria2ProcessManager>>,
    pub tasks: Arc<RwLock<HashMap<String, Aria2Task>>>,
    pub history: Arc<RwLock<Vec<DownloadHistory>>>,
    pub app_handle: Option<AppHandle>,
}

impl Aria2Manager {
    pub fn new(port: u16, secret: &str) -> Self {
        Aria2Manager {
            process_manager: Arc::new(RwLock::new(Aria2ProcessManager::new(port, secret))),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            app_handle: None,
        }
    }

    pub fn set_app_handle(&mut self, handle: AppHandle) {
        self.app_handle = Some(handle);
    }

    /// 启动 aria2 服务
    pub async fn start(&self, config_path: &str) -> Result<(), String> {
        let mut pm = self.process_manager.write().await;
        pm.start(config_path)
    }

    /// 停止 aria2 服务
    pub async fn stop(&self) -> Result<(), String> {
        let mut pm = self.process_manager.write().await;
        pm.stop()
    }

    /// 检查服务是否运行
    pub async fn is_running(&self) -> bool {
        let pm = self.process_manager.read().await;
        pm.is_running()
    }

    /// 添加下载任务
    pub async fn add_task(&self, request: AddTaskRequest) -> Result<String, String> {
        let pm = self.process_manager.read().await;

        let mut options = HashMap::new();
        if let Some(output_dir) = &request.output_dir {
            options.insert("dir".to_string(), output_dir.clone());
        }
        if let Some(file_name) = &request.file_name {
            options.insert("out".to_string(), file_name.clone());
        }
        if let Some(user_options) = &request.options {
            options.extend(user_options.clone());
        }

        // 保存第一个 URL 用于任务记录
        let first_url = request.urls.first().cloned().unwrap_or_default();

        let gid = pm.client.add_uri(request.urls, Some(options)).await?;

        // 记录任务创建时间
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let mut task = Aria2Task {
            gid: gid.clone(),
            name: "未知文件".to_string(),
            url: first_url,
            status: Aria2TaskStatus::Waiting,
            total_length: 0,
            completed_length: 0,
            download_speed: 0,
            upload_speed: 0,
            progress: 0.0,
            error_message: None,
            created_at: now,
            output_dir: request.output_dir.unwrap_or_else(|| "./downloads".to_string()),
            file_path: None,
        };

        // 立即获取任务状态
        if let Ok(status) = pm.client.get_status(&gid).await {
            task = Aria2Task::from_aria2_response(&status);
            task.created_at = now;
        }

        // 保存到任务列表
        self.tasks.write().await.insert(gid.clone(), task);

        // 发送事件通知
        self.emit_task_update().await;

        Ok(gid)
    }

    /// 暂停任务
    pub async fn pause_task(&self, gid: &str) -> Result<(), String> {
        let pm = self.process_manager.read().await;
        pm.client.pause(gid).await?;

        // 更新本地状态
        if let Some(task) = self.tasks.write().await.get_mut(gid) {
            task.status = Aria2TaskStatus::Paused;
        }

        self.emit_task_update().await;
        Ok(())
    }

    /// 恢复任务
    pub async fn resume_task(&self, gid: &str) -> Result<(), String> {
        let pm = self.process_manager.read().await;
        pm.client.resume(gid).await?;

        // 更新本地状态
        if let Some(task) = self.tasks.write().await.get_mut(gid) {
            task.status = Aria2TaskStatus::Active;
        }

        self.emit_task_update().await;
        Ok(())
    }

    /// 删除任务
    pub async fn remove_task(&self, gid: &str) -> Result<(), String> {
        let pm = self.process_manager.read().await;
        pm.client.remove(gid).await?;

        // 从任务列表移除
        self.tasks.write().await.remove(gid);

        self.emit_task_update().await;
        Ok(())
    }

    /// 获取所有任务
    pub async fn list_tasks(&self) -> Result<Vec<Aria2Task>, String> {
        let pm = self.process_manager.read().await;
        let statuses = pm.client.get_all_tasks().await?;

        let mut tasks = Vec::new();
        for status in statuses {
            let task = Aria2Task::from_aria2_response(&status);
            tasks.push(task);
        }

        // 更新本地缓存
        let mut task_map = self.tasks.write().await;
        for task in &tasks {
            task_map.insert(task.gid.clone(), task.clone());
        }

        Ok(tasks)
    }

    /// 刷新任务状态并发送事件
    pub async fn refresh_and_emit(&self) -> Result<(), String> {
        let _ = self.list_tasks().await?;
        self.emit_task_update().await;
        Ok(())
    }

    /// 发送任务更新事件
    async fn emit_task_update(&self) {
        if let Some(handle) = &self.app_handle {
            let tasks = self.tasks.read().await;
            let task_list: Vec<Aria2Task> = tasks.values().cloned().collect();
            let _ = handle.emit("aria2-tasks-update", task_list);
        }
    }

    /// 获取下载历史
    pub async fn get_history(&self) -> Result<Vec<DownloadHistory>, String> {
        let history = self.history.read().await;
        Ok(history.clone())
    }

    /// 添加历史记录
    pub async fn add_history(&self, task: &Aria2Task) -> Result<(), String> {
        let history_item = DownloadHistory::from_task(task);
        self.history.write().await.push(history_item);
        Ok(())
    }

    /// 清理已完成的任务（移到历史记录）
    pub async fn cleanup_completed_tasks(&self) -> Result<(), String> {
        let mut tasks = self.tasks.write().await;
        let mut completed = Vec::new();

        // 找出已完成的任务
        tasks.retain(|_, task| {
            if task.status == Aria2TaskStatus::Complete {
                completed.push(task.clone());
                false
            } else {
                true
            }
        });

        // 添加到历史记录
        for task in completed {
            drop(tasks);
            self.add_history(&task).await?;
            tasks = self.tasks.write().await;
        }

        Ok(())
    }
}
