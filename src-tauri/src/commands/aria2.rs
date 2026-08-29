use std::path::PathBuf;
use std::sync::Arc;

use tauri::{AppHandle, Manager, State};
use tokio::sync::RwLock;

use crate::aria2::{AddTaskRequest, Aria2Manager, Aria2Task, DownloadHistory};

/// aria2 管理器状态
pub type Aria2ManagerState = Arc<RwLock<Aria2Manager>>;

/// 启动 aria2 服务
#[tauri::command]
pub async fn aria2_start_service(
    app: AppHandle,
    manager: State<'_, Aria2ManagerState>,
) -> Result<(), String> {
    let mgr = manager.read().await;

    // 获取 aria2 可执行文件路径
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("获取资源目录失败: {}", e))?;

    let mut aria2_path = resource_dir.join("aria2").join("aria2c.exe");
    if !aria2_path.exists() {
        // 开发环境回退
        aria2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("bin")
            .join("aria2")
            .join("aria2c.exe");
    }

    if !aria2_path.exists() {
        return Err(format!("找不到 aria2c.exe: {}", aria2_path.display()));
    }

    // 设置工作目录为 aria2 所在目录（aria2 会自动读取同目录下的 aria2.conf）
    let work_dir = aria2_path.parent().unwrap();
    std::env::set_current_dir(work_dir)
        .map_err(|e| format!("设置工作目录失败: {}", e))?;

    mgr.start(&aria2_path.to_string_lossy()).await
}

/// 停止 aria2 服务
#[tauri::command]
pub async fn aria2_stop_service(
    manager: State<'_, Aria2ManagerState>,
) -> Result<(), String> {
    let mgr = manager.read().await;
    mgr.stop().await
}

/// 检查 aria2 服务状态
#[tauri::command]
pub async fn aria2_is_running(
    manager: State<'_, Aria2ManagerState>,
) -> Result<bool, String> {
    let mgr = manager.read().await;
    Ok(mgr.is_running().await)
}

/// 添加下载任务
#[tauri::command]
pub async fn aria2_add_task(
    manager: State<'_, Aria2ManagerState>,
    request: AddTaskRequest,
) -> Result<String, String> {
    let mgr = manager.read().await;
    mgr.add_task(request).await
}

/// 暂停任务
#[tauri::command]
pub async fn aria2_pause_task(
    manager: State<'_, Aria2ManagerState>,
    gid: String,
) -> Result<(), String> {
    let mgr = manager.read().await;
    mgr.pause_task(&gid).await
}

/// 恢复任务
#[tauri::command]
pub async fn aria2_resume_task(
    manager: State<'_, Aria2ManagerState>,
    gid: String,
) -> Result<(), String> {
    let mgr = manager.read().await;
    mgr.resume_task(&gid).await
}

/// 删除任务
#[tauri::command]
pub async fn aria2_remove_task(
    manager: State<'_, Aria2ManagerState>,
    gid: String,
) -> Result<(), String> {
    let mgr = manager.read().await;
    mgr.remove_task(&gid).await
}

/// 获取所有任务列表
#[tauri::command]
pub async fn aria2_list_tasks(
    manager: State<'_, Aria2ManagerState>,
) -> Result<Vec<Aria2Task>, String> {
    let mgr = manager.read().await;
    mgr.list_tasks().await
}

/// 刷新任务状态
#[tauri::command]
pub async fn aria2_refresh_tasks(
    manager: State<'_, Aria2ManagerState>,
) -> Result<(), String> {
    let mgr = manager.read().await;
    mgr.refresh_and_emit().await
}

/// 获取下载历史
#[tauri::command]
pub async fn aria2_get_history(
    manager: State<'_, Aria2ManagerState>,
) -> Result<Vec<DownloadHistory>, String> {
    let mgr = manager.read().await;
    mgr.get_history().await
}

/// 清理已完成的任务（移到历史记录）
#[tauri::command]
pub async fn aria2_cleanup_completed(
    manager: State<'_, Aria2ManagerState>,
) -> Result<(), String> {
    let mgr = manager.read().await;
    mgr.cleanup_completed_tasks().await
}
