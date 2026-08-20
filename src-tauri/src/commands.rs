use tauri::AppHandle;
use tauri::Manager;

use crate::recognizer::{history, run_identify, SongInfo};

/// 阶段 5：B 站下载相关 Tauri 命令
pub mod bili;

/// 解析历史数据库目录：优先应用配置目录，回退资源目录，再回退仓库 bin 旁。
fn history_dir(app: &AppHandle) -> std::path::PathBuf {
    if let Ok(dir) = app.path().app_config_dir() {
        return dir;
    }
    if let Ok(rd) = app.path().resource_dir() {
        return rd;
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Tauri 命令：供前端通过 `invoke('identify', { path })` 调用。
/// 接收音频文件绝对路径，返回识别结果；成功后自动写入历史库（失败仅告警不阻断）。
#[tauri::command]
pub fn identify(app: AppHandle, path: String) -> Result<SongInfo, String> {
    // 解析随附的 fpcalc 工具路径（开发期与打包后位置不同）
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("获取资源目录失败: {e}"))?;
    let mut fpcalc_path = resource_dir.join("fpcalc.exe");
    // 开发期（tauri dev）资源目录可能不含 fpcalc，回退到仓库 bin 目录
    if !fpcalc_path.exists() {
        fpcalc_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("bin")
            .join("fpcalc.exe");
    }
    let fpcalc_path = fpcalc_path.to_string_lossy().to_string();

    let path_clone = path.clone();
    let result = tauri::async_runtime::block_on(async move {
        run_identify(&fpcalc_path, &path_clone).await
    })
    .map_err(|e| e.to_string())?;

    // 识别成功，自动落历史库（失败不影响返回结果）
    if let Ok(conn) = history::open_db(&history_dir(&app)) {
        if let Err(e) = history::insert_record(&conn, &result, &path) {
            eprintln!("[history] 写入识别历史失败: {e}");
        }
    }

    Ok(result)
}

/// 获取识别历史列表（按时间倒序，默认 100 条）。
#[tauri::command]
pub fn get_recognize_history(app: AppHandle, limit: Option<usize>) -> Result<Vec<history::HistoryRecord>, String> {
    let conn = history::open_db(&history_dir(&app)).map_err(|e| e.to_string())?;
    history::list_records(&conn, limit.unwrap_or(100)).map_err(|e| e.to_string())
}

/// 按 id 获取单条历史记录。
#[tauri::command]
pub fn get_recognize_record(app: AppHandle, id: i64) -> Result<Option<history::HistoryRecord>, String> {
    let conn = history::open_db(&history_dir(&app)).map_err(|e| e.to_string())?;
    history::get_record(&conn, id).map_err(|e| e.to_string())
}

/// 按 id 删除一条历史记录。
#[tauri::command]
pub fn delete_recognize_record(app: AppHandle, id: i64) -> Result<(), String> {
    let conn = history::open_db(&history_dir(&app)).map_err(|e| e.to_string())?;
    history::delete_record(&conn, id).map_err(|e| e.to_string())
}
