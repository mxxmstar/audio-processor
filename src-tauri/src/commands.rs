use tauri::AppHandle;
use tauri::Manager;

use crate::history::{self, HistoryKind};
use crate::recognizer::run_identify;
use crate::recognizer::SongInfo;

/// 阶段 5：B 站下载相关 Tauri 命令
pub mod bili;
/// 音频质量处理相关 Tauri 命令
pub mod audio_quality;
/// aria2 下载器相关 Tauri 命令
pub mod aria2;

/// 解析历史数据库目录：优先应用配置目录，回退资源目录，再回退仓库 bin 旁。
/// 供 `commands` 与 `commands::bili` 共用。
pub(crate) fn history_dir(app: &AppHandle) -> std::path::PathBuf {
    if let Ok(dir) = app.path().app_config_dir() {
        return dir;
    }
    if let Ok(rd) = app.path().resource_dir() {
        return rd;
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// 解析随附的 fpcalc 工具路径，统一覆盖打包环境和开发环境。
pub(crate) fn fpcalc_path(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("获取资源目录失败: {e}"))?;
    let mut path = resource_dir.join("fpcalc.exe");
    if !path.exists() {
        path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("bin")
            .join("fpcalc.exe");
    }
    if !path.exists() {
        return Err(format!("找不到 fpcalc.exe: {}", path.display()));
    }
    Ok(path)
}

/// Tauri 命令：供前端通过 `invoke('identify', { path })` 调用。
/// 接收音频文件绝对路径，返回识别结果；成功后自动写入通用历史库。
#[tauri::command]
pub fn identify(app: AppHandle, path: String) -> Result<SongInfo, String> {
    let fpcalc_path = fpcalc_path(&app)?.to_string_lossy().into_owned();

    let path_clone = path.clone();
    let result =
        tauri::async_runtime::block_on(
            async move { run_identify(&fpcalc_path, &path_clone).await },
        )
        .map_err(|e| e.to_string())?;

    // 识别成功，写入通用历史库（失败不影响返回结果）
    if let Ok(conn) = history::open_db(&history_dir(&app)) {
        let payload = serde_json::to_string(&result).unwrap_or_default();
        if let Err(e) = history::insert(
            &conn,
            HistoryKind::Recognize,
            &result.title,
            &result.artist,
            &payload,
            "",
        ) {
            eprintln!("[history] 写入识别历史失败: {e}");
        }
    }

    Ok(result)
}

/// 通用历史查询：按种类过滤（可选），返回最近 `limit` 条（默认 200）。
#[tauri::command]
pub fn get_history(
    app: AppHandle,
    kind: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<history::HistoryItem>, String> {
    let conn = history::open_db(&history_dir(&app)).map_err(|e| e.to_string())?;
    let kind_ref = kind.as_deref();
    history::list(&conn, kind_ref, limit.unwrap_or(200)).map_err(|e| e.to_string())
}

/// 按 id 删除一条历史记录。
#[tauri::command]
pub fn delete_history(app: AppHandle, id: i64) -> Result<(), String> {
    let conn = history::open_db(&history_dir(&app)).map_err(|e| e.to_string())?;
    history::delete(&conn, id).map_err(|e| e.to_string())
}

/// 打开本地文件或目录（用于历史记录「打开下载目录」）。
///
/// - `path` 为文件时，打开其所在目录并选中该文件；
/// - `path` 为目录时，直接打开该目录；
/// - 若路径不存在（如文件已被删除），返回 `Err("文件不存在")`。
#[tauri::command]
pub fn open_path(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);
    if !p.exists() {
        return Err("文件不存在".to_string());
    }

    // 文件则定位到所在目录并选中；目录则直接打开。
    let (target, select) = if p.is_dir() {
        (p.to_path_buf(), None)
    } else {
        (
            p.parent()
                .map(|x| x.to_path_buf())
                .unwrap_or_else(|| p.to_path_buf()),
            Some(p),
        )
    };

    let status = if cfg!(target_os = "windows") {
        let mut cmd = std::process::Command::new("explorer");
        if let Some(f) = select {
            // explorer /select,"path" 可打开目录并选中文件
            cmd.arg("/select,").arg(f);
        } else {
            cmd.arg(&target);
        }
        cmd.status()
    } else if cfg!(target_os = "macos") {
        let mut cmd = std::process::Command::new("open");
        if let Some(f) = select {
            cmd.arg("-R").arg(f);
        } else {
            cmd.arg(&target);
        }
        cmd.status()
    } else {
        std::process::Command::new("xdg-open").arg(&target).status()
    };

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("打开失败（退出码 {s}）")),
        Err(e) => Err(format!("打开失败: {e}")),
    }
}
