use tauri::AppHandle;
use tauri::Manager;

use crate::recognizer::{run_identify, SongInfo};

/// Tauri 命令：供前端通过 `invoke('identify', { path })` 调用。
/// 接收音频文件绝对路径，返回识别结果或错误信息字符串。
///
/// 内部仅负责：解析随附的 `fpcalc.exe` 资源路径，然后把实际识别工作
/// 委托给独立模块 `crate::recognizer::run_identify`（与 Tauri 解耦）。
///
/// 这里用 Tauri 的异步运行时 `block_on` 在同步命令中等待结果，
/// 使命令签名保持简单；命令放在独立模块以规避宏同名冲突问题。
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

    tauri::async_runtime::block_on(async move {
        run_identify(&fpcalc_path, &path)
            .await
            .map_err(|e| e.to_string())
    })
}
