// 子模块声明
pub mod commands;       // Tauri 命令（桥接层，仅负责前端调用与 fpcalc 资源路径解析）
pub mod recognizer;     // 音频识别独立模块（指纹/查询/错误集中于此，与 Tauri 解耦）
pub mod http_client;    // HTTP 客户端封装（用于向外部服务器如 B 站发送请求）
pub mod biliapi;        // B 站 API 封装（基于 http_client，封装 bilidownload 中的 B 站调用）
pub mod bili_state;     // 阶段 5：B 站功能共享状态（登录态目录 + 任务列表）

use bili_state::BiliState;
use tauri::Manager;

/// Tauri 应用入口（由 `main.rs` 调用）。
/// 负责构建并运行 Tauri 运行时，注册命令、插件与窗口配置。
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(BiliState::new())
        .setup(|app| {
            // 阶段 5：注入配置目录到 B 站存储 / WBI 缓存层
            if let Ok(dir) = app.path().app_config_dir() {
                app.state::<BiliState>().init_config_dir(dir);
            }
            // 注入 ffmpeg 搜索目录
            // - 开发期：直接用项目根 bin/（CARGO_MANIFEST_DIR/../bin），最可靠
            // - 打包后：取 resource_dir（ffmpeg 由 tauri.conf.json 的 resources 分发）
            let ffmpeg_dir: std::path::PathBuf = if let Ok(m) = std::env::var("CARGO_MANIFEST_DIR") {
                std::path::Path::new(&m).join("../bin")
            } else {
                match app.path().resource_dir() {
                    Ok(rd) => rd,
                    Err(_) => std::path::PathBuf::from("bin"),
                }
            };
            crate::biliapi::media::set_ffmpeg_dir(Some(&ffmpeg_dir));
            Ok(())
        })
        // 注册命令，使前端可通过 invoke 调用
        .invoke_handler(tauri::generate_handler![
            commands::identify,
            // 阶段 5：B 站下载命令
            commands::bili::bili_resolve,
            commands::bili::bili_start_download,
            commands::bili::bili_list_tasks,
            commands::bili::bili_login_qr,
            commands::bili::bili_login_poll,
            commands::bili::bili_check_login,
            commands::bili::bili_logout,
        ])
        // 注册对话框插件（前端用其打开文件选择框）
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用失败");
}
