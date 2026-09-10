// 子模块声明
pub mod audio_rename; // 音频命名、冲突处理和本地文件移动
pub mod audio_quality;
pub mod image_quality; // Python AI 音频增强与 Worker 协议
pub mod commands; // Tauri 命令（桥接层，仅负责前端调用与 fpcalc 资源路径解析）
pub mod recognizer; // 音频识别独立模块（指纹/查询/错误集中于此，与 Tauri 解耦）
pub mod http_client; // HTTP 客户端封装（用于向外部服务器如 B 站发送请求）
pub mod biliapi; // B 站 API 封装（基于 http_client，封装 bilidownload 中的 B 站调用）
pub mod bili_state; // 阶段 5：B 站功能共享状态（登录态目录 + 任务列表）
pub mod history; // 通用历史记录模块（音频识别 / B站下载共用）
pub mod aria2; // aria2 下载器模块
pub mod port_checker; // 端口占用查询模块

use bili_state::BiliState;
use commands::audio_quality::AudioQualityState;
use commands::image_quality::ImageQualityState;
use commands::aria2::Aria2ManagerState;
use aria2::Aria2Manager;
use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::Manager;

/// Tauri 应用入口（由 `main.rs` 调用）。
/// 负责构建并运行 Tauri 运行时，注册命令、插件与窗口配置。
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(BiliState::new())
        .manage(AudioQualityState::default())
        .manage(ImageQualityState::default())
        .manage(Arc::new(RwLock::new(Aria2Manager::new(6800, "aria2_secret_token_2026"))) as Aria2ManagerState)
        .setup(|app| {
            // 阶段 5：注入配置目录到 B 站存储 / WBI 缓存层
            if let Ok(dir) = app.path().app_config_dir() {
                app.state::<BiliState>().init_config_dir(dir);
            }
            // 注入 ffmpeg 搜索目录（作为 `media` 模块回溯的起点之一）。
            // `media` 会从该目录逐级向上回溯寻找 `bin/ffmpeg.exe`，
            // 因此这里传当前工作目录或 resource_dir 均可覆盖 dev / 打包场景。
            let ffmpeg_dir: std::path::PathBuf = match app.path().resource_dir() {
                Ok(rd) => rd,
                Err(_) => std::path::PathBuf::from("."),
            };
            println!(
                "[setup] 计算 ffmpeg_dir = {} （存在: {}）",
                ffmpeg_dir.display(),
                ffmpeg_dir.exists()
            );
            crate::biliapi::media::set_ffmpeg_dir(Some(&ffmpeg_dir));
            // 启动即预取 buvid3/buvid4 指纹，规避后续 web-space 接口 -352 风控
            let _ = tauri::async_runtime::spawn(async {
                let _ = crate::biliapi::buvid_cache::ensure_buvid().await;
            });
            Ok(())
        })
        // 注册命令，使前端可通过 invoke 调用
        .invoke_handler(tauri::generate_handler![
            commands::identify,
            commands::get_history,
            commands::count_history,
            commands::delete_history,
            commands::open_path,
            commands::audio_quality::audio_quality_check_ai_runtime,
            commands::audio_quality::audio_quality_download_model,
            commands::audio_quality::audio_quality_start,
            commands::audio_quality::audio_quality_cancel,
            commands::audio_quality::audio_quality_list_models,
            commands::audio_quality::audio_quality_list_tasks,
            commands::image_quality::enhance_image,
            commands::image_quality::enhance_image_cancel,
            commands::image_quality::enhance_image_check_ai_runtime,
            commands::image_quality::enhance_image_download_model,
            commands::image_quality::enhance_image_list_models,
            commands::image_quality::enhance_image_list_tasks,
            // 阶段 5：B 站下载命令
            commands::bili::bili_resolve,
            commands::bili::bili_resolve_async,
            commands::bili::bili_start_download,
            commands::bili::bili_pause_download,
            commands::bili::bili_stop_download,
            commands::bili::bili_list_tasks,
            commands::bili::bili_login_qr,
            commands::bili::bili_login_poll,
            commands::bili::bili_check_login,
            commands::bili::bili_user_info,
            commands::bili::bili_logout,
            // aria2 下载器命令
            commands::aria2::aria2_start_service,
            commands::aria2::aria2_stop_service,
            commands::aria2::aria2_is_running,
            commands::aria2::aria2_add_task,
            commands::aria2::aria2_pause_task,
            commands::aria2::aria2_resume_task,
            commands::aria2::aria2_remove_task,
            commands::aria2::aria2_list_tasks,
            commands::aria2::aria2_refresh_tasks,
            commands::aria2::aria2_get_history,
            commands::aria2::aria2_cleanup_completed,
            // 端口占用查询命令
            commands::port_checker::port_query,
            commands::port_checker::port_check,
            commands::port_checker::port_query_by_pid,
            // 端口占用 kill 命令
            commands::port_checker::port_kill_process,
            commands::port_checker::port_kill_by_port,
        ])
        // 注册对话框插件（前端用其打开文件选择框）
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用失败");
}
