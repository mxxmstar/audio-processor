// 子模块声明
pub mod commands;       // Tauri 命令（桥接层，仅负责前端调用与 fpcalc 资源路径解析）
pub mod recognizer;     // 音频识别独立模块（指纹/查询/错误集中于此，与 Tauri 解耦）

/// Tauri 应用入口（由 `main.rs` 调用）。
/// 负责构建并运行 Tauri 运行时，注册命令、插件与窗口配置。
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // 注册命令，使前端可通过 invoke 调用
        .invoke_handler(tauri::generate_handler![commands::identify])
        // 注册对话框插件（前端用其打开文件选择框）
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用失败");
}
