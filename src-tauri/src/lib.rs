// 子模块声明
pub mod commands;     // Tauri 命令（单独成模块，避免与 generate_handler! 的宏同名冲突）
pub mod fingerprint;  // 音频解码 + Chromaprint 指纹生成
pub mod acoustid;     // AcoustID 指纹查询
pub mod musicbrainz;  // MusicBrainz 曲目详情查询
pub mod error;        // 统一错误类型

/// Tauri 应用入口（由 `main.rs` 调用）。
/// 负责构建并运行 Tauri 运行时，注册命令与窗口配置。
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // 注册命令，使前端可通过 invoke 调用
        .invoke_handler(tauri::generate_handler![commands::identify])
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用失败");
}
