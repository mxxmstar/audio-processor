// 仅在非调试构建（即 release 打包）时隐藏控制台窗口，
// 调试模式保留控制台方便查看日志。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // 调用 lib 中定义的 Tauri 应用入口
    audio_processor_lib::run()
}
