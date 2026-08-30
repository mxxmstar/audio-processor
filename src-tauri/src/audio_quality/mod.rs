//! 独立音频质量处理模块。
//!
//! AI-1 先实现 Rust 与 Python Worker 的进程协议。真实模型推理将在协议
//! 和生命周期验证完成后接入，不在本模块中嵌入 Python 运行时。

pub mod ai_worker;
