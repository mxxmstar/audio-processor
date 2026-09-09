//! 独立图像质量处理模块（Real-ESRGAN 超分）。
//!
//! AI-1 先实现 Rust 与 Python Worker 的进程协议。真实模型推理已在
//! Python 侧（`python/image_ai_realesrgan/`）完成，本模块负责进程生命周期与路由。

pub mod ai_worker;
pub mod backend;
