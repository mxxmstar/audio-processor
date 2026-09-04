"""FlashSR 音频超分 Worker 包。

独立进程，通过 JSONL 协议与 Rust 侧通信；协议版本与 `audio_ai` 一致（v1）。
环境搭建、依赖实测结论与模型约束见本目录下的 `README.md`。
"""
