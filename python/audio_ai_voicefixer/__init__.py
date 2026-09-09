"""VoiceFixer 语音修复 Worker 包。

各模块的职责与加载顺序约束见 `worker.py` 顶部说明：
`protocol.py` 与 `worker.py` 只依赖标准库（ready 握手前可用），
`pipeline.py` 在收到 `process` 命令后才被导入（torch 导入约 7 秒，
会超出 Rust 侧 3 秒的 ready 超时）。
"""
