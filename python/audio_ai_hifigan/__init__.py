"""HiFi-GAN 后端（方案 A：独立子进程 Worker）。

阶段 0 仅落地「vendor 导入桥接 + 薄封装」与「Generator 单次前向可行性自检」，
后续阶段（Worker / 路由 / 前端）再在 `vendor_bridge` 之上扩展。

本包刻意与 `audio_ai_flashsr` 平级、各自独立成进程（同 FlashSR 范式），
仅复用其 vendor 快照里的 HiFi-GAN 实现，不改动 vendor 源码。
"""

from .vendor_bridge import (
    bootstrap_vendor_path,
    ensure_inference_only_imports,
    install_vendor_aliases,
    get_default_config,
    build_generator,
    HiFiGanRunner,
)

__all__ = [
    "bootstrap_vendor_path",
    "ensure_inference_only_imports",
    "install_vendor_aliases",
    "get_default_config",
    "build_generator",
    "HiFiGanRunner",
]
