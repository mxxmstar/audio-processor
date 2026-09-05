"""以 `python -m audio_ai_flashsr` 启动 Worker。

与 Rust 侧的 `python <绝对路径>/worker.py` 等价：两者都把本包目录加入
sys.path 后以顶层模块方式解析 protocol / pipeline，因此行为一致。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
