"""阶段 0/1 可行性自检：真实权重加载 + 前向 + 自研 tile 分块正确性。

运行（需 `.venv-realesrgan`）：
    .\\.venv-realesrgan\\Scripts\\python.exe python/image_ai_realesrgan/selfcheck.py

本脚本验证代码路径（需要真实权重，不需要 Rust 侧）：
- 构造 RRDBNet 并按 manifest 真实权重加载 state_dict
- 小图走整图直推路径：输出尺寸 == 输入 × scale
- 大图走自研 tile 路径：输出尺寸 == 输入 × scale
- 取消标志在 tile 边界生效（协作式取消）
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipeline  # noqa: E402
from worker import ModelSpec  # noqa: E402


def _locate_weight() -> Path:
    env_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(pipeline.SCRIPT_ROOT / "models")))
    candidate = env_dir / "cache" / "realesrgan" / "RealESRGAN_x4plus.pth"
    return candidate


def _check_forward(model_path: Path) -> bool:
    runner = pipeline.RealESRGANRunner(str(model_path), "RealESRGAN_x4plus", 4, torch.device("cpu"))
    print(f"[1] 模型加载完成（x4plus, CPU）")

    rng = np.random.default_rng(0)

    # 小图整图直推：24x32 -> 96x128
    small = (rng.random((24, 32, 3)) * 255).astype(np.uint8)
    out = runner.upscale(small, threading.Event(), "selfcheck", "png")
    print(f"[2] 小图整图直推: {small.shape[1]}x{small.shape[0]} -> {out.shape[1]}x{out.shape[0]}")
    if out.shape[:2] != (96, 128):
        print(f"    [失败] 小图输出尺寸 {out.shape[:2]} 应为 (128, 96) 的转置 (96, 128)")
        return False

    # 大图 tile 路径：200x150 -> 800x600，tile=64
    big = (rng.random((200, 150, 3)) * 255).astype(np.uint8)
    out_big = runner.upscale(big, threading.Event(), "selfcheck", "png", tile=64)
    print(f"[3] 大图 tile 路径(tile=64): {big.shape[1]}x{big.shape[0]} -> {out_big.shape[1]}x{out_big.shape[0]}")
    if out_big.shape[:2] != (800, 600):
        print(f"    [失败] 大图输出尺寸 {out_big.shape[:2]} 应为 (800, 600)")
        return False

    # tile 与整图一致性（取整图与 tile 在共同区域的容差对比，验证拼接无接缝）
    out_big_whole = runner.upscale(big, threading.Event(), "selfcheck", "png", tile=0)
    # 取中心 400x400 区域对比（避开 pad 边界）
    a = out_big_whole[200:600, 200:600].astype(np.int16)
    b = out_big[200:600, 200:600].astype(np.int16)
    max_diff = int(np.abs(a - b).max())
    print(f"[4] tile vs 整图 中心区域最大像素差: {max_diff}")
    # 该差异主要来自 tile 边界处模型看到的上下文不同（官方 enhance 同样存在，R11）：
    # 仅当差异过大（>32/255）才提示可能存在可见接缝/色块。
    if max_diff > 32:
        print(f"    [警告] tile 与整图差异较大（{max_diff}），可能存在接缝/色块（R11）")

    # RGBA 处理：alpha 单独放大
    rgba = np.zeros((40, 30, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = (rng.random((40, 30, 3)) * 255).astype(np.uint8)
    rgba[:, :, 3] = 200
    out_rgba = runner.upscale(rgba, threading.Event(), "selfcheck", "png")
    print(f"[5] RGBA: {rgba.shape[1]}x{rgba.shape[0]} -> {out_rgba.shape[1]}x{out_rgba.shape[0]} channels={out_rgba.shape[2]}")
    if out_rgba.shape[2] != 4 or out_rgba.shape[:2] != (160, 120):
        print("    [失败] RGBA 输出应为 (160,120,4)")
        return False

    # 取消路径：在 tile 边界应触发 CANCELLED
    from protocol import WorkerFailure  # noqa: E402

    cancel = threading.Event()
    cancel.set()
    try:
        runner.upscale(big, cancel, "selfcheck", "png", tile=64)
        print("    [失败] 取消标志未生效")
        return False
    except WorkerFailure as failure:
        if failure.code != "CANCELLED":
            print(f"    [失败] 取消抛出的错误码为 {failure.code}，应为 CANCELLED")
            return False
        print(f"[6] 取消标志生效: {failure.code}")

    return True


def main() -> int:
    model_path = _locate_weight()
    if not model_path.is_file():
        print(f"权重缺失，跳过自检: {model_path}")
        print("（先运行阶段 0：下载 RealESRGAN_x4plus.pth 到 models/cache/realesrgan/）")
        return 1
    ok = _check_forward(model_path)
    print("\nReal-ESRGAN 自检:", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
