"""环境自检：验证 FlashSR 依赖、导入与（可选）权重加载和一次前向。

不带参数时只做依赖与导入检查，不需要权重文件；
加 `--weights` 会加载三个权重并对随机张量做一次 5.12 秒前向。

    python -m audio_ai_flashsr.selfcheck
    python -m audio_ai_flashsr.selfcheck --weights
    python -m audio_ai_flashsr.selfcheck --weights --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_MODEL_DIR = SCRIPT_ROOT / "models" / "cache" / "flashsr"

#: 与 models/manifest.json 中 flashsr 条目的 file 名保持一致
WEIGHT_NAMES = ("student_ldm.pth", "sr_vocoder.pth", "vae.pth")

#: 推理期真实依赖。`matplotlib` 与 `sklearn` 由 backend 打桩屏蔽，
#: 故意不在其中 —— 若误装，这里也不会检查，见 README「依赖实测结论」。
REQUIRED_DEPENDENCIES = (
    "torch",
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
    "einops",
    "yaml",
    "tqdm",
)

#: 应由 backend.ensure_inference_only_imports() 打桩屏蔽的模块
STUBBED_MODULES = (
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.pylab",
    "sklearn.manifold",
    "librosa.display",
)


def check_dependencies() -> bool:
    print("=== [1/3] 依赖检查 ===")
    missing = []
    for name in REQUIRED_DEPENDENCIES:
        try:
            __import__(name)
        except Exception as error:  # noqa: BLE001
            missing.append(f"{name}: {type(error).__name__}: {error}")
            print(f"  缺失  {name}: {error}")
        else:
            print(f"  已安装 {name}")
    if missing:
        print(f"\n缺少 {len(missing)} 个依赖，请执行：")
        print("  pip install -r requirements-flashsr.txt")
        return False
    return True


def check_import() -> object | None:
    print("\n=== [2/3] 上游源码导入 ===")
    sys.path.insert(0, str(PACKAGE_ROOT))
    try:
        from backend import import_flashsr
    except Exception as error:  # noqa: BLE001
        print(f"  导入 backend 失败: {type(error).__name__}: {error}")
        return None
    start = time.perf_counter()
    try:
        flashsr_cls = import_flashsr()
    except Exception as error:  # noqa: BLE001
        print(f"  导入 FlashSR 失败: {error}")
        return None
    print(f"  导入成功，耗时 {time.perf_counter() - start:.2f}s")
    print(f"  {flashsr_cls.__module__}.{flashsr_cls.__name__}")
    _report_stubs()
    return flashsr_cls


def _report_stubs() -> None:
    """确认绘图依赖确实被替身取代，而不是被真实加载。"""
    print("  替身状态：")
    for name in STUBBED_MODULES:
        module = sys.modules.get(name)
        if module is None:
            print(f"    {name:20s} 未加载")
            continue
        spec = getattr(module, "__spec__", None)
        is_stub = spec is not None and spec.loader is None
        print(f"    {name:20s} {'替身' if is_stub else '真实模块'}")


def check_weights(flashsr_cls: object, model_dir: Path, device_name: str) -> bool:
    print(f"\n=== [3/3] 权重加载与前向（device={device_name}）===")
    paths = [model_dir / name for name in WEIGHT_NAMES]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        print("  权重未就绪，跳过：")
        for path in absent:
            print(f"    缺失 {path}")
        print("\n  请在应用界面点击「安装模型」（model_id=flashsr），或执行：")
        print("  python python/audio_ai/model_manager.py --model-id flashsr")
        return False

    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        print("  CUDA 不可用，回退到 cpu")
        device_name = "cpu"
    device = torch.device(device_name)

    sys.path.insert(0, str(PACKAGE_ROOT))
    from backend import (
        FLASHSR_CHUNK_SAMPLES,
        FLASHSR_SAMPLE_RATE,
        DEFAULT_NUM_STEPS,
        load_flashsr,
    )

    start = time.perf_counter()
    try:
        model = load_flashsr(tuple(paths), device)
    except Exception as error:  # noqa: BLE001
        print(f"  加载权重失败: {type(error).__name__}: {error}")
        return False
    print(f"  权重加载完成，耗时 {time.perf_counter() - start:.2f}s")

    block = torch.randn(1, FLASHSR_CHUNK_SAMPLES, device=device)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(block, num_steps=DEFAULT_NUM_STEPS, lowpass_input=False)
    except Exception as error:  # noqa: BLE001
        print(f"  前向失败: {type(error).__name__}: {error}")
        return False
    elapsed = time.perf_counter() - start

    print(f"  前向完成，耗时 {elapsed:.2f}s")
    print(f"  输入形状 {tuple(block.shape)} -> 输出形状 {tuple(output.shape)}")
    duration = FLASHSR_CHUNK_SAMPLES / FLASHSR_SAMPLE_RATE
    print(f"  实时倍率：{duration / elapsed:.2f}x（{duration:.2f}s 音频 / {elapsed:.2f}s）")

    if tuple(output.shape) != tuple(block.shape):
        print("  输出形状与输入不一致，请检查模型约束")
        return False
    if not bool(torch.isfinite(output).all()):
        print("  输出包含 NaN 或 Inf")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="FlashSR 环境自检")
    parser.add_argument("--weights", action="store_true", help="额外加载权重并做一次前向")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    if not check_dependencies():
        return 1
    flashsr_cls = check_import()
    if flashsr_cls is None:
        return 1
    if not args.weights:
        print("\n未指定 --weights，跳过权重检查。依赖与导入均正常。")
        return 0
    device_name = "cuda" if args.device == "auto" and _cuda_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    return 0 if check_weights(flashsr_cls, Path(args.model_dir), device_name) else 1


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
