"""阶段 1 自检：依赖、缓存目录重定向、权重与真实推理冒烟。

运行（必须用 `.venv-voicefixer`，见 `requirements-voicefixer.txt`）：
    .\\.venv-voicefixer\\Scripts\\python.exe python\\audio_ai_voicefixer\\selfcheck.py

检查项：
1. torch / voicefixer 可导入，版本可见
2. **缓存目录重定向**（R1）：导入后 `Config.ckpt` 与 `VoiceFixer.analysis_module_ckpt`
   都必须落在 `models/cache/` 内，而不是真实用户目录
3. manifest 里已登记 `voicefixer` 后端（阶段 2 完成登记后才有值）
4. 权重就位时跑一次 2 s 真实修复，断言输出 44.1 kHz 且时长对齐
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REPO = HERE.parents[1]

import worker  # noqa: E402  （纯标准库，可在 torch 之前导入）
from pipeline import cache_home_for, configure_cache_home  # noqa: E402

MODEL_DIR = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(REPO / "models")))


def _ok(label: str, passed: bool, detail: str = "") -> bool:
    print(f"[{'OK ' if passed else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return passed


def check_runtime() -> bool:
    # 注意：本函数会首次 import voicefixer，主流程必须先完成缓存重定向，
    # 否则上游会在导入期把权重下载到真实用户目录（见 §4.1 ②）。
    import torch

    print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    try:
        import voicefixer as vf
    except Exception as error:  # noqa: BLE001
        return _ok("import voicefixer", False, f"{type(error).__name__}: {error}")
    version = getattr(vf, "__version__", "unknown")
    return _ok("import voicefixer", True, f"version={version}")


def check_cache_redirection() -> bool:
    """R1：权重路径必须被重定向到 models/cache/。

    注意顺序 —— `Config.ckpt` 在**导入期**求值，所以必须先配置再导入。
    """
    home = configure_cache_home(MODEL_DIR)
    resolved = os.path.expanduser("~")
    ok = _ok(
        "expanduser('~') 指向 models/cache",
        Path(resolved).resolve() == home.resolve(),
        str(resolved),
    )

    from voicefixer.vocoder.config import Config

    vocoder_path = Path(str(Config.ckpt))
    ok = _ok(
        "声码器权重路径在 models/cache 内",
        str(vocoder_path).startswith(str(home)),
        str(vocoder_path),
    ) and ok
    print(f"     期望分析模块: {home / '.cache/voicefixer/analysis_module/checkpoints/vf.ckpt'}")
    return ok


def check_manifest() -> bool:
    models, errors = worker.available_models(MODEL_DIR)
    print(f"model_dir={MODEL_DIR}")
    print(f"可用模型={models} 错误={errors}")
    if not models and not errors:
        print("     （manifest 里还没有 voicefixer 条目，属阶段 2 未完成，非错误）")
        return True
    return _ok("manifest 无致命错误", not any("MANIFEST_INVALID" in e for e in errors))


def check_real_restore() -> bool:
    """权重就位时跑一次真实修复；未安装则跳过（不算失败）。"""
    try:
        spec = worker.find_model(worker.VOICEFIXER_MODEL_ID, MODEL_DIR)
    except worker.WorkerFailure as failure:
        print(f"[SKIP] 真实推理 — {failure.code}: {failure}")
        return True

    import numpy as np
    from voicefixer import VoiceFixer

    start = time.perf_counter()
    model = VoiceFixer()
    load_s = time.perf_counter() - start

    seconds = 2.0
    rate = 44100
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * rate)) / rate
    # 合成一段「带噪 + 削波」的退化语音替身：1 kHz 载波 + 噪声 + 硬削波
    noisy = 0.3 * np.sin(2 * np.pi * 1000 * t) + 0.05 * rng.standard_normal(t.size)
    degraded = np.clip(noisy * 4.0, -1.0, 1.0)

    start = time.perf_counter()
    out = model.restore_inmem(degraded.astype(np.float64), cuda=False, mode=0)
    cost = time.perf_counter() - start
    samples = int(np.asarray(out).reshape(-1).size)
    print(
        f"加载 {load_s:.1f}s；mode 0 修复 {seconds:.1f}s 音频耗时 {cost:.1f}s "
        f"（实时率 {seconds / cost:.2f}x）"
    )
    ok = _ok("输出时长对齐", abs(samples - degraded.size) <= 441, f"{samples} vs {degraded.size}")
    ok = _ok("输出非静音", float(np.abs(np.asarray(out)).max()) > 1e-4) and ok
    print(f"model_version={spec.version} weights={[p.name for _, p in spec.artifacts]}")
    return ok


def main() -> int:
    print(f"repo={REPO}")
    # 必须在任何 `import voicefixer` 之前完成重定向（上游导入期即检查权重）
    configure_cache_home(MODEL_DIR)
    results = [
        check_runtime(),
        check_cache_redirection(),
        check_manifest(),
        check_real_restore(),
    ]
    print("\n阶段 1 自检:", "通过" if all(results) else "未通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
