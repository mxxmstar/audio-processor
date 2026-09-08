"""阶段 0 可行性自检：在 torch 环境下跑通 Generator 单次前向 + 验证 vendor 导入桥接。

运行（任意带 torch 的 Python；官方环境见 README：
`python/audio_ai_flashsr/README.md` 的 `.venv-flashsr`）：
    python python/audio_ai_hifigan/selfcheck.py

本脚本仅验证模型代码路径（不需要真实权重、不需要 librosa）：
- 用 48k 配置构造 Generator（随机初始化）
- 对随机 mel 跑一次前向，确认输出形状与有限性
- 验证 P2 的 vendor 导入桥接使训练仓库布局 `UtilHiFiGanWrapper` 可导入
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from vendor_bridge import (  # noqa: E402
    bootstrap_vendor_path,
    ensure_inference_only_imports,
    install_vendor_aliases,
    get_default_config,
    build_generator,
)


def _check_forward() -> bool:
    import torch

    cfg = get_default_config()
    print(f"[1] 48k 配置: sampling_rate={cfg['sampling_rate']} "
          f"num_mels={cfg['num_mels']} hop_size={cfg['hop_size']} "
          f"fmax={cfg['fmax']}")

    device = torch.device("cpu")
    t0 = time.perf_counter()
    generator = build_generator(cfg, device, remove_weight_norm=True)
    build_ms = (time.perf_counter() - t0) * 1000.0
    params_m = sum(p.numel() for p in generator.parameters()) / 1e6
    print(f"[2] Generator 构造完成: {params_m:.2f}M 参数, 耗时 {build_ms:.1f} ms")

    mel_time = 200  # 帧
    mel = torch.randn(1, cfg["num_mels"], mel_time)
    t1 = time.perf_counter()
    with torch.no_grad():
        wav = generator(mel)
    fwd_ms = (time.perf_counter() - t1) * 1000.0
    print(f"[3] 前向: mel{mel.shape} -> wav{wav.shape}, 耗时 {fwd_ms:.1f} ms")

    # 校验：上采样倍率 ≈ hop_size（480）；输出时间维度应接近 mel_time * 480
    expect_approx = mel_time * cfg["hop_size"]
    if abs(wav.shape[-1] - expect_approx) > cfg["hop_size"]:
        print(f"    [警告] 输出时长 {wav.shape[-1]} 与预期 ~{expect_approx} 偏差较大")
    if not torch.isfinite(wav).all():
        print("    [失败] 输出含 NaN/Inf")
        return False
    print(f"[4] 输出有限性 OK，幅度范围 "
          f"[{float(wav.min()):.3f}, {float(wav.max()):.3f}]（tanh 限幅 [-1,1]）")
    return True


def _check_import_shim() -> bool:
    print("[5] 验证 P2：训练仓库导入桥接 ...")
    install_vendor_aliases()
    try:
        import UtilHiFiGanWrapper  # noqa: F401  (仅验证导入可行，不实例化)
    except Exception as error:  # noqa: BLE001
        print(f"    [失败] import UtilHiFiGanWrapper 仍报错: {type(error).__name__}: {error}")
        return False
    print("    [OK] `import UtilHiFiGanWrapper` 解析成功（桥接生效）")
    return True


def main() -> int:
    bootstrap_vendor_path()
    ensure_inference_only_imports()
    ok = True
    ok = _check_forward() and ok
    ok = _check_import_shim() and ok
    print("\n阶段 0 可行性:", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
