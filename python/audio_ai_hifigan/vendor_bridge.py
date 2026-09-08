"""HiFi-GAN vendor 导入桥接与薄封装（阶段 0 shim）。

仓库内已 vendor 的 HiFi-GAN 实现位于
`python/audio_ai_flashsr/vendor/FlashSR_Inference/`，其导入布局是训练仓库的
`HParams` / `DataProcess.Util.UtilAudioMelSpec` /
`Model.vocoder.hifigan.env` / `Model.vocoder.hifigan.models`，
与 vendor 实际布局（`TorchJaekwon` / `FlashSR`）不符，直接
`import UtilHiFiGanWrapper` 会 `ModuleNotFoundError`（实施计划 P2）。

本模块把训练仓库的导入路径**桥接**到 vendor 实际布局，并提供一个只依赖真实
vendor 模块的薄封装 `HiFiGanRunner`，从而：

1. 不改动 vendor 源码（与 FlashSR `backend.py` 的只读约定一致）。
2. 隔离 HiFi-GAN 与 FlashSR 超分流水线（方案 A）。
3. 屏蔽上游训练期绘图依赖（matplotlib / sklearn / librosa.display），
   复用 FlashSR 已验证的打桩逻辑，避免拖慢启动、并在无这些包时也能导入。
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import io
import sys
import types
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parent
#: vendor 快照与 audio_ai_flashsr 平级：../audio_ai_flashsr/vendor/FlashSR_Inference
VENDOR_ROOT = MODULE_ROOT.parent / "audio_ai_flashsr" / "vendor" / "FlashSR_Inference"


def bootstrap_vendor_path() -> Path:
    """把上游源码快照根目录加入 `sys.path`。

    `FlashSR` 与 `TorchJaekwon` 是上游仓库顶层的两个包，必须把快照根目录整体
    加入搜索路径；只加 `FlashSR/` 会导致 `TorchJaekwon` 导入失败。
    """
    if not VENDOR_ROOT.is_dir():
        raise RuntimeError(f"上游源码快照缺失: {VENDOR_ROOT}")
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return VENDOR_ROOT


def _install_stub(name: str, extra: dict[str, object] | None = None) -> None:
    """若 `name` 尚未被加载，则在 `sys.modules` 中放入最小替身。

    替身必须带合法的 `__spec__`：上游内联的 diffusers 会用
    `importlib.util.find_spec("matplotlib")` 探测依赖，而 `find_spec` 对已在
    `sys.modules` 中的模块直接返回其 `__spec__`，缺失时抛
    `ValueError: matplotlib.__spec__ is None`。

    仅注入尚未加载的模块，绝不覆盖其他代码已导入的真实模块。
    """
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for attribute, value in (extra or {}).items():
        setattr(module, attribute, value)
    sys.modules[name] = module
    parent, _, own = name.rpartition(".")
    if parent and parent in sys.modules and not getattr(sys.modules[parent], own, None):
        setattr(sys.modules[parent], own, module)


def _install_pkg_stub_if_missing(
    name: str, submodules: dict[str, dict[str, object]]
) -> None:
    """若 `name` 包整体不可导入，则注入最小替身，仅用于让导入路径可解析。

    上游某些模块顶层 `import scipy` / `import librosa`，但其推理真正用到的函数
    （如 `scipy.signal.resample_poly`、`librosa.filters.mel`）在运行期才调用，
    非导入期。官方环境 `.venv-flashsr` 已装这些依赖，无需落桩；仅在当前环境
    缺失时注入占位，保证导入解析（P2 验证）不依赖这些包的本体。
    """
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = pkg
    for sub, attrs in submodules.items():
        sub_mod = types.ModuleType(f"{name}.{sub}")
        sub_mod.__spec__ = importlib.machinery.ModuleSpec(f"{name}.{sub}", loader=None)
        for attr, value in attrs.items():
            setattr(sub_mod, attr, value)
        sys.modules[f"{name}.{sub}"] = sub_mod
        setattr(pkg, sub, sub_mod)


def ensure_inference_only_imports() -> None:
    """为推理路径屏蔽上游训练期绘图依赖，省掉 matplotlib 与 sklearn。

    `UtilAudioMelSpec` / `UtilAudioSTFT` 顶层导入 `matplotlib.pyplot` 与
    `librosa.display`，仅服务于频谱绘图。其中 `librosa.display` 必须一并打桩
    —— 它内部 `from matplotlib import colormaps`，只桩 `matplotlib` 会抛
    `ImportError: cannot import name 'colormaps' from 'matplotlib'`。`librosa`
    本体仍是真实依赖（`librosa.filters.mel` 被 mel 提取使用），只替换 `display`
    子模块；若 librosa / scipy 整体缺失则注入最小替身，便于导入解析。
    """
    _install_stub("matplotlib", extra={"use": lambda *a, **k: None})
    _install_stub("matplotlib.pyplot")
    _install_stub("matplotlib.pylab")
    _install_stub(
        "sklearn.manifold", extra={"TSNE": _unavailable("sklearn.manifold.TSNE")}
    )
    try:
        import librosa  # noqa: F401

        _install_stub(
            "librosa.display",
            extra={"specshow": _unavailable("librosa.display.specshow")},
        )
    except ImportError:
        _install_pkg_stub_if_missing(
            "librosa",
            {
                "filters": {"mel": lambda *a, **k: None},
                "display": {"specshow": _unavailable("librosa.display.specshow")},
            },
        )
    # UtilAudio 顶层 `from scipy.signal import resample_poly`：运行期才用，
    # 导入期只需可解析。
    _install_pkg_stub_if_missing("scipy", {"signal": {"resample_poly": lambda *a, **k: None}})


def _unavailable(name: str):
    def factory(*args, **kwargs):
        raise RuntimeError(f"{name} 在推理路径不可用")

    return factory


def _alias_module(dotted_path: str, target: types.ModuleType) -> None:
    """把真实模块对象挂到训练仓库的导入路径下，并补全中间的包。

    例如 `_alias_module("Model.vocoder.hifigan.models", real_models)` 会创建
    `Model` / `Model.vocoder` / `Model.vocoder.hifigan` 三个空包，并把
    `Model.vocoder.hifigan.models` 指向 `target`。
    """
    if dotted_path in sys.modules:
        return
    parts = dotted_path.split(".")
    # 逐级创建并挂接父包
    for depth in range(1, len(parts)):
        parent_path = ".".join(parts[:depth])
        if parent_path not in sys.modules:
            pkg = types.ModuleType(parent_path)
            pkg.__spec__ = importlib.machinery.ModuleSpec(parent_path, loader=None)
            sys.modules[parent_path] = pkg
        else:
            pkg = sys.modules[parent_path]
        child = parts[depth]
        if not getattr(pkg, child, None):
            setattr(pkg, child, sys.modules.get(parent_path))
    sys.modules[dotted_path] = target
    # 把末级名字挂到其父包的命名空间，便于 `from ... import`
    parent_path = ".".join(parts[:-1])
    if parent_path in sys.modules:
        setattr(sys.modules[parent_path], parts[-1], target)


def install_vendor_aliases() -> None:
    """把训练仓库导入路径桥接到 vendor 实际布局（解决 P2）。

    覆盖 `UtilHiFiGanWrapper.py` 的四个错误导入：

    - `HParams` → 推理期仅需占位类（本后端用薄封装，不直接构造 HParams）。
    - `DataProcess.Util.UtilAudioMelSpec` → `TorchJaekwon.Util.UtilAudioMelSpec`
    - `Model.vocoder.hifigan.env` → 提供 `AttrDict` 的占位包
    - `Model.vocoder.hifigan.models` → `FlashSR.AudioSR.hifigan.models`

    调用后 `import UtilHiFiGanWrapper` 即可解析（但 `UtilHiFiGanWrapper` 的
    实例化会读权重、属后续阶段，这里只保证导入可行）。
    """
    bootstrap_vendor_path()
    ensure_inference_only_imports()

    # 必须先建立 wrapper 顶层导入所依赖的全部别名，再导入 wrapper 本体。
    # wrapper 依赖：HParams / DataProcess.Util.UtilAudioMelSpec /
    # Model.vocoder.hifigan.env / Model.vocoder.hifigan.models。

    # 1) HParams 占位（推理薄封装不依赖训练仓库 HParams 结构）
    if "HParams" not in sys.modules:
        hparams_pkg = types.ModuleType("HParams")
        hparams_pkg.__spec__ = importlib.machinery.ModuleSpec("HParams", loader=None)

        class HParams:  # type: ignore[no-redef]
            """占位：仅用于让训练仓库导入路径可解析。"""

        hparams_pkg.HParams = HParams
        sys.modules["HParams"] = hparams_pkg

    # 2) Model.vocoder.hifigan.env —— 提供 AttrDict
    env_pkg = types.ModuleType("Model.vocoder.hifigan.env")
    env_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "Model.vocoder.hifigan.env", loader=None
    )

    class AttrDict(dict):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    env_pkg.AttrDict = AttrDict
    sys.modules["Model.vocoder.hifigan.env"] = env_pkg

    # 3) 真实 vendor 模块
    import FlashSR.AudioSR.hifigan.models as real_models  # noqa: F401
    import TorchJaekwon.Util.UtilAudioMelSpec as real_mel  # noqa: F401
    _alias_module("Model.vocoder.hifigan.models", real_models)
    _alias_module("DataProcess.Util.UtilAudioMelSpec", real_mel)

    # 4) 现在 wrapper 的四个顶层导入都已可解析，再导入并注册别名
    import TorchJaekwon.Util.UtilHiFiGanWrapper as real_wrapper  # noqa: F401
    # 训练仓库里 `UtilHiFiGanWrapper` 位于顶层 `Util/`，vendor 里在
    # `TorchJaekwon/Util/`；注册别名让 `import UtilHiFiGanWrapper` 也能解析。
    _alias_module("UtilHiFiGanWrapper", real_wrapper)


def get_default_config() -> dict[str, Any]:
    """返回 48 kHz 配置（直出 48k，与 App 其它后端对齐）。

    来源：`FlashSR.AudioSR.Vocoder.get_vocoder_config_48k()`。若只能拿到 22.05k
    公开 checkpoint，则改用 `get_config(22050)` 并在下游重采样（见计划 §3.4）。
    """
    return get_config(48000)


def get_config(native_rate: int) -> dict[str, Any]:
    """按权重原生采样率返回 HiFi-GAN 配置。

    - 48000：原生 48k（FlashSR 复用检查点即此档），直出 48k，无需重采样。
    - 22050：jik876 UNIVERSAL_LJSPEECH 公开 checkpoint（标准 hifigan_universal
      配置，80 mel / hop 256）；重建后由 `pipeline.enhance` 重采样到 48k（§3.4）。
    """
    bootstrap_vendor_path()
    ensure_inference_only_imports()
    if native_rate == 48000:
        from FlashSR.AudioSR.Vocoder import get_vocoder_config_48k

        return get_vocoder_config_48k()
    if native_rate == 22050:
        # jik876 hifigan_universal：与 UNIVERSAL_LJSPEECH / LJSpeech 训练配置一致
        # （Generator 用同款架构，mel 用相同参数提取，因此可直接加载该 checkpoint）。
        return {
            "resblock": "1",
            "num_mels": 80,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 8000,
            "upsample_rates": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        }
    raise ValueError(f"unsupported HiFi-GAN native sample rate: {native_rate}")


class _AttrDict(dict):
    """配置 dict 的属性访问包装，供 `Generator.__init__` 读取超参。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def build_generator(
    config: dict[str, Any] | None = None,
    device: Any = None,
    remove_weight_norm: bool = True,
):
    """基于 vendor `Generator`（`models.py`，对应 48k 配置）构造生成器。

    `config` 默认取 48k 配置。返回未加载权重的生成器（随机初始化），适用于
    前向可行性验证；真实权重由 `HiFiGanRunner.load_model` 加载。
    """
    import torch

    bootstrap_vendor_path()
    ensure_inference_only_imports()
    from FlashSR.AudioSR.hifigan.models import Generator

    cfg = config if config is not None else get_default_config()
    # Generator 用属性访问读超参，dict → AttrDict
    h = _AttrDict(cfg)
    generator = Generator(h)
    if remove_weight_norm:
        generator.remove_weight_norm()
    if device is not None:
        generator.to(torch.device(device))
    generator.eval()
    return generator


class HiFiGanRunner:
    """HiFi-GAN 薄封装：mel 提取 → 生成器前向 → 写盘。

    只依赖 vendor 实际布局（FlashSR.AudioSR.hifigan / TorchJaekwon），不经由
    训练仓库导入路径；阶段 1 的 Worker 在收到 `process` 后再构造本对象。
    """

    def __init__(self, config: dict[str, Any] | None = None, device: Any = None):
        import torch

        self.config = config if config is not None else get_default_config()
        self.device = torch.device(device or "cpu")
        self.generator = None
        self.mel_extractor = None

    def load_model(self, generator_path: str | None = None):
        import torch

        bootstrap_vendor_path()
        ensure_inference_only_imports()
        state = None
        if generator_path:
            state = torch.load(generator_path, map_location=self.device, weights_only=False)
        sd = state["generator"] if isinstance(state, dict) and "generator" in state else state

        # 公开 checkpoint 有两种保存方式，必须先判断再决定如何建图：
        # - 带 weight_norm 保存：键形如 `conv_pre.weight_g` / `conv_pre.weight_v`；
        #   此时若先 remove_weight_norm 再加载会因键名不匹配而失败，须先载入仍带
        #   weight_norm 的生成器，再 remove 将其折叠回 `weight`。
        # - 已 remove_weight_norm 保存（如 jik876 官方 generator）：键为 `*.weight`。
        weight_normed = sd is not None and any(str(k).endswith(".weight_g") for k in sd)
        self.generator = build_generator(
            self.config, self.device, remove_weight_norm=not weight_normed
        )
        if sd is not None:
            self.generator.load_state_dict(sd)
            if weight_normed:
                self.generator.remove_weight_norm()
        self.generator.eval()
        return self

    def audio_to_mel(self, audio):
        """音频 → log-mel（需 librosa；阶段 0 仅前向验证不依赖此路径）。

        `audio` 形状 `[time]` 或 `[batch, time]`，float32，范围 [-1, 1]。
        """
        if self.mel_extractor is None:
            bootstrap_vendor_path()
            ensure_inference_only_imports()
            from TorchJaekwon.Util.UtilAudioMelSpec import UtilAudioMelSpec

            c = self.config
            self.mel_extractor = UtilAudioMelSpec(
                nfft=c["n_fft"],
                hop_size=c["hop_size"],
                sample_rate=c["sampling_rate"],
                mel_size=c["num_mels"],
                frequency_min=c["fmin"],
                frequency_max=c["fmax"],
            )
        return self.mel_extractor.get_hifigan_mel_spec(audio)

    def mel_to_audio(self, mel):
        """mel → 波形。`mel` 形状 `[mel_bins, time]` 或 `[batch, mel_bins, time]`。

        单声道时上游 `audio_to_mel` 返回 `[mel_bins, time]`（二维），这里自动补
        上 batch 维喂给生成器；返回已是 CPU 上的 `(time,)` numpy 数组。
        """
        import torch

        if self.generator is None:
            raise RuntimeError("Generator 未加载：请先调用 load_model()")
        if not torch.is_tensor(mel):
            mel = torch.from_numpy(mel)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)
        with torch.no_grad():
            wav = self.generator(mel)
        return wav.squeeze().cpu().numpy()
