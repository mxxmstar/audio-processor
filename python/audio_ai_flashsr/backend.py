"""FlashSR 后端：上游源码引导、导入期兼容与模型加载。

上游 `FlashSR_Inference` 不是 PyPI 包，以源码快照形式放在 `vendor/` 下。
它在导入期会做三件对本项目有害的事：向 stdout 打印诊断信息、导入训练期
可视化依赖（matplotlib / sklearn）、以及通过 `torch.load` 读取权重。
本模块把这些副作用收敛到一处，供 `worker.py` 在受控环境中使用。
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import io
import sys
import threading
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = MODULE_ROOT / "vendor" / "FlashSR_Inference"

#: 模型硬限制：单次前向只接受 245760 个样本（48 kHz 下 5.12 秒）。
FLASHSR_CHUNK_SAMPLES = 245_760
FLASHSR_SAMPLE_RATE = 48_000
DEFAULT_NUM_STEPS = 1
DEFAULT_OVERLAP_SECONDS = 0.5

#: FlashSR 构造函数要求三个权重，顺序固定。清单 `files[]` 中的 `name` 必须
#: 与之一致；取权重时按名称挑选，不依赖清单声明顺序。
WEIGHT_NAMES = ("student_ldm", "sr_vocoder", "vae")

#: 上游模型按 `[B, T]` 处理，第 0 维是 batch 而非声道。立体声即 batch=2，
#: 一次前向同时处理两个声道；超过 2 声道未定义，须在解码阶段下混。
MAX_CHANNELS = 2

#: `lowpass_input` 默认值。
#:
#: 上游 `FlashSR.forward` 的默认值是 `True`，但其官方 `Example.py` 用的是
#: `False`。该路径存在与 AudioSR 相同的 Nyquist 陷阱：
#: `UtilAudioLowPassFilter.lowpass` 只在 `cutoff_freq == sr` 时减 1（第 25 行），
#: 不防 `cutoff_freq == nyq`；而第 47 行 `hi = highcutoff_freq / nyq` 一旦达到
#: 1.0，scipy 会拒绝（`Digital filter critical frequencies must be 0 < Wn < 1`）。
#: 在能用真实权重验证之前，沿用上游示例的安全取值。
DEFAULT_LOWPASS_INPUT = False


class BackendError(Exception):
    """后端不可用：缺少依赖、权重缺失或加载失败。"""


@contextlib.contextmanager
def torch_load_fallback() -> Iterator[None]:
    """临时包裹 `torch.load`，在 `weights_only` 拦截时自动回退。

    上游 `FlashSR.__init__` **在自己内部**调用 `torch.load`，我们无法从
    外部替换加载逻辑，只能临时打补丁 —— 与 `audio_ai/worker.py` 对
    `download_checkpoint` 的做法一致：仅在构造期间生效，`finally` 中恢复。

    torch 2.6 起 `torch.load` 默认 `weights_only=True`。实测纯
    `OrderedDict[str, Tensor]` 可正常加载，但权重里若混有 numpy 标量等
    对象会抛 `UnpicklingError`。HF 对三个权重的 pickle 扫描结果均在白名单
    内，预期走默认路径；这里保留回退，避免上游换权重后直接崩溃。
    """
    import torch

    original = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        # 权重以 CUDA 设备张量保存时，CPU 环境必须显式 map_location='cpu'，
        # 否则 torch 在反序列化 CUDA storage 时抛
        # "Attempting to deserialize object on a CUDA device but
        # torch.cuda.is_available() is False" 并卡死（阶段 6 实测）。
        if "map_location" not in kwargs and not torch.cuda.is_available():
            kwargs["map_location"] = torch.device("cpu")
        try:
            return original(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - torch 抛出的类型随版本变化
            if not _is_weights_only_error(error):
                raise
            print(
                "torch.load 被 weights_only 拦截，回退 weights_only=False",
                file=sys.stderr,
                flush=True,
            )
            kwargs["weights_only"] = False
            return original(*args, **kwargs)

    torch.load = load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def _is_weights_only_error(error: BaseException) -> bool:
    return type(error).__name__ in {"UnpicklingError", "PickleError"} or (
        "weights_only" in str(error).lower()
    )


def load_flashsr(paths: tuple[Path, ...], device: "torch.device") -> Any:
    """构造并加载 FlashSR 模型。

    `paths` 必须按上游构造函数的顺序给出三个权重：
    `(student_ldm.pth, sr_vocoder.pth, vae.pth)`。
    """
    import torch

    if len(paths) != 3:
        raise BackendError(f"FlashSR 需要 3 个权重文件，收到 {len(paths)} 个")
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise BackendError("权重文件缺失: " + ", ".join(missing))

    flashsr_cls = import_flashsr()
    buffer = io.StringIO()
    try:
        # 构造函数内部直接 torch.load，导入与实例化都可能向 stdout 打印。
        with contextlib.redirect_stdout(buffer), torch_load_fallback():
            model = flashsr_cls(*[str(path) for path in paths])
            model = model.to(device)
            model.eval()
    except BackendError:
        raise
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise BackendError(f"显存不足: {error}") from error
        raise BackendError(f"模型加载失败: {error}") from error
    except Exception as error:  # noqa: BLE001
        raise BackendError(f"模型加载失败: {type(error).__name__}: {error}") from error
    finally:
        noise = buffer.getvalue()
        if noise.strip():
            print(noise.rstrip(), file=sys.stderr, flush=True)
    if not isinstance(model, torch.nn.Module):
        raise BackendError("FlashSR 构造函数未返回 nn.Module")
    return model


_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[tuple[str, ...], str], Any] = {}


def cached_flashsr(paths: tuple[Path, ...], device) -> Any:
    """按 `(权重路径, 设备)` 缓存已加载的模型。

    Worker 是单任务进程，但 `prepare` 与推理线程可能先后请求同一模型；
    缓存键带上设备，避免切换 device 后复用错位的权重。
    """
    key = (tuple(str(Path(path).resolve()) for path in paths), str(device))
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = load_flashsr(paths, device)
            _MODEL_CACHE[key] = model
        return model


def infer_flashsr(
    model: Any,
    block: np.ndarray,
    device,
    num_steps: int = DEFAULT_NUM_STEPS,
    lowpass_input: bool = DEFAULT_LOWPASS_INPUT,
) -> np.ndarray:
    """对一个定长块做一次前向。

    `block` 形状为 `(channels, FLASHSR_CHUNK_SAMPLES)`，channels ∈ {1, 2}。
    上游把第 0 维当作 batch，因此立体声一次前向即可完成，无需逐声道调用。

    返回与 `block` 同形状的 float32 数组。
    """
    import torch

    if block.ndim != 2 or block.shape[0] < 1 or block.shape[0] > MAX_CHANNELS:
        raise BackendError(f"unsupported block shape: {tuple(block.shape)}")
    if block.shape[1] != FLASHSR_CHUNK_SAMPLES:
        raise BackendError(
            f"FlashSR requires exactly {FLASHSR_CHUNK_SAMPLES} samples, got {block.shape[1]}"
        )

    tensor = torch.from_numpy(np.ascontiguousarray(block)).to(device)
    try:
        # 上游与 diffusers 调度器可能向 stdout 打印，而 stdout 属协议专用。
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            output = model(tensor, num_steps=num_steps, lowpass_input=lowpass_input)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise BackendError(f"显存不足: {error}") from error
        raise BackendError(f"推理失败: {error}") from error
    except Exception as error:  # noqa: BLE001
        raise BackendError(f"推理失败: {type(error).__name__}: {error}") from error

    if not isinstance(output, torch.Tensor):
        raise BackendError("FlashSR 输出不是 torch.Tensor")
    result = output.detach().float().cpu().numpy()
    if result.shape != block.shape:
        raise BackendError(
            f"FlashSR 输出形状 {tuple(result.shape)} 与输入 {tuple(block.shape)} 不一致"
        )
    if not np.isfinite(result).all():
        raise BackendError("FlashSR 输出包含 NaN 或 Inf")
    return np.ascontiguousarray(result, dtype=np.float32)


def bootstrap_vendor_path() -> Path:
    """把上游源码快照加入 `sys.path`。

    `FlashSR` 与 `TorchJaekwon` 是上游仓库顶层的两个包，必须把快照根目录
    整体加入搜索路径，只加 `FlashSR/` 会导致 `TorchJaekwon` 导入失败。
    """
    if not VENDOR_ROOT.is_dir():
        raise BackendError(f"上游源码快照缺失: {VENDOR_ROOT}")
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return VENDOR_ROOT


def ensure_inference_only_imports() -> None:
    """为推理路径屏蔽上游的训练期绘图依赖，省掉 `matplotlib` 与 `sklearn`。

    上游有四处顶层导入会拖慢启动、并引入体积可观的依赖：

    | 位置 | 顶层导入 | 实际用途 |
    |---|---|---|
    | `TorchJaekwon/Util/UtilTorch.py:10-11` | `matplotlib.pyplot`、`sklearn.manifold.TSNE` | 仅 `tsne_plot()`（训练期 t-SNE 绘图；方法体内还引用了模块级并未导入的 `pandas`，属上游遗留代码） |
    | `TorchJaekwon/Util/UtilAudioSTFT.py:6,10` | `matplotlib.pyplot`、`librosa.display` | 仅 `specshow()` 等频谱绘图方法 |
    | `TorchJaekwon/Util/UtilAudioMelSpec.py:9` | `librosa.display` | **导入后从未使用** |
    | `FlashSR/BigVGAN/utils.py:9-10` | `matplotlib.use("Agg")`、`matplotlib.pylab` | 仅 `plot_spectrogram()`；但它被 `FlashSR.SRVocoder` 直接导入，位于推理主链路上 |

    其中 `matplotlib` 尤其值得屏蔽：它首次被导入时会构建字体缓存，实测在
    本机耗时约 30 秒，而 Rust 侧的 ready 握手超时只有 3 秒（见
    `ai_worker.rs` 的 `READY_TIMEOUT`）。用户机器上首次启动若因此超时，
    表现就是"AI 运行时不可用"。

    注意：`librosa.display` 必须一并打桩。它内部
    `from matplotlib import colormaps`，若只桩 `matplotlib` 会抛
    `ImportError: cannot import name 'colormaps' from 'matplotlib'`。
    `librosa` 本体仍是真实依赖（`librosa.filters.mel` 被推理使用），
    只替换 `display` 子模块。

    这里在 `sys.modules` 中预置最小替身：`from X.Y import Z` 会先命中
    `sys.modules["X.Y"]`，从而跳过真实包的加载。替身只在目标尚未被加载时
    注入，绝不覆盖其他代码已经导入的真实模块。
    """
    _install_stub("matplotlib", extra={"use": lambda *args, **kwargs: None})
    _install_stub("matplotlib.pyplot")
    _install_stub("matplotlib.pylab")
    _install_stub(
        "sklearn.manifold", extra={"TSNE": _unavailable("sklearn.manifold.TSNE")}
    )
    _install_stub(
        "librosa.display", extra={"specshow": _unavailable("librosa.display.specshow")}
    )


def _install_stub(name: str, extra: dict[str, object] | None = None) -> None:
    """若 `name` 尚未被加载，则在 `sys.modules` 中放入最小替身。

    替身必须带合法的 `__spec__`：上游内联的 diffusers 会用
    `importlib.util.find_spec("matplotlib")` 探测依赖，而 `find_spec` 对
    已在 `sys.modules` 中的模块直接返回其 `__spec__`，缺失时抛
    `ValueError: matplotlib.__spec__ is None`。
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


def _unavailable(name: str):
    def factory(*args, **kwargs):
        raise BackendError(f"{name} 在推理路径不可用")

    return factory


def import_flashsr():
    """导入上游 `FlashSR` 类，并吞掉它在导入期写向 stdout 的诊断输出。

    实测上游在导入时会打印三行文本：`There is no Hparams`、
    `import error: torch`（实际指 torchaudio）、`import error: pydub`。
    本 Worker 的 stdout 被 JSONL 协议独占，混入任何非协议文本都会让
    Rust 侧解析失败，因此必须重定向到 stderr。
    """
    bootstrap_vendor_path()
    ensure_inference_only_imports()
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            from FlashSR.FlashSR import FlashSR
    except Exception as error:  # 上游导入可能因缺失原生依赖而失败
        raise BackendError(f"无法导入 FlashSR: {type(error).__name__}: {error}") from error
    finally:
        noise = buffer.getvalue()
        if noise.strip():
            print(noise.rstrip(), file=sys.stderr, flush=True)
    return FlashSR
