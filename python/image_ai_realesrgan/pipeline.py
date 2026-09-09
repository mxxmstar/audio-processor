"""Real-ESRGAN 推理流水线：权重加载、分块（tiling）超分、进度与协作式取消。

本模块在收到 `process` 命令之后才被 `worker.py` 导入（风险 R12）：顶层
`import torch` 实测约 5+ 秒，若放在启动时执行会超出 Rust 侧 3 秒的
ready 握手超时。清单解析留在 `worker.py`（纯标准库），这样
`from worker import find_model` 不会拉起 torch。

与音频侧（HiFi-GAN 的逐块 mel 生成）对应，图像侧按**空间 tile** 分块：
官方 `RealESRGANer.enhance` 虽支持 `tile`，但是黑盒 —— 无法上报进度、无法
中途取消（计划风险 R6）。因此这里**自研 tile 循环**（照搬 `realesrgan/utils.py`
的 `tile_process` 坐标几何），在每块前检查取消标志、每块后上报进度，并直接把
每块结果写入输出 numpy 缓冲（避免官方 `self.output` 巨幅 float 张量的内存峰值，
缓解 R2）。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2  # noqa: E402

import worker  # noqa: E402  (仅用于 find_model / ModelSpec，不触发 torch 外的重导入)
from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    SCRIPT_ROOT,
    WorkerFailure,
    emit_progress,
    log,
)

#: 进度区间：与音频侧保持一致，前端无需区分后端
PERCENT_LOAD_MODEL = 5.0
PERCENT_INFERENCE_START = 10.0
PERCENT_INFERENCE_SPAN = 80.0
PERCENT_ENCODE = 95.0

#: 默认 tile 边长（像素）。大图按此切分；小图（长边 ≤ 此值）走整图直推路径
#: （计划 §3.5 #5，省掉 pad/拼接开销）。后续可据可用内存自适应下调（R2）。
DEFAULT_TILE = 512
#: tile 重叠边距：每块多取一段上下文送入模型，输出后按原区间裁掉，消除
#: 块边界的感受野伪影（与音频 CONTEXT_PAD 同理）。
TILE_PAD = 10

#: 运行时缓存：同一 (权重路径, 模型名, 倍数, 设备) 只加载一次模型。
_RUNNER_CACHE: dict[tuple[str, str, int, str], Any] = {}
_RUNNER_LOCK = threading.Lock()


def warm_up_imports() -> None:
    """在主线程完成上游导入，避免推理线程内首次 import 的潜在阻塞。

    与音频侧 `pipeline.warm_up_imports` 同构：basicsr 导入期会向 **stderr**
    打印 `torchvision.transforms.functional_tensor` 弃用告警（不影响功能），
    这里统一重定向，避免任何非预期输出污染 JSONL 协议占用的 stdout。
    """
    import contextlib
    import io
    import sys

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            import basicsr  # noqa: F401
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
            import realesrgan  # noqa: F401
            import cv2  # noqa: F401
    finally:
        noise = buffer.getvalue()
        if noise.strip():
            print(noise.rstrip(), file=sys.stderr, flush=True)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA_UNAVAILABLE", "CUDA is not available")
        return torch.device("cuda")
    if requested not in {"auto", "cpu"}:
        raise WorkerFailure("UNSUPPORTED_DEVICE", f"unsupported device: {requested}")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu")


def _build_rrdbnet(model_name: str, scale: int) -> Any:
    """按 model_name 构造对应的 RRDBNet 架构。

    首发仅 x4plus；动漫 6B 与 x2plus 架构参数一并登记，新增后端无需改结构。
    """
    from basicsr.archs.rrdbnet_arch import RRDBNet

    specs = {
        "RealESRGAN_x4plus": (64, 23, 32),
        "RealESRGAN_x4plus_anime_6B": (64, 6, 32),
        "RealESRGAN_x2plus": (64, 23, 32),
    }
    num_feat, num_block, num_grow_ch = specs.get(model_name, (64, 23, 32))
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        scale=scale,
    )


class RealESRGANRunner:
    """封装已加载的 Real-ESRGAN 模型，提供按 tile 分块的上采样。

    `upscale` 内部完全复用上游 `RealESRGANer` 的坐标几何（`tile_process`），
    但每块前检查取消、每块后上报进度，并把结果直接写入 uint8 输出缓冲。
    """

    def __init__(self, model_path: str, model_name: str, scale: int, device: torch.device) -> None:
        import torch  # noqa: F401

        model = _build_rrdbnet(model_name, scale)
        # torch>=2.6 默认 weights_only=True；本权重为 OrderedDict[str, Tensor]，
        # 在白名单内，可直接加载（与 FlashSR 计划 §3.5 结论一致）。
        loadnet = torch.load(model_path, map_location="cpu")
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(device)
        self.scale = scale
        self.device = device
        self.img: Any = None
        self.mod_pad_h = 0
        self.mod_pad_w = 0

    # ------------------------------------------------------------------
    # 预处理：BGR/RGBA/gray/16-bit 归一化为 RGB float 张量，并做 mod_pad
    # ------------------------------------------------------------------

    def pre_process(self, rgb_hwc: np.ndarray) -> None:
        """`rgb_hwc`：HWC float32，值域 [0,1]，RGB 顺序。

        仅做通道转置与 mod_pad（scale=4 时 mod_scale=None，无 pad）；
        pre_pad 设为 0（计划 v1，避免 RGBA 通道对齐复杂度，后续可调）。
        """
        import torch.nn.functional as F

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_hwc, (2, 0, 1)))).float()
        self.img = img.unsqueeze(0).to(self.device)
        self.mod_pad_h = 0
        self.mod_pad_w = 0
        _, _, h, w = self.img.shape
        mod_scale = 2 if self.scale == 2 else (4 if self.scale == 1 else None)
        if mod_scale is not None:
            if h % mod_scale != 0:
                self.mod_pad_h = mod_scale - h % mod_scale
            if w % mod_scale != 0:
                self.mod_pad_w = mod_scale - w % mod_scale
            if self.mod_pad_h or self.mod_pad_w:
                self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    @staticmethod
    def _tensor_to_uint8_bgr(output_tile: Any) -> np.ndarray:
        """模型输出 (1,3,H,W) float → HWC uint8 BGR，clamp 到 [0,1]。"""
        out = output_tile.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))
        return (out * 255.0).round().astype(np.uint8)

    def _crop_mod_pad(self, out_np: np.ndarray) -> np.ndarray:
        h, w = out_np.shape[0], out_np.shape[1]
        return out_np[
            0 : h - self.mod_pad_h * self.scale,
            0 : w - self.mod_pad_w * self.scale,
        ]

    def whole_upscale(self) -> np.ndarray:
        with torch.no_grad():
            output = self.model(self.img)
        out = self._tensor_to_uint8_bgr(output)
        return self._crop_mod_pad(out)

    def tile_upscale(self, tile_size: int, cancel_event: threading.Event, request_id: str) -> np.ndarray:
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        out_np = np.zeros((output_height, output_width, channel), dtype=np.uint8)
        tiles_x = int(np.ceil(width / tile_size))
        tiles_y = int(np.ceil(height / tile_size))
        total = tiles_x * tiles_y
        done = 0
        for y in range(tiles_y):
            for x in range(tiles_x):
                if cancel_event.is_set():
                    raise WorkerFailure("CANCELLED", "处理已取消")
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                in_sx = ofs_x
                in_ex = min(ofs_x + tile_size, width)
                in_sy = ofs_y
                in_ey = min(ofs_y + tile_size, height)
                in_sx_pad = max(in_sx - TILE_PAD, 0)
                in_ex_pad = min(in_ex + TILE_PAD, width)
                in_sy_pad = max(in_sy - TILE_PAD, 0)
                in_ey_pad = min(in_ey + TILE_PAD, height)
                tile_h = in_ey - in_sy
                tile_w = in_ex - in_sx

                input_tile = self.img[:, :, in_sy_pad:in_ey_pad, in_sx_pad:in_ex_pad]
                with torch.no_grad():
                    output_tile = self.model(input_tile)

                ot = self._tensor_to_uint8_bgr(output_tile)
                # 裁掉 pad，定位到无 pad 区域在输出 tile 中的像素坐标
                osy_tile = (in_sy - in_sy_pad) * self.scale
                oey_tile = osy_tile + tile_h * self.scale
                osx_tile = (in_sx - in_sx_pad) * self.scale
                oex_tile = osx_tile + tile_w * self.scale
                dy0 = in_sy * self.scale
                dy1 = in_ey * self.scale
                dx0 = in_sx * self.scale
                dx1 = in_ex * self.scale
                out_np[dy0:dy1, dx0:dx1] = ot[osy_tile:oey_tile, osx_tile:oex_tile]

                done += 1
                emit_progress(
                    request_id,
                    "inference",
                    PERCENT_INFERENCE_START + PERCENT_INFERENCE_SPAN * done / total,
                    processed_tiles=done,
                    total_tiles=total,
                )
        return self._crop_mod_pad(out_np)

    def upscale(
        self,
        img: np.ndarray,
        cancel_event: threading.Event,
        request_id: str,
        out_format: str,
        tile: int | None = None,
    ) -> np.ndarray:
        """对一张已解码的图片做超分，返回 uint8 BGR / BGRA / gray（按需）。

        `img` 来自 `cv2.imread(..., IMREAD_UNCHANGED)`：可为 uint8 或 uint16、
        BGR / BGRA / gray。返回张量在 [0,1] 经 *255 量化；16-bit 输入在返回前
        再量化到 uint16（R9 之外的位深保留）。
        """
        if img.dtype == np.uint16:
            max_range = 65535.0
            img_f = img.astype(np.float32) / max_range
        else:
            max_range = 255.0
            img_f = img.astype(np.float32) / max_range

        if img_f.ndim == 2:
            img_mode = "L"
            rgb = cv2.cvtColor(img_f, cv2.COLOR_GRAY2RGB)
        elif img_f.shape[2] == 4:
            img_mode = "RGBA"
            alpha = img_f[:, :, 3]
            rgb = cv2.cvtColor(img_f[:, :, 0:3], cv2.COLOR_BGR2RGB)
        else:
            img_mode = "RGB"
            rgb = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)

        self.pre_process(rgb)
        if cancel_event.is_set():
            raise WorkerFailure("CANCELLED", "处理已取消")

        # 选择分块尺寸：显式 tile > 0 优先；否则小图整图直推，大图默认 512。
        if tile and tile > 0:
            tile_size = tile
        else:
            h_pix, w_pix = self.img.shape[2], self.img.shape[3]
            tile_size = 0 if max(h_pix, w_pix) <= DEFAULT_TILE else DEFAULT_TILE

        if tile_size and tile_size > 0:
            out = self.tile_upscale(tile_size, cancel_event, request_id)
        else:
            out = self.whole_upscale()

        if img_mode == "L":
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        elif img_mode == "RGBA":
            # alpha 通道用双线性单独放大（与上游 alpha_upsampler='realesrgan'
            # 的 cv2.resize 分支等价；R3：alpha 不参与生成、单独合并）
            alpha_up = cv2.resize(
                alpha, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            out = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)
            out[:, :, 3] = (alpha_up * 255.0).round().astype(np.uint8)
        # 16-bit 回量化为 uint16
        if max_range == 65535.0:
            out = (out.astype(np.float32) / 255.0 * 65535.0).round().astype(np.uint16)
        return out


class SwinIRRunner(RealESRGANRunner):
    """SwinIR 后端：复用 `RealESRGANRunner` 的 tile 分块、预处理与写回框架，
    仅替换网络构造与权重加载。

    SwinIR 与 Real-ESRGAN 的差异：
    - 网络为窗口多头自注意力（SwinIR），要求**输入空间尺寸是 `window_size`(8)
      的倍数**，因此 `pre_process` 改为按 8 对齐做 reflect pad（而非 2/4 的
      mod_scale）。
    - 权重以 `{'params': state_dict}` 或 `{'params_ema': state_dict}` 包裹
      （官方保存格式），需解包；DataParallel 保存时键含 `module.` 前缀，需剥离。
    - 每个权重固定一个上采样倍数（`upscale`），由 `model_name` 查表决定
      `upsampler`（pixelshuffle 用于 classicalSR，nearest+conv 用于真实世界）。
    """

    # model_name -> (上采样倍数, upsampler 类型, 训练 img_size)
    # classicalSR 训练用 img_size=48（文件名 s48w8），realSR 用 img_size=64（s64w8）。
    # 该值仅决定构建期 attn_mask buffer 形状以匹配权重；推理时 forward 按输入
    # 动态重算 attention mask，因此任意尺寸均可处理。
    _PARAMS: dict[str, tuple[int, str, int]] = {
        "swinir_classical_x2": (2, "pixelshuffle", 48),
        "swinir_classical_x3": (3, "pixelshuffle", 48),
        "swinir_classical_x4": (4, "pixelshuffle", 48),
        "swinir_real_x4": (4, "nearest+conv", 64),
    }
    _WINDOW = 8

    def __init__(self, model_path: str, model_name: str, scale: int, device: torch.device) -> None:
        from basicsr.archs.swinir_arch import SwinIR

        if model_name not in self._PARAMS:
            raise WorkerFailure("MODEL_MANIFEST_INVALID", f"unknown SwinIR model: {model_name}")
        upscale, upsampler, img_size = self._PARAMS[model_name]
        model = SwinIR(
            upscale=upscale,
            in_chans=3,
            img_size=img_size,
            window_size=self._WINDOW,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler=upsampler,
            resi_connection="1conv",
        )
        loadnet = torch.load(model_path, map_location="cpu")
        if "params_ema" in loadnet:
            state_dict = loadnet["params_ema"]
        elif "params" in loadnet:
            state_dict = loadnet["params"]
        else:
            state_dict = loadnet
        # DataParallel 保存时键含 `module.` 前缀，需剥离；单卡保存则原样保留。
        state_dict = {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model.to(device)
        self.scale = scale
        self.device = device
        self.img: Any = None
        self.mod_pad_h = 0
        self.mod_pad_w = 0

    def pre_process(self, rgb_hwc: np.ndarray) -> None:
        """与 `RealESRGANRunner.pre_process` 同，但按 SwinIR 习惯归一化到 [0,1]
        并对齐到 `window_size`(8)。

        SwinIR 在 [0,1] 空间训练（img_range=1），因此输入需除以 255；Real-ESRGAN
        直接用 0-255，两者在 `pre_process` 上分叉。
        """
        import torch.nn.functional as F

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_hwc, (2, 0, 1)))).float() / 255.0
        self.img = img.unsqueeze(0).to(self.device)
        self.mod_pad_h = 0
        self.mod_pad_w = 0
        _, _, h, w = self.img.shape
        ws = self._WINDOW
        if h % ws != 0:
            self.mod_pad_h = ws - h % ws
        if w % ws != 0:
            self.mod_pad_w = ws - w % ws
        if self.mod_pad_h or self.mod_pad_w:
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")


    def tile_upscale(self, tile_size: int, cancel_event: threading.Event, request_id: str) -> np.ndarray:
        """SwinIR 专用分块：每块输入先 pad 到 `window_size`(8) 倍数以满足窗口注意力
        约束，输出再裁掉该 pad；重叠（TILE_PAD）、进度与取消逻辑与基类一致。"""
        import torch.nn.functional as F

        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        out_np = np.zeros((output_height, output_width, channel), dtype=np.uint8)
        tiles_x = int(np.ceil(width / tile_size))
        tiles_y = int(np.ceil(height / tile_size))
        total = tiles_x * tiles_y
        done = 0
        for y in range(tiles_y):
            for x in range(tiles_x):
                if cancel_event.is_set():
                    raise WorkerFailure("CANCELLED", "处理已取消")
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                in_sx = ofs_x
                in_ex = min(ofs_x + tile_size, width)
                in_sy = ofs_y
                in_ey = min(ofs_y + tile_size, height)
                in_sx_pad = max(in_sx - TILE_PAD, 0)
                in_ex_pad = min(in_ex + TILE_PAD, width)
                in_sy_pad = max(in_sy - TILE_PAD, 0)
                in_ey_pad = min(in_ey + TILE_PAD, height)
                tile_h = in_ey - in_sy
                tile_w = in_ex - in_sx
                input_tile = self.img[:, :, in_sy_pad:in_ey_pad, in_sx_pad:in_ex_pad]
                # 补齐到 window_size 倍数（SwinIR 窗口注意力要求输入为 8 的倍数）
                _, _, th_, tw_ = input_tile.shape
                ph = (self._WINDOW - th_ % self._WINDOW) % self._WINDOW
                pw = (self._WINDOW - tw_ % self._WINDOW) % self._WINDOW
                if ph or pw:
                    input_tile = F.pad(input_tile, (0, pw, 0, ph))
                with torch.no_grad():
                    output_tile = self.model(input_tile)
                ot = self._tensor_to_uint8_bgr(output_tile)
                # 裁掉 window pad（输出侧）
                if ph or pw:
                    ot = ot[: (th_ + ph) * self.scale, : (tw_ + pw) * self.scale]
                osy_tile = (in_sy - in_sy_pad) * self.scale
                oey_tile = osy_tile + tile_h * self.scale
                osx_tile = (in_sx - in_sx_pad) * self.scale
                oex_tile = osx_tile + tile_w * self.scale
                dy0 = in_sy * self.scale
                dy1 = in_ey * self.scale
                dx0 = in_sx * self.scale
                dx1 = in_ex * self.scale
                out_np[dy0:dy1, dx0:dx1] = ot[osy_tile:oey_tile, osx_tile:oex_tile]
                done += 1
                emit_progress(
                    request_id,
                    "inference",
                    PERCENT_INFERENCE_START + PERCENT_INFERENCE_SPAN * done / total,
                    processed_tiles=done,
                    total_tiles=total,
                )
        return self._crop_mod_pad(out_np)

class GFPGANRunner(RealESRGANRunner):
    """GFPGAN 人脸修复后端：复用进程/协议框架，仅替换网络为 GFPGANer。

    原生输出为修复后的 ×`scale` 人脸图（`paste_back` 贴回原图）；无人脸时回退到
    LANCZOS ×`scale` 放大，保证输出尺寸语义与超分后端一致。倍数缩放仍由
    `enhance` 的二次 LANCZOS 机制处理。
    """

    def __init__(self, model_path: str, model_name: str, scale: int, device: torch.device) -> None:
        from gfpgan import GFPGANer

        self.model = GFPGANer(
            model_path=model_path,
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        self.scale = scale
        self.device = device
        self.img: Any = None
        self.mod_pad_h = 0
        self.mod_pad_w = 0

    def upscale(
        self,
        img: np.ndarray,
        cancel_event: threading.Event,
        request_id: str,
        out_format: str,
        tile: int | None = None,
    ) -> np.ndarray:
        if cancel_event.is_set():
            raise WorkerFailure("CANCELLED", "处理已取消")
        _, _, restored = self.model.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        if restored is None:
            h, w = img.shape[:2]
            return cv2.resize(img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LANCZOS4)
        return restored


def _get_gfpgan_runner(spec: "worker.ModelSpec", device: torch.device) -> GFPGANRunner:
    weight_path = str(spec.artifacts[0][1])
    key = (weight_path, spec.model_name, int(spec.scale), str(device))
    with _RUNNER_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            runner = GFPGANRunner(
                model_path=weight_path,
                model_name=spec.model_name,
                scale=int(spec.scale),
                device=device,
            )
            _RUNNER_CACHE[key] = runner
        return runner


def get_runner(spec: "worker.ModelSpec", device: torch.device) -> RealESRGANRunner:
    """按 (权重路径, 模型名, 倍数, 设备) 缓存已加载的模型，避免重复加载。

    按 `spec.backend` 选择后端：realesrgan 走 `RealESRGANRunner`，swinir 走
    `SwinIRRunner`，gfpgan 走 `GFPGANRunner`；三者接口一致（`upscale` 同签名），
    `enhance` 无需区分。
    """
    if spec.backend == "gfpgan":
        return _get_gfpgan_runner(spec, device)
    if spec.backend == "swinir":
        return _get_swinir_runner(spec, device)
    weight_path = str(spec.artifacts[0][1])
    key = (weight_path, spec.model_name, int(spec.scale), str(device))
    with _RUNNER_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            runner = RealESRGANRunner(
                model_path=weight_path,
                model_name=spec.model_name,
                scale=int(spec.scale),
                device=device,
            )
            _RUNNER_CACHE[key] = runner
        return runner


def _get_swinir_runner(spec: "worker.ModelSpec", device: torch.device) -> SwinIRRunner:
    weight_path = str(spec.artifacts[0][1])
    key = (weight_path, spec.model_name, int(spec.scale), str(device))
    with _RUNNER_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            runner = SwinIRRunner(
                model_path=weight_path,
                model_name=spec.model_name,
                scale=int(spec.scale),
                device=device,
            )
            _RUNNER_CACHE[key] = runner
        return runner


def _write_image(output_path: str, output: np.ndarray, out_format: str, jpg_quality: int | None) -> None:
    if out_format == "jpg":
        # jpg 不支持 alpha 与 16-bit：合并前已在 enhance 内转 BGR/uint8
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality if jpg_quality else 95)]
        ok = cv2.imwrite(output_path, output, params)
    else:
        ok = cv2.imwrite(output_path, output)
    if not ok:
        raise WorkerFailure("ENCODE_FAILED", f"cv2.imwrite failed: {output_path}")


def enhance(request: dict[str, Any], cancel_event: threading.Event) -> dict[str, Any]:
    """执行一次 Real-ESRGAN 超分，返回 `result` 事件载荷。"""
    request_id = str(request.get("request_id", ""))
    input_path = str(request.get("input_path", ""))
    output_path = str(request.get("output_path", ""))
    model_id = str(request.get("model_id", ""))

    if not input_path or not Path(input_path).is_file():
        raise WorkerFailure("UNSUPPORTED_INPUT", f"input image does not exist: {input_path!r}")
    if not output_path or Path(output_path).resolve() == Path(input_path).resolve():
        raise WorkerFailure("INVALID_OUTPUT", "output path must differ from input path")
    if not model_id:
        raise WorkerFailure("MODEL_NOT_FOUND", "model_id is required")

    suffix = output_path.lower()
    if suffix.endswith((".jpg", ".jpeg")):
        out_format = "jpg"
        raw_q = request.get("jpg_quality")
        try:
            jpg_quality = int(raw_q) if raw_q is not None else 95
        except (TypeError, ValueError):
            jpg_quality = 95
    elif suffix.endswith((".png", ".webp", ".bmp")):
        out_format = "png"
        jpg_quality = None
    else:
        # 计划 §3.3：默认 png（无损）；未知后缀统一按 png 落盘
        out_format = "png"
        jpg_quality = None

    model_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models")))
    spec = worker.find_model(model_id, model_dir)
    device = choose_device(str(request.get("device", "auto")))

    emit_progress(request_id, "load_model", PERCENT_LOAD_MODEL, message=f"loading {model_id}")
    runner = get_runner(spec, device)
    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "处理已取消")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise WorkerFailure("UNSUPPORTED_INPUT", f"cannot decode image: {input_path!r}")
    h_in, w_in = img.shape[:2]
    if h_in <= 0 or w_in <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "input image has invalid dimensions")

    outscale = request.get("outscale")
    try:
        outscale = float(outscale) if outscale is not None else float(spec.scale)
    except (TypeError, ValueError):
        outscale = float(spec.scale)

    emit_progress(request_id, "inference", PERCENT_INFERENCE_START, message="超分推理中")
    raw_tile = request.get("tile")
    try:
        tile = int(raw_tile) if raw_tile else None
    except (TypeError, ValueError):
        tile = None

    output = runner.upscale(img, cancel_event, request_id, out_format, tile=tile)
    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "处理已取消")

    # 可选：按 outscale 二次缩放（模型固定倍数为 scale；用户请求其它倍数时做 LANCZOS 重采样）
    if abs(outscale - spec.scale) > 1e-6:
        new_w = max(1, int(round(w_in * outscale)))
        new_h = max(1, int(round(h_in * outscale)))
        interp = cv2.INTER_LANCZOS4
        is_bgra = output.ndim == 3 and output.shape[2] == 4
        if is_bgra:
            bgr = output[:, :, 0:3]
            a = output[:, :, 3]
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
            a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            output = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            output[:, :, 3] = a
        else:
            output = cv2.resize(output, (new_w, new_h), interpolation=interp)
        emit_progress(
            request_id,
            "resample",
            (PERCENT_INFERENCE_START + PERCENT_INFERENCE_SPAN + PERCENT_ENCODE) / 2,
            message=f"重采样到 {int(round(outscale))}x",
        )

    # jpg 不支持 alpha / 16-bit：转 BGR uint8（R3/R5）
    if out_format == "jpg":
        if output.ndim == 3 and output.shape[2] == 4:
            output = output[:, :, 0:3]
        if output.dtype == np.uint16:
            output = (output.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    emit_progress(request_id, "encode", PERCENT_ENCODE, message="写入输出图片")
    _write_image(output_path, output, out_format, jpg_quality)

    written = Path(output_path).stat().st_size if Path(output_path).is_file() else -1
    if not Path(output_path).is_file() or written == 0:
        raise WorkerFailure("OUTPUT_INVALID", "AI output file is empty")

    out_h, out_w = output.shape[:2]
    log(f"realesrgan done: {w_in}x{h_in} -> {out_w}x{out_h} ({out_format})")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "type": "result",
        "status": "completed",
        "output_path": output_path,
        "model_id": model_id,
        "model_version": spec.version,
        "width": int(out_w),
        "height": int(out_h),
        "scale": int(spec.scale),
        "format": out_format,
    }
