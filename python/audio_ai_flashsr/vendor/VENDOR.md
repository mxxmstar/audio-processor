# 第三方源码快照（vendored source）

本目录是上游仓库的**只读快照**，随本项目一起入库，用于 FlashSR 推理。
**不要**在此目录内直接修改源码；如需修补，请在本项目的
`python/audio_ai_flashsr/backend.py` 中以 monkey patch 方式处理，并注明原因。

## 来源

| 项目 | 值 |
|---|---|
| 仓库 | https://github.com/jakeoneijk/FlashSR_Inference |
| Commit | `2292814a7ef74f61a5479c8d96e653d2f90f369d` |
| 快照日期 | 2026-09-03 |
| 论文 | FlashSR: One-step Versatile Audio Super-resolution via Diffusion Distillation（arXiv:2501.10807） |
| 权重 | https://huggingface.co/datasets/jakeoneijk/FlashSR_weights |
| License | **上游仓库与权重均未声明 License**（`FlashSRInfer` 的 `setup.py` 中 `license` 为空） |

> 合规提示：许可未明确，权重不随应用分发，仅由用户在界面中显式下载。
> 详见 `docs/FlashSR集成实施计划.md` 风险 R8。

## 快照范围

| 路径 | 说明 |
|---|---|
| `FlashSR/` | FlashSR 模型本体：UNet、VAEWrapper、SRVocoder、BigVGAN、AudioSR 工具函数 |
| `TorchJaekwon/` | 上游的工具与扩散框架，含内联的 DPM-Solver 调度器 |
| `setup.py` / `Readme.md` / `Example.py` | 原始依赖声明与官方示例，仅作参考，不参与运行 |

### 已从快照中剔除

| 路径 | 原因 |
|---|---|
| `Assets/` | 示例图片与示例音频（仓库根目录已有测试音频，无需重复） |
| `FlashSR/BigVGAN/LibriTTS/train-full.txt` | 49.7 MB 训练文件清单，仅被 `BigVGAN/train.py` 的 CLI 默认值与 `parse_scripts/parse_libritts.py` 引用，推理路径不涉及 |
| `Script/` | 上游 conda / PyTorch 安装脚本，本项目使用自有虚拟环境 |
| `.git/` | 快照不需要版本历史，来源与 commit 已记录在上表 |

## 运行时依赖说明

上游 `setup.py` 声明了 9 个依赖，其中 `wandb`、`tensorboardX`、`psutil`、
`matplotlib` 属于训练期工具。实际推理是否触碰这些包，以
`python/audio_ai_flashsr/README.md` 中的实测结论为准。

`diffusers` **不需要**安装：上游已把 DPM-Solver Multistep 调度器内联在
`TorchJaekwon/Model/Diffusion/External/diffusers/`。
