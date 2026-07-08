# InnoCIM-DistortAlpha

面向存算一体（Compute-in-Memory, CIM）神经网络部署的非线性输入失真分析、非线性感知训练（NAT）与固定点位串行鲁棒编码（FPC）实验项目。

本项目围绕赛题给定的三次非线性输入响应

```text
f_alpha(u) = alpha * u^3 + (1 - alpha) * u
```

研究当 `Conv1d`、`Conv2d`、`ConvTranspose1d`、`ConvTranspose2d`、`Linear` 等矩阵算子输入端出现未知非线性失真时，CIFAR-10 图像分类模型与 VoxForge Spanish 语音识别模型的推理性能、层级敏感性、激活漂移、训练适配能力和鲁棒部署编码方案。

## 核心结论

- 输入相关非线性不是普通加性噪声。它会随激活幅值、层位置和算子调用逐层累积，导致分类精度或 ASR WER/CER 快速退化。
- 在严格 `alpha ~ Uniform(-1, 1)` 且每个算子、每次 forward 独立采样的协议下，单纯依靠 NAT 训练不能稳定消除未知随机非线性。
- FPC（Fixed-Point Bit-Serial Coding）利用 `{-1, 0, +1}` 是该三次函数对任意 `alpha` 的不动点这一结构，将连续激活拆为固定点 bit-plane，使硬件非线性主要退化为可控量化误差。
- CIFAR-10 实验中，裸随机非线性可使 clean checkpoint 从约 `89%` 精度跌至接近随机猜测，而 FPC `B=5`/`B=6` 可恢复到接近 clean 的水平。
- 标准 FP32 IEEE754 bit-string 分解是存储格式，不是鲁棒计算编码；在随机非线性硬件上明显弱于 FPC。

## 目录结构

```text
.
├── src/                         # 可复用模块：非线性注入、模型、指标、训练工具、绘图工具
├── scripts/                     # 任务脚本：敏感性分析、NAT 训练、FPC/FP32 评估、图表聚合
├── docs/                        # 技术报告、论文草稿与发布用图表
│   └── paper_figures/           # 论文级图片，已纳入版本库
├── data/                        # 本地数据缓存，不纳入版本库
├── outputs/                     # 本地实验输出、checkpoint、日志，不纳入版本库
├── requirements.txt             # Python 依赖
└── requirements-task1.txt       # 早期任务一依赖清单，保留用于兼容
```

## 主要模块

| 路径 | 作用 |
|---|---|
| `src/nonlinear.py` | 三次非线性注入、随机 alpha 注入、FPC 注入、FP32/black-box pilot 相关算子封装 |
| `src/cifar_models.py` | CIFAR-10 数据加载与模型构建 |
| `src/voxforge_data.py` | VoxForge Spanish 流式采样、缓存、字符词表和 CTC collate |
| `src/asr_models.py` | VoxForge CRNN-CTC 模型 |
| `src/metrics.py` | 分类准确率、WER/CER、激活差异等指标 |
| `src/sensitivity.py` | 敏感层选择与权重构造 |
| `src/training.py` | checkpoint、CSV/JSON 输出、随机种子等训练工具 |
| `scripts/run_*_task1*.py` | 任务一：非线性敏感性与随机 alpha 协议评估 |
| `scripts/train_*_task2.py` | 任务二：固定/分布 alpha 的 NAT 训练 |
| `scripts/train_*_per_occ_nat.py` | 每算子每次调用随机 alpha 的 NAT 训练 |
| `scripts/evaluate_*_bitserial_fixedpoint.py` | FPC 评估 |
| `scripts/evaluate_cifar_fp32_ieee_bitserial.py` | FP32 IEEE754 bit-string 对照实验 |
| `scripts/aggregate_*.py`、`scripts/make_*.py` | 结果聚合与论文图表生成 |

## 文档与图表

建议优先阅读：

1. [`docs/innoCIM_full_paper_zh.md`](docs/innoCIM_full_paper_zh.md)：项目完整中文论文稿。
2. [`docs/per_occurrence_random_alpha_protocol.md`](docs/per_occurrence_random_alpha_protocol.md)：当前主协议定义，即每个目标算子、每次 forward 独立采样 `alpha`。
3. [`docs/task1_task2_integrated_technical_report.md`](docs/task1_task2_integrated_technical_report.md)：任务一与任务二综合技术报告。
4. [`docs/task3_fpc_paper_experiments_report.md`](docs/task3_fpc_paper_experiments_report.md)：FPC 论文实验报告。
5. [`docs/task4_unknown_distortion_bpic_report.md`](docs/task4_unknown_distortion_bpic_report.md)：未知公式失真下的 BPIC 扩展实验。

论文主图保存在 [`docs/paper_figures/`](docs/paper_figures/)：

| 图 | 内容 |
|---|---|
| `fig01_distortion_and_sensitivity` | 三次非线性曲线与跨任务整网敏感性 |
| `fig02_layer_sensitivity_and_drift` | 单层敏感性与激活漂移 |
| `fig03_nat_baseline` | NAT baseline |
| `fig04_fpc_main_result` | FPC 主结果 |
| `fig05_nbit_ablation` | FPC bit 位宽消融 |
| `fig06_fp32_counterexample` | FP32 IEEE754 bit-string 反例 |
| `fig07_efficiency_tradeoff` | 精度与 bit-serial 激活流量权衡 |
| `fig08_voxforge_fpc` | VoxForge FPC 验证 |

## 环境准备

建议使用 Python 3.10 或更新版本。GPU 不是必需条件，但完整训练和大规模评估建议使用 CUDA。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch、TorchVision、TorchAudio 的 CUDA 版本需要按本机 CUDA 环境选择安装源。若默认 `pip install torch torchvision torchaudio` 安装的是 CPU 版本，请参考 PyTorch 官方安装命令替换。

## 数据准备

数据和模型 checkpoint 体积较大，默认不提交到 Git。

### CIFAR-10

脚本默认从 `data/` 读取 CIFAR-10，且当前代码使用 `download=False`。请将 CIFAR-10 Python 版数据解压为：

```text
data/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── ...
    └── test_batch
```

### VoxForge Spanish

VoxForge 相关脚本通过 Hugging Face `datasets` 流式读取 `ciempiess/voxforge_spanish`，并在 `data/voxforge_task2/` 下生成本地 `.pt` 与 `.json` 缓存。缓存文件不进入版本库。

## 常用命令

以下命令可使用较小 subset 快速验证流程。完整论文实验需要更大的训练预算和已有 checkpoint。

### CIFAR-10 随机 alpha 敏感性分析

```powershell
python scripts\run_cifar_task1_random.py `
  --output-dir outputs\task1_random_smoke `
  --test-subset 512 `
  --random-repeats 2 `
  --layer-repeats 1 `
  --drift-batches 1
```

### CIFAR-10 非线性感知训练

```powershell
python scripts\train_cifar_task2.py `
  --method clean `
  --run-name cifar_resnet20_clean_smoke `
  --epochs 1 `
  --train-subset 2048 `
  --val-subset 512 `
  --test-subset 512 `
  --batch-size 128
```

### 每算子随机 alpha NAT

```powershell
python scripts\train_cifar_per_occ_nat.py `
  --output-dir outputs\task2_random_cifar_nat_smoke `
  --run-name cifar_per_occ_nat_smoke `
  --epochs 1 `
  --train-subset 2048 `
  --val-subset 512 `
  --test-subset 512 `
  --eval-repeats 2
```

### FPC 评估

FPC 评估需要提供已训练 checkpoint：

```powershell
python scripts\evaluate_cifar_bitserial_fixedpoint.py `
  --checkpoint outputs\task2\checkpoints\cifar\cifar_resnet20_clean_best.pt `
  --output-dir outputs\task3_fpc_cifar `
  --test-subset 1024 `
  --random-repeats 3 `
  --bits "1,2,3,4,5,6,8"
```

### FP32 IEEE754 对照实验

```powershell
python scripts\evaluate_cifar_fp32_ieee_bitserial.py `
  --checkpoint outputs\task2\checkpoints\cifar\cifar_resnet20_clean_best.pt `
  --output-dir outputs\paper_fp32_ieee_cifar_clean `
  --test-subset 1024 `
  --random-repeats 3 `
  --fp32-repeats 1
```

### 一键批量实验脚本

```powershell
.\scripts\run_final_fpc_random_alpha_evals.ps1
.\scripts\run_fp32_ieee_cifar_evals.ps1
```

这些批量脚本依赖 `outputs/` 中已有训练 checkpoint，适合在完成任务二训练后运行。

## 结果输出约定

实验脚本通常在 `outputs/<run-name>/` 下生成：

- `tables/`：CSV/JSON 指标、metadata 和聚合表。
- `figures/`：实验图表。
- `checkpoints/`：模型权重，默认不进入 Git。
- `logs/`：长任务日志，默认不进入 Git。

已经整理出的论文级静态图片位于 `docs/paper_figures/`，会随仓库提交。

## 复现实验注意事项

- 多数 CIFAR 脚本默认要求 `data/cifar-10-batches-py/` 已存在。
- `torch.hub.load("chenyaofo/pytorch-cifar-models", ...)` 首次运行需要网络访问。
- VoxForge 脚本首次运行需要访问 Hugging Face 数据集并生成本地缓存。
- 完整训练与 FPC 全量评估耗时较长，README 中的命令偏向 smoke test。
- 随机 alpha 主协议请以 [`docs/per_occurrence_random_alpha_protocol.md`](docs/per_occurrence_random_alpha_protocol.md) 为准。

## 仓库大文件策略

本仓库提交源码、实验脚本、技术文档和论文级图片。以下内容保留在本机或外部存储中，不直接提交到 GitHub：

- 原始数据集与数据缓存：`data/`
- 模型权重与训练 checkpoint：`*.pt`、`*.pth`、`*.ckpt`
- 大规模实验输出与日志：`outputs/`
- Python 缓存与虚拟环境：`__pycache__/`、`.venv/`

如果后续需要发布可复现实验权重，建议使用 GitHub Releases、对象存储或 Git LFS，而不是直接提交到普通 Git 历史。

