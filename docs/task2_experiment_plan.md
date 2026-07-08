# 任务二实验实施规划：非线性感知训练

> **重要修正：本文档原规划基于“全局/窄范围 alpha 训练”，现在只能作为历史实施计划。**
> 最新主协议为：每个目标矩阵算子输入端、每次 forward 调用独立采样
> `alpha_{l,c} ~ Uniform(-1,1)`，且非线性函数保持官方形式不变。任务二正式主实验应使用
> `RandomAlphaNonlinearityInjector(-1,1)`；本文中 `Uniform(-0.2,0.2)`、
> `Uniform(-0.1,0.1)` 和 alpha sweep 网格只保留为预热、消融或辅助诊断。
> 统一定义见 [per_occurrence_random_alpha_protocol.md](per_occurrence_random_alpha_protocol.md)。

## 1. 任务目标

任务二的目标是在训练阶段加入存算一体芯片输入非线性映射模型，研究非线性感知训练（Nonlinearity-Aware Training, NAT）对模型收敛性、泛化性和非线性鲁棒性的影响，并对比两种训练策略：

1. 微调策略（fine-tuning）：先获得 clean 模型，再在训练阶段加入非线性映射继续训练。
2. 从头训练策略（training from scratch）：模型从随机初始化开始就在训练阶段加入非线性映射。

数据集继续使用任务一中的两个任务：

```text
图像分类：CIFAR-10
语音识别：VoxForge
```

任务二需要回答的问题包括：

1. NAT 是否能恢复或提升模型在非线性推理条件下的精度？
2. NAT 会不会牺牲 clean 条件下的推理精度？
3. fine-tuning 和 training from scratch 哪种更容易收敛？
4. 两种策略在未见过的 alpha 上是否具备泛化性？
5. NAT 对图像分类和语音识别是否表现出一致规律？

## 2. 与任务一结果的衔接

任务一已经完成非线性敏感性分析，并得到以下关键发现：

1. CIFAR-10 上 ResNet20 从 clean accuracy 92.59% 下降到强非线性下约 10%，说明非线性误差会导致模型接近随机猜测。
2. MobileNetV2 对弱非线性高度敏感，`alpha=±0.1` 附近已经出现严重退化。
3. ResNet20 的敏感层主要集中在 stage 起始层和中后段卷积层，例如 `layer3.0.conv1`、`layer1.0.conv1`、`layer2.0.conv1`。
4. VoxForge + Whisper-tiny 中，`alpha=±0.1` 已使 WER 明显恶化，说明语音模型对非线性误差更加敏感。
5. 负 alpha 更容易造成幅值漂移，正 alpha 更容易造成特征方向偏移，两类误差都需要在训练中覆盖。

旧版规划据此设计了弱到中等非线性的预热区间：

```text
alpha_train ∈ [-0.2, 0.2] 或 [-0.3, 0.3]
alpha_test  ∈ {-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5}
```

该设计可以观察 NAT 的初步恢复能力，但不再是官方主协议。正式任务二必须使用每算子每次独立随机 `alpha~Uniform(-1,1)`，并通过随机 alpha 场的多次重复评估收敛性和泛化性。

## 3. 非线性映射与训练注入方式

训练阶段沿用任务一的非线性映射：

```text
u = x / max(|x|)
f(u) = alpha * u^3 + (1 - alpha) * u
x' = max(|x|) * f(u)
```

实现中使用：

```python
max_val = x.detach().abs().amax().clamp_min(1e-12)
u = x / max_val
y = alpha * (u ** 3) + (1 - alpha) * u
x_nl = y * max_val
```

训练注入位置与任务一保持一致，统一注入到目标算子输入端：

```text
Conv1d
Conv2d
ConvTranspose1d
ConvTranspose2d
Linear
```

对于 CIFAR-10，主要注入：

```text
Conv2d 输入
Linear 输入
```

对于 VoxForge，主要注入：

```text
Conv1d 输入
Linear 输入
Transformer / CTC 模型中的 projection 输入
```

训练阶段仍采用 forward pre-hook 或可训练 wrapper 注入非线性，保证 clean 模型、NAT fine-tuning 和 NAT scratch 共享同一套网络定义，避免实现差异影响对比。

## 4. 实验总体设计

任务二采用“主实验 + 泛化实验 + 消融实验”的结构。

### 4.1 主实验

主实验比较三类模型：

| 编号 | 方法 | 初始化 | 训练阶段是否注入非线性 | 目的 |
|---|---|---|---|---|
| A | Clean Training | 随机初始化 | 否 | 得到 clean baseline |
| B | NAT Fine-tuning | 从 clean checkpoint 初始化 | 是 | 评估低成本硬件适配能力 |
| C | NAT From Scratch | 随机初始化 | 是 | 评估从头适应非线性硬件的能力 |

主实验需要在相同测试 alpha 网格上评估三类模型：

```text
alpha_test = [-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5]
```

其中 `alpha=0` 用于观察 NAT 是否损失 clean 精度，非零 alpha 用于观察鲁棒性。

### 4.2 NAT 训练模式

任务二至少设计两种 NAT 模式：

| 模式 | alpha 设置 | 说明 |
|---|---|---|
| Fixed-alpha NAT | 固定 `alpha_train = 0.2` 或 `alpha_train = -0.2` | 检查模型是否能适应单一非线性方向 |
| Random-alpha NAT | 每个 batch 从 `Uniform(-A, A)` 采样 alpha | 检查模型对非线性强度和方向的泛化能力 |

建议主线使用 Random-alpha NAT：

```text
alpha ~ Uniform(-0.2, 0.2)
```

可选扩展使用更强范围：

```text
alpha ~ Uniform(-0.3, 0.3)
```

原因是任务一显示模型对 `|alpha| >= 0.4` 容易接近完全失效，若训练初期直接使用过强扰动，可能导致训练不稳定。

### 4.3 课程式 alpha 训练

为了改善从头训练的收敛性，可加入 curriculum NAT 作为扩展：

```text
epoch 0%  - 30%: alpha ~ Uniform(-0.05, 0.05)
epoch 30% - 60%: alpha ~ Uniform(-0.10, 0.10)
epoch 60% - 100%: alpha ~ Uniform(-0.20, 0.20)
```

或者使用连续调度：

```text
A_t = A_max * min(1, epoch / warmup_epochs)
alpha ~ Uniform(-A_t, A_t)
```

该策略用于回答：

```text
强扰动是否会降低 early training 的稳定性？
课程式增强是否能让 NAT scratch 更容易收敛？
```

## 5. CIFAR-10 实验规划

### 5.1 模型选择

CIFAR-10 主模型选择 ResNet20，原因如下：

1. 任务一中 ResNet20 已完成完整敏感性分析。
2. ResNet20 训练成本低，适合重复实验。
3. ResNet20 有残差结构，适合观察 NAT 对误差传播的缓解作用。
4. 与任务一结果直接衔接，便于报告形成闭环。

辅助模型选择 MobileNetV2-x1.0：

1. 任务一显示 MobileNetV2 对非线性极其敏感。
2. 可用于验证 NAT 是否能提升轻量化网络鲁棒性。
3. 可作为结构泛化实验，而不是第一阶段主线。

### 5.2 CIFAR-10 实验分组

**表1 CIFAR-10 主实验分组**

| 实验编号 | 模型 | 方法 | 初始化 | 训练 alpha | 训练 epoch | 主要用途 |
|---|---|---|---|---|---:|---|
| C-A1 | ResNet20 | Clean Training | 随机初始化 | 无 | 160 或 200 | clean baseline |
| C-B1 | ResNet20 | NAT Fine-tuning | C-A1 checkpoint | `Uniform(-0.2, 0.2)` | 30 或 50 | 低成本硬件适配 |
| C-C1 | ResNet20 | NAT From Scratch | 随机初始化 | `Uniform(-0.2, 0.2)` | 160 或 200 | 从头鲁棒训练 |
| C-D1 | ResNet20 | Curriculum NAT Scratch | 随机初始化 | 逐步扩大到 `[-0.2, 0.2]` | 160 或 200 | 验证收敛稳定性 |
| C-E1 | ResNet20 | Fixed-alpha FT | C-A1 checkpoint | `alpha=0.2` | 30 或 50 | 单点硬件适配 |
| C-F1 | ResNet20 | Fixed-alpha FT | C-A1 checkpoint | `alpha=-0.2` | 30 或 50 | 正负方向差异 |

**表2 CIFAR-10 辅助结构实验**

| 实验编号 | 模型 | 方法 | 初始化 | 训练 alpha | 训练 epoch | 主要用途 |
|---|---|---|---|---|---:|---|
| C-A2 | MobileNetV2-x1.0 | Clean Training 或预训练权重 | 随机/预训练 | 无 | 160 或直接加载 | 结构对比 baseline |
| C-B2 | MobileNetV2-x1.0 | NAT Fine-tuning | clean checkpoint | `Uniform(-0.2, 0.2)` | 30 或 50 | 验证敏感模型能否恢复 |

第一阶段优先完成表1。表2作为拓展，不应阻塞任务二主线。

### 5.3 CIFAR-10 训练超参数

建议使用标准 CIFAR-10 训练设置：

| 项目 | 设置 |
|---|---|
| 输入增强 | RandomCrop(32, padding=4), RandomHorizontalFlip |
| 标准化 | CIFAR-10 mean/std |
| 优化器 | SGD |
| 初始学习率 | 0.1 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Batch size | 128 或 256 |
| Epoch | 160 或 200 |
| LR schedule | CosineAnnealingLR 或 MultiStepLR(80, 120) |
| Loss | CrossEntropyLoss |
| AMP | 可开启 |

Fine-tuning 超参数建议更保守：

| 项目 | 设置 |
|---|---|
| 优化器 | SGD 或 AdamW |
| 学习率 | 0.01 或 0.005 |
| Epoch | 30 或 50 |
| Weight decay | 5e-4 |
| LR schedule | CosineAnnealingLR |

Fine-tuning 学习率不宜过大，否则可能破坏 clean checkpoint 的已有判别能力。

### 5.4 CIFAR-10 评价指标

每个 checkpoint 都在统一 alpha 网格上评估：

```text
alpha_test = [-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5]
```

输出指标：

| 指标 | 说明 |
|---|---|
| Clean Accuracy | `alpha=0` 下 accuracy |
| Target Accuracy | 训练目标范围内，例如 `alpha=±0.2` 下 accuracy |
| Average Robust Accuracy | 所有 `alpha_test` 上 accuracy 平均值 |
| Worst-case Accuracy | 所有 `alpha_test` 上最低 accuracy |
| Accuracy Drop | `Acc(0)-Acc(alpha)` |
| Robust AUC | alpha-accuracy 曲线下离散面积 |
| Train Loss Curve | 训练集 loss 随 epoch 变化 |
| Val Accuracy Curve | validation accuracy 随 epoch 变化 |
| Epoch-to-threshold | 达到指定 accuracy 所需 epoch |

收敛性分析重点：

```text
Clean training、NAT fine-tuning、NAT scratch 的 loss 曲线是否稳定？
NAT scratch 是否需要更多 epoch 才能达到较高 clean accuracy？
Fine-tuning 是否能在少量 epoch 内快速恢复 nonlinear accuracy？
```

泛化性分析重点：

```text
固定 alpha 训练是否只在目标 alpha 上有效？
Random-alpha NAT 是否能提升未见 alpha 的平均鲁棒精度？
NAT 是否牺牲 alpha=0 的 clean accuracy？
```

## 6. VoxForge 实验规划

### 6.1 VoxForge 任务选择

VoxForge 是语音识别数据集，天然任务是 ASR。任务二需要训练阶段加入非线性，并对比 fine-tuning 与 from scratch。直接从头训练 Whisper 或大型 Wav2Vec2 不现实，且无法公平对比。因此 VoxForge 规划采用两条路线：

1. 控制实验主线：训练轻量级 ASR CTC 模型，完成 clean、NAT fine-tuning、NAT scratch 的公平比较。
2. 预训练模型补充：对 Whisper-tiny 或 Wav2Vec2 做轻量 fine-tuning，作为工程部署参考，但不作为 from-scratch 对照主线。

任务二主线应优先采用路线 1。

### 6.2 VoxForge 主模型：轻量 CTC ASR

建议构建一个轻量 CTC 模型：

```text
Input waveform
  -> log-mel spectrogram
  -> CNN subsampling frontend
  -> BiGRU / Transformer encoder
  -> Linear CTC head
  -> CTC loss
```

两个可选结构：

**方案 A：CRNN-CTC**

| 模块 | 设置 |
|---|---|
| 特征 | 80-dim log-mel |
| 前端 | 2-3 层 Conv2d + ReLU + BatchNorm |
| 序列模型 | 2 层 BiGRU 或 BiLSTM |
| 输出头 | Linear 到字符 vocabulary |
| Loss | CTC Loss |

优点：

```text
训练成本低
从头训练可行
代码复杂度较低
适合快速完成 fine-tuning vs scratch 对比
```

**方案 B：Tiny Transformer-CTC**

| 模块 | 设置 |
|---|---|
| 特征 | 80-dim log-mel |
| 前端 | Conv2d subsampling |
| Encoder | 4 层 TransformerEncoder |
| hidden dim | 256 |
| attention heads | 4 |
| 输出头 | Linear 到字符 vocabulary |
| Loss | CTC Loss |

优点：

```text
更贴近 Transformer 语音模型
与任务一 Whisper 结果呼应
可分析 attention/FFN 层非线性敏感性
```

建议第一阶段选择 CRNN-CTC 保证跑通；若时间允许，再补 Tiny Transformer-CTC。

### 6.3 VoxForge 数据处理

VoxForge 数据使用 Hugging Face 镜像：

```text
ciempiess/voxforge_spanish
```

字段：

```text
audio
normalized_text
```

建议数据划分：

| split | 样本数建议 | 用途 |
|---|---:|---|
| train | 5,000 - 20,000 | 训练 |
| valid | 500 - 1,000 | 早停和模型选择 |
| test | 500 - 1,000 | 最终 WER/CER |

如果下载和训练资源受限，可先采用小规模版本：

| split | 样本数 |
|---|---:|
| train | 2,000 |
| valid | 200 |
| test | 200 |

文本建模建议：

```text
字符级 vocabulary
blank token for CTC
保留西班牙语常见字符
统一 lowercase
去除极少见标点
```

训练特征：

```text
采样率：16 kHz
特征：80-bin log-mel spectrogram
帧长：25 ms
帧移：10 ms
可选 SpecAugment
```

### 6.4 VoxForge 实验分组

**表3 VoxForge 主实验分组**

| 实验编号 | 模型 | 方法 | 初始化 | 训练 alpha | 训练 epoch | 主要用途 |
|---|---|---|---|---|---:|---|
| V-A1 | CRNN-CTC | Clean Training | 随机初始化 | 无 | 50 - 100 | clean ASR baseline |
| V-B1 | CRNN-CTC | NAT Fine-tuning | V-A1 checkpoint | `Uniform(-0.1, 0.1)` | 10 - 30 | 低成本硬件适配 |
| V-C1 | CRNN-CTC | NAT From Scratch | 随机初始化 | `Uniform(-0.1, 0.1)` | 50 - 100 | 从头鲁棒训练 |
| V-D1 | CRNN-CTC | Curriculum NAT Scratch | 随机初始化 | 逐步扩大到 `[-0.1, 0.1]` | 50 - 100 | 改善 ASR 收敛 |
| V-E1 | CRNN-CTC | Fixed-alpha FT | V-A1 checkpoint | `alpha=0.1` | 10 - 30 | 单方向适配 |
| V-F1 | CRNN-CTC | Fixed-alpha FT | V-A1 checkpoint | `alpha=-0.1` | 10 - 30 | 正负方向差异 |

VoxForge 的训练 alpha 建议比 CIFAR-10 更保守：

```text
alpha_train ~ Uniform(-0.1, 0.1)
```

原因是任务一中 Whisper-tiny 在 `alpha=±0.1` 已经严重退化。ASR 模型对非线性更敏感，训练初期不宜直接使用 `[-0.2, 0.2]`。

### 6.5 VoxForge 评价指标

统一测试网格：

```text
alpha_test = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
```

输出指标：

| 指标 | 说明 |
|---|---|
| Clean WER/CER | `alpha=0` 下 WER/CER |
| Robust WER/CER | 所有非零 alpha 下平均 WER/CER |
| Worst WER/CER | 测试 alpha 网格上最大 WER/CER |
| Relative WER Increase | `(WER(alpha)-WER(0))/WER(0)` |
| CTC Training Loss | 训练收敛曲线 |
| Valid WER/CER Curve | 验证集识别性能曲线 |
| Decode Examples | 展示 reference / hypothesis 对比 |

ASR 报告中应展示至少 5 条典型样例：

```text
reference
clean hypothesis
nonlinear hypothesis before NAT
nonlinear hypothesis after NAT
```

这样可以直观展示 NAT 对识别错误的恢复效果。

## 7. 训练策略对比逻辑

任务二的核心不是只证明 NAT 有效，而是比较 fine-tuning 和 from scratch 的差异。因此每个任务都应从以下维度对比。

### 7.1 收敛性

对比对象：

```text
Clean Training
NAT Fine-tuning
NAT From Scratch
Curriculum NAT From Scratch
```

评价：

| 对比项 | 观察内容 |
|---|---|
| 训练 loss 曲线 | NAT 是否导致训练不稳定 |
| 验证指标曲线 | NAT 是否收敛更慢 |
| 达到阈值所需 epoch | fine-tuning 是否更快 |
| 最终 train/valid gap | NAT 是否改变过拟合程度 |

预期：

```text
Fine-tuning 收敛最快，因为继承 clean 模型特征。
NAT scratch 收敛较慢，但可能在强非线性测试上更鲁棒。
Curriculum NAT scratch 比直接 NAT scratch 更稳定。
```

### 7.2 泛化性

泛化性包括两类：

1. alpha 泛化：训练时只见过部分 alpha，测试未见 alpha。
2. clean-nonlinear 兼容性：模型能否同时保持 clean accuracy 和 nonlinear accuracy。

评价表应包含：

| 方法 | Clean Metric | Avg Robust Metric | Worst Metric | 泛化结论 |
|---|---:|---:|---:|---|
| Clean baseline | 高 clean，低 robust | 低 | 差 | 无硬件适配 |
| Fixed-alpha FT | 目标 alpha 高 | 未见 alpha 一般 | 中 | 单点适配 |
| Random-alpha FT | clean 略降 | robust 高 | 高 | 适配效率好 |
| Random-alpha Scratch | clean 可能略低 | robust 高 | 高 | 更彻底适应 |
| Curriculum Scratch | clean/robust 平衡 | 高 | 高 | 收敛更稳 |

### 7.3 fine-tuning 与 scratch 的判定标准

需要避免只用最终 accuracy 判断优劣。建议使用综合指标：

```text
Score = AvgRobustAcc - lambda * CleanAccDrop
```

对于 CIFAR：

```text
CleanAccDrop = Acc_clean_baseline(alpha=0) - Acc_method(alpha=0)
AvgRobustAcc = mean_{alpha_test != 0} Acc_method(alpha)
```

对于 VoxForge：

```text
CleanWERIncrease = WER_method(alpha=0) - WER_clean_baseline(alpha=0)
AvgRobustWER = mean_{alpha_test != 0} WER_method(alpha)
```

最终报告中可以不一定合成为一个单分数，但应同时展示 clean 性能和 robust 性能，避免牺牲 clean 性能换取单点鲁棒性的片面结论。

## 8. 预期图表与报告产物

任务二应至少生成以下图表。

### 8.1 CIFAR-10 图表

| 图表 | 内容 |
|---|---|
| 训练 loss 曲线 | Clean / NAT FT / NAT Scratch 对比 |
| 验证 accuracy 曲线 | 分析收敛速度 |
| alpha-accuracy 曲线 | 不同方法在测试 alpha 上的鲁棒性 |
| accuracy drop 曲线 | 相对 clean baseline 的下降幅度 |
| clean vs robust 柱状图 | clean accuracy、avg robust accuracy、worst accuracy |
| fine-tuning vs scratch 对比表 | 收敛 epoch、最终 clean、最终 robust |
| 固定 alpha vs random alpha 对比图 | 验证泛化性 |

### 8.2 VoxForge 图表

| 图表 | 内容 |
|---|---|
| CTC train loss 曲线 | Clean / NAT FT / NAT Scratch 对比 |
| valid WER/CER 曲线 | 分析 ASR 收敛速度 |
| alpha-WER 曲线 | 不同方法在非线性下 WER |
| relative WER increase 曲线 | 非线性导致的相对恶化 |
| clean / avg robust / worst WER 柱状图 | 汇总鲁棒性 |
| 识别样例表 | reference 与不同方法 hypothesis |

### 8.3 推荐输出目录

```text
outputs/task2/
  checkpoints/
    cifar/
    voxforge/
  tables/
    cifar_training_logs.csv
    cifar_alpha_eval.csv
    cifar_method_summary.csv
    voxforge_training_logs.csv
    voxforge_alpha_eval.csv
    voxforge_method_summary.csv
    voxforge_decode_examples.csv
  figures/
    cifar_train_loss.png
    cifar_val_accuracy.png
    cifar_alpha_accuracy.png
    cifar_method_summary.png
    voxforge_train_loss.png
    voxforge_val_wer.png
    voxforge_alpha_wer.png
    voxforge_method_summary.png
```

## 9. 代码实施规划

任务二建议新增如下文件。

**表4 任务二拟新增工程文件**

| 文件 | 作用 |
|---|---|
| `src/training.py` | 通用训练循环、checkpoint、日志保存 |
| `src/alpha_scheduler.py` | fixed/random/curriculum alpha 采样策略 |
| `src/cifar_models.py` | CIFAR ResNet20/MobileNetV2 构建与加载 |
| `src/voxforge_data.py` | VoxForge 数据下载、切分、文本规范化、collate |
| `src/asr_models.py` | CRNN-CTC / Tiny Transformer-CTC 模型 |
| `scripts/train_cifar_task2.py` | CIFAR-10 clean/NAT 训练入口 |
| `scripts/eval_cifar_task2.py` | CIFAR-10 alpha 扫描评估入口 |
| `scripts/train_voxforge_task2.py` | VoxForge ASR clean/NAT 训练入口 |
| `scripts/eval_voxforge_task2.py` | VoxForge alpha 扫描与解码评估入口 |
| `scripts/make_task2_figures.py` | 根据 CSV 生成任务二图表 |
| `docs/task2_technical_report.md` | 任务二实验完成后的技术报告 |

任务一已有 `src/nonlinear.py` 可以复用，但需要扩展两个能力：

1. 支持训练时动态 alpha，即每个 batch 更新 alpha。
2. 支持只对指定层集合注入非线性，用于后续敏感层训练消融。

### 9.1 Alpha 调度接口

建议设计统一接口：

```python
class AlphaScheduler:
    def sample(self, epoch: int, step: int) -> float:
        ...
```

实现：

```text
FixedAlphaScheduler(alpha=0.2)
UniformAlphaScheduler(low=-0.2, high=0.2)
CurriculumAlphaScheduler(max_abs=0.2, warmup_epochs=80)
```

### 9.2 训练日志字段

每个 epoch 保存：

```text
dataset
model
method
epoch
train_loss
train_accuracy 或 train_cer
val_loss
val_accuracy 或 val_wer/cer
alpha_mode
alpha_low
alpha_high
learning_rate
checkpoint_path
```

每次 alpha 评估保存：

```text
dataset
model
method
checkpoint
alpha
accuracy/loss 或 wer/cer
clean_metric
metric_drop 或 relative_metric_increase
```

## 10. 推荐执行顺序

为了降低工程风险，任务二建议按以下顺序执行。

### 阶段一：CIFAR-10 跑通完整闭环

1. 实现 alpha scheduler。
2. 实现 CIFAR clean training。
3. 训练 ResNet20 clean baseline。
4. 对 clean baseline 做 alpha sweep，复现任务一趋势。
5. 基于 clean checkpoint 做 NAT fine-tuning。
6. 训练 NAT from scratch。
7. 生成 CIFAR 训练曲线和 alpha-accuracy 对比图。

阶段一完成后，任务二已经具备核心结果。

### 阶段二：CIFAR-10 扩展实验

1. 加入 fixed-alpha fine-tuning。
2. 加入 curriculum NAT scratch。
3. 选择 MobileNetV2 做 NAT fine-tuning。
4. 比较 ResNet20 与 MobileNetV2 的 NAT 收益差异。

### 阶段三：VoxForge 主线

1. 实现 VoxForge 数据处理和字符 vocabulary。
2. 实现 CRNN-CTC。
3. 训练 clean ASR baseline。
4. 基于 clean ASR checkpoint 做 NAT fine-tuning。
5. 训练 NAT from scratch。
6. 在 alpha_test 网格上评估 WER/CER。
7. 生成 WER 曲线和识别样例表。

### 阶段四：报告与消融

1. 汇总 CIFAR 与 VoxForge 的收敛曲线。
2. 汇总 fine-tuning vs scratch 的鲁棒性指标。
3. 对比 fixed-alpha NAT 与 random-alpha NAT。
4. 分析 clean 性能与 robust 性能 trade-off。
5. 撰写任务二技术报告。

## 11. 预期结论假设

任务二实验开始前可提出以下假设，后续用实验验证：

1. Clean baseline 在 `alpha=0` 下精度最高，但在非零 alpha 下鲁棒性最差。
2. NAT fine-tuning 能以较少 epoch 快速提升非线性推理精度，是部署前硬件适配的高性价比方案。
3. NAT from scratch 收敛更慢，但在更宽 alpha 范围上可能获得更高 worst-case robustness。
4. Random-alpha NAT 比 fixed-alpha NAT 更适合未知硬件非线性，因为它训练的是一个误差族而不是单点误差。
5. Curriculum NAT 能改善 NAT from scratch 的早期收敛稳定性。
6. VoxForge ASR 对非线性扰动比 CIFAR-10 分类更敏感，因此需要更小的训练 alpha 范围和更保守的学习率。

## 12. 风险与应对策略

**风险一：从头训练 ASR 模型耗时较长。**

应对：

```text
先使用 2,000/200/200 的小规模 VoxForge split 跑通闭环；
CRNN-CTC 优先于 Transformer-CTC；
保存中间 checkpoint，支持断点续训。
```

**风险二：NAT scratch 训练不稳定。**

应对：

```text
降低 alpha_train 范围；
使用 curriculum alpha；
降低学习率；
先冻结部分前端层进行 warmup；
检查 max_val detach 和 eps 防止数值异常。
```

**风险三：NAT 提升 robust accuracy 但明显降低 clean accuracy。**

应对：

```text
报告 clean-robust trade-off；
尝试混合训练 batch，即部分 batch alpha=0，部分 batch alpha!=0；
调整 alpha 采样范围；
加入 clean consistency loss 作为后续任务三方法基础。
```

**风险四：VoxForge 数据集 streaming 导致复现实验不稳定。**

应对：

```text
首次运行后保存固定样本索引或本地 manifest；
固定 random seed；
将 train/valid/test 划分写入 CSV；
后续实验严格复用相同 manifest。
```

**风险五：不同方法训练预算不公平。**

应对：

```text
同时报告 epoch 数、训练 step 数和训练耗时；
fine-tuning 与 scratch 分别以“低成本适配”和“全流程训练”定位；
主表中明确训练预算。
```

## 13. 最小可交付版本

如果时间有限，任务二最小可交付版本应至少包含：

1. CIFAR-10 ResNet20 clean baseline。
2. CIFAR-10 ResNet20 NAT fine-tuning。
3. CIFAR-10 ResNet20 NAT from scratch。
4. 三种方法在统一 alpha_test 网格上的 accuracy 曲线。
5. 三种方法的训练 loss 和 validation accuracy 曲线。
6. VoxForge CRNN-CTC clean baseline。
7. VoxForge CRNN-CTC NAT fine-tuning。
8. VoxForge CRNN-CTC NAT from scratch。
9. 三种 ASR 方法的 WER/CER-alpha 曲线。
10. 一张 fine-tuning vs scratch 汇总表。

最小交付版本已经能够覆盖赛题任务二的核心要求：

```text
训练阶段加入非线性映射模型
分析收敛性
分析泛化性
比较 fine-tuning 与 from scratch
覆盖 CIFAR-10 和 VoxForge
```

## 14. 推荐最终实验矩阵

**表5 任务二最终推荐实验矩阵**

| 数据集 | 模型 | Clean | NAT Fine-tuning | NAT Scratch | Curriculum NAT | Fixed-alpha 消融 | 主要指标 |
|---|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 | ResNet20 | 必做 | 必做 | 必做 | 推荐 | 推荐 | Acc / Loss / Robust AUC |
| CIFAR-10 | MobileNetV2 | 可选 | 推荐 | 可选 | 可选 | 可选 | Acc / Robust AUC |
| VoxForge | CRNN-CTC | 必做 | 必做 | 必做 | 推荐 | 推荐 | WER / CER / CTC Loss |
| VoxForge | Tiny Transformer-CTC | 可选 | 可选 | 可选 | 可选 | 可选 | WER / CER |
| VoxForge | Whisper-tiny | 已有推理基线 | 可选 FT | 不建议 | 不建议 | 可选 | WER / CER |

最终报告中建议将 ResNet20 与 CRNN-CTC 作为任务二主线，将 MobileNetV2 与 Whisper-tiny 作为结构泛化或部署参考补充。

## 15. 当前实施状态与下一步执行清单

截至当前版本，任务二已经完成 CIFAR-10 主线的代码框架和首轮实验闭环，VoxForge 主线仍处于待实现阶段。因此后续工作应按“先固化 CIFAR 图表与结论，再实现 VoxForge ASR 闭环，最后统一写报告”的顺序推进。

### 15.1 已完成工程状态

**表6 当前已完成文件与功能**

| 文件 | 当前状态 | 说明 |
|---|---|---|
| `src/alpha_scheduler.py` | 已完成 | 支持 none、fixed、uniform、curriculum 四类 alpha 调度 |
| `src/training.py` | 已完成 | 支持随机种子、目录创建、CSV 追加、JSON 保存、checkpoint 保存与加载 |
| `src/cifar_models.py` | 已完成 | 支持 CIFAR-10 dataloader、ResNet20/MobileNetV2 等模型构建 |
| `scripts/train_cifar_task2.py` | 已完成 | 支持 clean training、NAT fine-tuning、NAT scratch、训练日志、checkpoint、alpha sweep |
| `outputs/task2/checkpoints/cifar/` | 已生成 | 已保存 CIFAR-10 三类方法的 best/last checkpoint |
| `outputs/task2/tables/cifar_training_logs.csv` | 已生成 | 已保存 CIFAR-10 逐 epoch 收敛日志 |
| `outputs/task2/tables/cifar_alpha_eval.csv` | 已生成 | 已保存 CIFAR-10 不同 alpha 下的测试精度 |
| `outputs/task2/tables/cifar_method_summary.csv` | 已生成 | 已保存 CIFAR-10 clean/robust/worst 汇总指标 |
| `src/voxforge_data.py` | 待实现 | VoxForge 下载、切分、文本规范化、collate |
| `src/asr_models.py` | 待实现 | CRNN-CTC / Tiny Transformer-CTC |
| `scripts/train_voxforge_task2.py` | 待实现 | VoxForge clean/NAT 训练入口 |
| `scripts/make_task2_figures.py` | 待实现 | CIFAR-10 与 VoxForge 图表统一生成 |
| `docs/task2_technical_report.md` | 待撰写 | 任务二技术报告 |

### 15.2 CIFAR-10 首轮实验记录

当前 CIFAR-10 首轮实验使用 ResNet20，目标是先验证 NAT 训练、checkpoint 保存、alpha 扫描和表格记录全流程可用。该轮实验预算低于正式推荐预算，适合作为第一轮结果和方法调试依据；正式提交前若时间允许，应补跑 160 或 200 epoch 版本。

**表7 CIFAR-10 首轮实验配置**

| 方法 | run name | 初始化 | 训练 alpha | epoch | batch size | 学习率 | 说明 |
|---|---|---|---|---:|---:|---:|---|
| Clean Training | `cifar_resnet20_clean` | 随机初始化 | 无 | 60 | 512 | 0.1 | clean baseline |
| NAT Fine-tuning | `cifar_resnet20_nat_ft` | clean best checkpoint | `Uniform(-0.2, 0.2)` | 20 | 512 | 0.01 | 低成本硬件适配 |
| NAT From Scratch | `cifar_resnet20_nat_scratch` | 随机初始化 | `Uniform(-0.2, 0.2)` | 60 | 512 | 0.1 | 从头鲁棒训练 |

**表8 CIFAR-10 首轮 alpha sweep 汇总**

| 方法 | Clean Accuracy | Avg Robust Accuracy | Worst Accuracy | 观察 |
|---|---:|---:|---:|---|
| Clean Training | 0.8939 | 0.3141 | 0.1000 | clean 精度正常，但强非线性下接近随机猜测 |
| NAT Fine-tuning | 0.8941 | 0.3122 | 0.1079 | clean 精度保持，但当前 uniform 训练未显著提升平均鲁棒精度 |
| NAT From Scratch | 0.8965 | 0.2812 | 0.0935 | clean 精度略高，但对正负 alpha 的泛化不均衡，需进一步调参 |

这组结果说明：仅使用 clean validation accuracy 选择 checkpoint，可能会偏向 clean 表现而不是 robust 表现；后续 CIFAR 扩展实验应加入 robust validation checkpoint selection，或者在验证阶段同时计算 `alpha ∈ {-0.2, 0, 0.2}` 的平均指标作为模型选择依据。

### 15.3 CIFAR-10 后续补充实验

为了让任务二结论更充分，CIFAR-10 建议继续补充以下实验。

**表9 CIFAR-10 补充实验优先级**

| 优先级 | 实验 | 目的 | 是否阻塞任务二主线 |
|---:|---|---|---|
| P0 | 根据现有 CSV 生成训练曲线和 alpha-accuracy 曲线 | 形成可汇报图表 | 是 |
| P0 | 增加 robust validation 选择 best checkpoint | 判断 NAT 未提升是否由 checkpoint 选择导致 | 是 |
| P1 | Fixed-alpha FT，`alpha=0.2` 和 `alpha=-0.2` | 分析单方向硬件适配效果 | 否 |
| P1 | Curriculum NAT scratch | 改善从头训练早期稳定性 | 否 |
| P2 | MobileNetV2 NAT fine-tuning | 结构泛化分析 | 否 |
| P2 | 160/200 epoch 正式复现实验 | 提高最终报告说服力 | 否，但推荐 |

### 15.4 VoxForge 落地优先路线

VoxForge 的任务二实现应先追求公平可训练，而不是直接使用大型预训练 ASR 模型。因此第一版采用 CRNN-CTC，待数据、训练和 WER/CER 评估跑通后，再考虑 Tiny Transformer-CTC 或 Whisper-tiny fine-tuning。

**表10 VoxForge 第一版实施细节**

| 模块 | 采用方案 |
|---|---|
| 数据源 | `ciempiess/voxforge_spanish` |
| 训练子集 | 首轮 `train=2000, valid=200, test=200`；正式版可扩大到 `train=5000-20000` |
| 文本单元 | 字符级 vocabulary + CTC blank |
| 音频特征 | 16 kHz waveform -> 80-bin log-mel spectrogram |
| 模型 | Conv2d subsampling + 2 层 BiGRU + Linear CTC head |
| clean 训练 | 随机初始化，不注入非线性 |
| NAT fine-tuning | 从 clean best checkpoint 初始化，`alpha ~ Uniform(-0.1, 0.1)` |
| NAT scratch | 随机初始化，`alpha ~ Uniform(-0.1, 0.1)` |
| 测试 alpha | `[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]` |
| 指标 | WER、CER、CTC loss、decode examples |

### 15.5 首轮与正式实验预算区分

后续报告中需要明确区分“首轮实现验证实验”和“正式复现实验”，避免因 epoch 较少导致结论被误读。

**表11 首轮实验与正式实验预算**

| 数据集 | 实验阶段 | Clean | NAT Fine-tuning | NAT Scratch | 用途 |
|---|---|---:|---:|---:|---|
| CIFAR-10 | 首轮实现验证 | 60 epoch | 20 epoch | 60 epoch | 验证代码闭环、生成初步曲线 |
| CIFAR-10 | 正式复现实验 | 160/200 epoch | 30/50 epoch | 160/200 epoch | 最终报告主结果 |
| VoxForge | 首轮实现验证 | 10/20 epoch | 5/10 epoch | 10/20 epoch | 验证 ASR 数据流、CTC loss、WER 评估 |
| VoxForge | 正式复现实验 | 50/100 epoch | 10/30 epoch | 50/100 epoch | 最终报告主结果 |

### 15.6 下一步执行顺序

当前最合理的推进顺序如下：

1. 完成 `scripts/make_task2_figures.py`，先把 CIFAR-10 已有 CSV 转成训练 loss、validation accuracy、alpha-accuracy 和 clean/robust/worst 柱状图。
2. 在 CIFAR-10 训练脚本中加入 robust validation 选 checkpoint 的选项，复查 NAT fine-tuning 的鲁棒性收益。
3. 实现 `src/voxforge_data.py` 和 `src/asr_models.py`，先用小规模 VoxForge subset 跑通 clean CRNN-CTC。
4. 实现 VoxForge NAT fine-tuning 和 NAT scratch，统一输出 training logs、alpha eval、method summary 和 decode examples。
5. 生成 VoxForge 图表，并与 CIFAR-10 图表使用一致的视觉风格。
6. 撰写 `docs/task2_technical_report.md`，按“方法、收敛性、泛化性、fine-tuning vs scratch、跨任务对比、局限性”组织。

### 15.7 任务二报告建议结论口径

当前 CIFAR-10 首轮结果已经提示一个重要现象：NAT 并不一定天然提升所有 alpha 下的平均鲁棒性，训练 alpha 范围、checkpoint 选择准则、非线性注入粒度都会影响最终效果。因此任务二报告不应简单写成“NAT 一定有效”，而应更严谨地表述为：

```text
非线性感知训练提供了将硬件非理想特性纳入训练闭环的机制。
其收益取决于训练 alpha 分布、模型选择准则和任务敏感性。
fine-tuning 适合低成本硬件适配，scratch 更适合构建端到端鲁棒模型，但需要更长训练预算和更稳定的扰动调度。
```

后续如果 robust validation、fixed-alpha 或 curriculum NAT 获得更明显提升，可以在最终报告中进一步归纳为任务三鲁棒性增强方法的动机。
