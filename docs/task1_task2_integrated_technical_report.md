# 任务一与任务二综合技术报告：非线性误差敏感性分析与非线性感知训练

> **重要修正：本文旧版中的固定 alpha sweep 与全局 alpha 训练只能作为辅助诊断或历史基线。**
> 根据最新确认，任务一和任务二的官方主协议应为“每个矩阵算子输入端、每次发生失真时
> 都独立随机采样 `alpha_{l,c} ~ Uniform(-1,1)`”，同一次推理中不同层可以遇到不同
> alpha。主实验、主表和主图应使用 `RandomAlphaNonlinearityInjector(-1,1)` 得到的
> random-field 统计。统一定义见
> [per_occurrence_random_alpha_protocol.md](per_occurrence_random_alpha_protocol.md)。

## 摘要

本文整合任务一和任务二的技术文档，围绕存算一体芯片中模拟域乘累加运算的输入相关非线性误差，形成从“误差敏感性分析”到“非线性感知训练”的完整实验闭环。

任务一研究非线性误差在推理阶段对神经网络性能的影响。实验将赛题给定的三次非线性映射注入到 `Conv1d`、`Conv2d`、`ConvTranspose1d`、`ConvTranspose2d`、`Linear` 等矩阵计算算子的输入端，分别在 CIFAR-10 图像分类和 VoxForge Spanish 语音识别任务上分析整网精度衰减、单层敏感性、激活分布偏移和逐层误差累积行为。

任务二在任务一的基础上，将同一非线性映射加入训练闭环，研究非线性感知训练（Nonlinearity-Aware Training, NAT）对模型收敛性、泛化性和硬件非线性鲁棒性的影响。任务二比较 clean training、NAT fine-tuning 和 NAT from scratch 三类策略，并在 CIFAR-10 ResNet20 与 VoxForge CRNN-CTC 上完成训练、评估、图表和报告闭环。

综合两项任务可以得到一个明确结论：输入相关非线性误差会显著破坏模型推理精度，并且误差会在网络中逐层累积；非线性感知训练提供了将存算一体硬件非理想输入响应纳入训练闭环的可行机制，但最终收益取决于训练 alpha 分布、checkpoint 选择准则、任务类型和训练预算。在当前实验中，fine-tuning + robust validation 是更现实的第一阶段硬件适配方案，from scratch 更适合在更大训练预算下构建端到端鲁棒模型。

## 1. 背景与研究目标

存算一体芯片通过在存储阵列附近或阵列内部完成乘累加运算来降低数据搬移开销，但模拟域计算会引入器件非理想性。赛题关注的一类非理想性是输入激活相关的非线性映射：理想输入激活值在进入模拟乘累加阵列前不再保持线性响应，而会按照特定三次函数发生压缩或放大。

本文围绕两个连续任务展开：

| 任务 | 核心问题 | 实验目标 |
|---|---|---|
| 任务一 | 推理阶段非线性误差会造成多大性能损失，哪些层最敏感，误差如何累积 | 建立非线性误差敏感性分析框架 |
| 任务二 | 训练阶段显式加入非线性映射后，模型能否适应硬件非理想性 | 建立非线性感知训练与评估闭环 |

任务一回答“问题在哪里”和“误差如何传播”，任务二回答“训练能否缓解”和“哪种训练策略更现实”。二者共同构成后续鲁棒训练、敏感层校准、特征一致性约束和硬件部署适配的实验基础。

## 2. 非线性误差模型

两项任务使用同一个赛题非线性映射模型。给定输入激活 `x`，先按当前张量最大绝对值归一化：

```text
u = x / max(|x|)
```

再进行三次非线性映射：

```text
f(u) = alpha * u^3 + (1 - alpha) * u
```

最后恢复到原始幅值尺度：

```text
x' = max(|x|) * f(u)
```

等价展开为：

```text
x' = (1 - alpha) * x + alpha * x^3 / max(|x|)^2
e(x) = x' - x = alpha * (x^3 / max(|x|)^2 - x)
```

其中：

| 符号 | 含义 |
|---|---|
| `x` | 理想输入激活 |
| `x'` | 经过非线性失真后的实际输入 |
| `alpha` | 非线性强度 |
| `alpha=0` | 理想线性硬件 |
| `|alpha|` 增大 | 非线性偏离增强 |

该误差不是独立同分布随机噪声，而是与输入幅值强相关的系统性非线性失真。正负 `alpha` 对中间幅值激活的影响方向不同，因此实验分别测试正向和负向非线性。

任务二训练实现中对数值稳定性做了处理：

```python
max_val = x.detach().abs().amax().clamp_min(1e-12)
u = x / max_val
y = alpha * (u ** 3) + (1 - alpha) * u
x_nl = y * max_val
```

`max_val` 使用 `detach()`，避免训练过程通过归一化尺度传播不必要的梯度；`clamp_min(1e-12)` 用于防止全零激活导致除零。

![不同 alpha 下的三次非线性映射曲线](../outputs/task1/figures/nonlinearity_curves.png)

## 3. 工程注入方式与统一评估指标

### 3.1 算子输入端注入

非线性误差被注入到矩阵计算算子的输入端，与存算一体芯片中“输入激活值进入模拟乘累加阵列后发生非理想响应”的物理含义一致。

对于任意目标层 `module`，原始前向传播为：

```text
y = module(x)
```

注入非线性后变为：

```text
x_nonlinear = nonlinearity(x, alpha)
y = module(x_nonlinear)
```

实现采用 PyTorch forward pre-hook，不修改模型结构和权重。覆盖算子包括：

```text
Conv1d
Conv2d
ConvTranspose1d
ConvTranspose2d
Linear
```

不同任务的主要影响层如下：

| 任务 | 主要算子 |
|---|---|
| CIFAR-10 图像分类 | `Conv2d`、`Linear` |
| VoxForge Whisper 推理敏感性分析 | `Conv1d`、`Linear` |
| VoxForge CRNN-CTC 训练 | log-mel 后端卷积前端、CTC 线性输出头 |

### 3.2 分类指标

CIFAR-10 图像分类使用以下指标：

```text
Top-1 Accuracy
Cross Entropy Loss
Accuracy Drop = Acc(alpha=0) - Acc(alpha)
Worst Accuracy = min_alpha Acc(alpha)
Discrete Robust AUC = mean_alpha Acc(alpha)
```

任务二进一步报告：

```text
Clean Accuracy
Average Robust Accuracy
Worst-case Accuracy
Max Accuracy Drop
```

### 3.3 语音识别指标

VoxForge ASR 使用：

```text
WER = Word Error Rate
CER = Character Error Rate
CTC Loss
```

需要注意，WER 可以大于 1，因为当插入错误较多时，编辑距离可能超过参考文本词数。对于任务二的小规模 CRNN-CTC 实验，WER 接近 1，CER 能更细粒度反映字符级识别改善，因此报告同时使用 WER 和 CER。

### 3.4 层级漂移指标

层级误差传播分析使用：

```text
relative_l2 = ||h'_l - h_l||_2 / ||h_l||_2
cosine_drift = 1 - cos(h'_l, h_l)
JS divergence = clean 激活分布与 nonlinear 激活分布之间的 Jensen-Shannon divergence
mean shift = mean(h'_l) - mean(h_l)
std ratio = std(h'_l) / std(h_l)
```

其中 `h_l` 表示 clean 模型第 `l` 层输出，`h'_l` 表示非线性注入后对应层输出。

## 4. 任务一：非线性误差敏感性分析

### 4.1 任务一实验目标

任务一面向推理阶段，重点回答三个问题：

1. 不同非线性强度 `alpha` 对整网推理精度的影响趋势是什么？
2. 单个层独立受到非线性扰动时，哪些层对最终精度最敏感？
3. 当全网算子同时受到非线性扰动时，误差如何在网络中逐层累积？

### 4.2 任务一实验对象

任务一覆盖 CIFAR-10 图像分类和 VoxForge Spanish 语音识别。

**CIFAR-10 模型配置**

| 模型 | 类型 | 参数量 | 非线性注入算子数 | 测试样本数 | 评价指标 |
|---|---:|---:|---:|---:|---|
| `cifar10_resnet20` | CNN/ResNet | 272,474 | 22 | 10,000 | Top-1 Accuracy / Loss |
| `cifar10_mobilenetv2_x1_0` | CNN/MobileNetV2 | 2,236,682 | 53 | 10,000 | Top-1 Accuracy / Loss |

**VoxForge 模型配置**

| 数据集 | 模型 | 类型 | 参数量 | 非线性注入算子数 | 样本数 | 评价指标 |
|---|---|---|---:|---:|---:|---|
| `ciempiess/voxforge_spanish` | `openai/whisper-tiny` | Encoder-Decoder Transformer ASR | 37,760,640 | 67 | 32 | WER / CER |

VoxForge 部分最初也预留了 Wav2Vec2/CTC 模型脚本，但大型模型在当前环境中下载和加载耗时较长。因此任务一采用 Whisper-tiny 完成 VoxForge 推理敏感性实验闭环。Whisper-tiny 虽不是 VoxForge 专门微调模型，但能够代表 Transformer ASR 模型在 VoxForge 数据上的非线性推理敏感性。

## 5. 任务一整网推理精度衰减

下图作为任务一整体敏感性结果的统一摘要：左图给出不同 `alpha` 下的三次非线性映射，中图给出 CIFAR-10 两个分类模型的整网精度衰减，右图给出 VoxForge Whisper-tiny 的 WER/CER 变化。它把“非线性函数形状 - 图像分类精度 - 语音识别错误率”放在同一视角下，便于直接观察：只要硬件输入端存在系统性非线性，模型输出会从 clean 状态快速退化到接近失效状态。

![统一图 1：非线性映射与跨任务整网敏感性](../outputs/unified_paper/figures/fig01_distortion_and_sensitivity.png)

### 5.1 CIFAR-10 整网 alpha 扫描

CIFAR-10 中，对所有目标算子输入同时注入非线性，扫描：

```text
alpha ∈ {-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0}
```

![CIFAR-10 上不同 alpha 对整网 Top-1 Accuracy 的影响](../outputs/task1/figures/cifar_accuracy_alpha.png)

![CIFAR-10 上不同 alpha 对整网 Accuracy Drop 的影响](../outputs/task1/figures/cifar_accuracy_drop.png)

**CIFAR-10 整网 alpha 扫描结果**

| alpha | MobileNetV2 Accuracy | ResNet20 Accuracy | MobileNetV2 Loss | ResNet20 Loss |
|---:|---:|---:|---:|---:|
| -1.0000 | 0.0965 | 0.1138 | 81.3426 | 220.3268 |
| -0.8000 | 0.1006 | 0.1149 | 84.4736 | 75.5399 |
| -0.6000 | 0.1001 | 0.1082 | 87.7288 | 23.9976 |
| -0.4000 | 0.0983 | 0.1186 | 61.2827 | 8.1620 |
| -0.2000 | 0.1144 | 0.5041 | 27.3104 | 2.1495 |
| -0.1000 | 0.1836 | 0.8332 | 4.0344 | 0.6407 |
| 0.0000 | 0.9405 | 0.9259 | 0.2442 | 0.2815 |
| 0.1000 | 0.2191 | 0.7997 | 2.7336 | 0.7264 |
| 0.2000 | 0.1000 | 0.2031 | 2.9227 | 3.7342 |
| 0.4000 | 0.1000 | 0.1000 | 2.6162 | 4.5845 |
| 0.6000 | 0.1000 | 0.1000 | 2.4502 | 3.8725 |
| 0.8000 | 0.1000 | 0.1007 | 2.3706 | 3.6002 |
| 1.0000 | 0.1000 | 0.1000 | 2.3359 | 3.4172 |

从结果可以看出，ResNet20 的 clean accuracy 为 0.9259，MobileNetV2 的 clean accuracy 为 0.9405。在强非线性条件下，两者均下降到约 0.10，接近 CIFAR-10 十分类随机猜测水平。MobileNetV2 的弱扰动鲁棒性明显更差：`alpha=-0.1` 时 accuracy 仅为 0.1836，`alpha=0.1` 时为 0.2191；而 ResNet20 在相同条件下仍分别保持 0.8332 和 0.7997。

![CIFAR-10 模型 clean、平均鲁棒和最坏精度汇总](../outputs/task1/figures/cifar_accuracy_summary.png)

**CIFAR-10 鲁棒性汇总**

| 模型 | Clean Accuracy | Mean Accuracy | Worst Accuracy | Max Accuracy Drop | Discrete Robust AUC | 参数量 | 注入算子数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cifar10_mobilenetv2_x1_0` | 0.9405 | 0.1810 | 0.0965 | 0.8440 | 0.1810 | 2,236,682 | 53 |
| `cifar10_resnet20` | 0.9259 | 0.3171 | 0.1000 | 0.8259 | 0.3171 | 272,474 | 22 |

参数量更大的 MobileNetV2 并没有表现出更强的非线性容忍能力。相反，它的平均鲁棒精度明显低于 ResNet20。这表明网络结构、算子链条、层间归一化、残差连接和 depthwise/pointwise 设计等因素比单纯参数量更能影响非线性误差鲁棒性。

### 5.2 VoxForge Whisper 整网趋势

VoxForge 部分在 Spanish VoxForge 数据上抽取 32 条样本，使用 Whisper-tiny 进行推理。实验扫描：

```text
alpha ∈ {-1.0, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1.0}
```

![VoxForge Whisper-tiny 上不同 alpha 对 WER 的影响](../outputs/task1/figures/voxforge_whisper_wer_alpha.png)

**VoxForge Whisper-tiny alpha 扫描结果**

| alpha | WER | CER | 样本数 |
|---:|---:|---:|---:|
| -1.0000 | 1.0000 | 1.0000 | 32 |
| -0.5000 | 1.0000 | 1.0000 | 32 |
| -0.2000 | 1.0017 | 0.9820 | 32 |
| -0.1000 | 1.1304 | 0.6764 | 32 |
| 0.0000 | 0.3293 | 0.1102 | 32 |
| 0.1000 | 1.1492 | 1.0188 | 32 |
| 0.2000 | 1.0000 | 1.0000 | 32 |
| 0.5000 | 1.0000 | 1.0000 | 32 |
| 1.0000 | 1.0000 | 1.0000 | 32 |

Whisper-tiny 在 clean 条件下的 WER 为 0.3293，CER 为 0.1102。当 `alpha=±0.1` 时，WER 已超过 1.0；当 `|alpha| >= 0.2` 时，模型基本进入失效状态。语音识别任务对非线性误差更加敏感，原因可能在于 encoder-decoder ASR 的误差传播链条更长：前端声学表示受到扰动后，不仅影响 encoder 表示，还会通过 decoder 的自回归生成过程进一步放大。

## 6. 任务一单层敏感性与激活漂移

任务一的层级分析用于回答“整体失效从哪里开始、如何传播”。下图将 ResNet20 在 `alpha=1.0` 下的最高敏感层和全网注入后的 cosine drift 放在一起：左侧显示 stage 起始层和中后部卷积层会造成最大精度下降，右侧显示全网扰动会沿层级持续改变表示方向。这说明非线性误差不是只影响最后分类头，而是在中间表征形成过程中逐层累积。

![统一图 2：ResNet20 单层敏感性与表示漂移](../outputs/unified_paper/figures/fig02_layer_sensitivity_and_drift.png)

### 6.1 ResNet20 单层敏感性

整网精度曲线只能说明模型整体对非线性误差敏感，但不能回答哪些层最脆弱。因此任务一在 ResNet20 上进一步进行单层敏感性分析：每次只对一个目标层的输入注入非线性，其余层保持理想线性状态，记录最终分类精度下降。实验样本数为 2,048。

![ResNet20 单层非线性敏感性热力图](../outputs/task1/figures/cifar10_resnet20_layer_sensitivity_heatmap.png)

**ResNet20 单层敏感性按 alpha 汇总**

| alpha | Mean Accuracy Drop | Max Accuracy Drop | 最高敏感层 |
|---:|---:|---:|---|
| -1.0 | 0.0939 | 0.3286 | `layer2.0.conv2` |
| -0.5 | 0.0254 | 0.0879 | `layer2.0.conv2` |
| 0.5 | 0.0498 | 0.1855 | `layer2.0.conv2` |
| 1.0 | 0.4427 | 0.8145 | `layer3.0.conv1` |

**alpha=1.0 下最敏感的 ResNet20 层**

| 排名 | 层 | 单层注入后 Accuracy | Accuracy Drop |
|---:|---|---:|---:|
| 1 | `layer3.0.conv1` | 0.1089 | 0.8145 |
| 2 | `layer1.0.conv1` | 0.1113 | 0.8120 |
| 3 | `layer2.0.conv1` | 0.1294 | 0.7939 |
| 4 | `layer2.0.conv2` | 0.1436 | 0.7798 |
| 5 | `layer3.1.conv1` | 0.2207 | 0.7026 |
| 6 | `layer3.2.conv2` | 0.2583 | 0.6650 |

单层敏感性实验表明，不同层对非线性误差的容忍能力差异显著。各 stage 的开始位置通常更敏感，例如 `layer1.0.conv1`、`layer2.0.conv1`、`layer3.0.conv1`。这些层处于特征尺度或语义阶段变化的边界，输入分布被扰动后会影响后续整个 stage。相比之下，最终 `fc` 层在单层注入下几乎不敏感，说明分类头单独受到该映射影响时不会像中间卷积层那样引起大规模特征重构错误。

### 6.2 激活分布偏移

为了观察非线性误差对中间激活分布的影响，实验记录 clean 模型和 nonlinear 模型在代表层上的输出分布。选取浅层 `conv1`、中层 `layer2.1.conv2` 和末层 `fc` 进行展示。

![ResNet20 conv1 层在 alpha=1.0 下的激活分布偏移](../outputs/task1/figures/cifar10_resnet20_activation_hist_conv1_alpha_1.png)

![ResNet20 layer2.1.conv2 层在 alpha=1.0 下的激活分布偏移](../outputs/task1/figures/cifar10_resnet20_activation_hist_layer2_1_conv2_alpha_1.png)

![ResNet20 fc 层在 alpha=1.0 下的激活分布偏移](../outputs/task1/figures/cifar10_resnet20_activation_hist_fc_alpha_1.png)

浅层卷积输出在非线性注入后已经出现分布压缩和形状偏移，但仍保留部分 clean 分布轮廓。中层特征分布偏移更加明显，说明前面层的非线性误差已经传播到中层，并与当前层输入非线性叠加。接近分类输出的特征分布已经明显偏离 clean 状态，由于分类决策依赖 logits 之间的相对大小和方向，末层特征分布偏移会直接导致类别预测错误。

### 6.3 ResNet20 逐层误差累积

在全网非线性注入条件下，记录每个目标层的输出，并与 clean 模型对应层输出比较。

![ResNet20 全网非线性注入后的逐层 relative L2 漂移](../outputs/task1/figures/cifar10_resnet20_activation_relative_l2.png)

![ResNet20 全网非线性注入后的逐层 cosine drift](../outputs/task1/figures/cifar10_resnet20_activation_cosine_drift.png)

**ResNet20 全网注入后的逐层误差统计汇总**

| alpha | Mean Relative L2 | Max Relative L2 | Mean Cosine Drift | Max Cosine Drift | Mean JS Divergence | Max JS Divergence | Mean Mean Shift | Max Mean Shift | Mean Std Ratio | Max Std Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1.0000 | 41.4098 | 315.4180 | 0.5000 | 0.9858 | 0.3913 | 0.6727 | -6.4390 | 0.0022 | 35.6950 | 222.7964 |
| -0.5000 | 3.6615 | 13.3577 | 0.3490 | 0.9952 | 0.1893 | 0.5725 | -0.4343 | 0.0011 | 3.9545 | 11.1321 |
| 0.5000 | 0.8014 | 1.1193 | 0.3971 | 1.0029 | 0.1250 | 0.3319 | 0.0469 | 0.1570 | 0.4445 | 0.7409 |
| 1.0000 | 1.0042 | 1.7283 | 0.6552 | 1.0056 | 0.2518 | 0.5185 | 0.0720 | 0.2836 | 0.3632 | 1.3626 |

**ResNet20 relative L2 最大的层**

| 层 | alpha | Relative L2 | Cosine Drift | JS Divergence |
|---|---:|---:|---:|---:|
| `layer3.2.conv1` | -1.0000 | 315.4180 | 0.2655 | 0.6727 |
| `layer3.1.conv2` | -1.0000 | 174.6765 | 0.4410 | 0.6013 |
| `layer3.0.conv2` | -1.0000 | 86.6685 | 0.6674 | 0.6062 |
| `layer2.2.conv2` | -1.0000 | 59.3146 | 0.8920 | 0.5776 |
| `layer3.1.conv1` | -1.0000 | 46.1393 | 0.6608 | 0.5768 |
| `layer3.0.conv1` | -1.0000 | 42.2137 | 0.6033 | 0.5787 |
| `layer3.0.downsample.0` | -1.0000 | 37.2747 | 0.6147 | 0.5464 |
| `layer2.1.conv2` | -1.0000 | 25.9239 | 0.5411 | 0.5051 |
| `fc` | -1.0000 | 22.8117 | 0.9858 | 0.5526 |
| `layer3.2.conv2` | -1.0000 | 20.4877 | 0.9648 | 0.2263 |

`alpha=-1.0` 会导致极大的幅值漂移，平均 relative L2 达到 41.4098，最大值达到 315.4180，说明放大型非线性会在深层网络中严重累积。`alpha=1.0` 的 relative L2 虽然没有负 alpha 那么极端，但平均 cosine drift 达到 0.6552，说明特征方向被明显破坏。负 alpha 主要体现为幅值尺度失控，正 alpha 则更明显地改变特征方向，二者都会导致最终分类边界失效。

### 6.4 Whisper-tiny Transformer 层级误差传播

除最终 WER/CER 外，任务一还对 Whisper-tiny 的前 32 个目标层进行了层级漂移分析。该分析使用 1 条语音样本进行 clean/nonlinear 对齐比较，目的是观察 Transformer ASR 模型中哪些算子输出更容易受到输入非线性的影响。

![Whisper-tiny 全网非线性注入后的逐层 relative L2 漂移](../outputs/task1/figures/voxforge_whisper_activation_relative_l2.png)

![Whisper-tiny 全网非线性注入后的逐层 cosine drift](../outputs/task1/figures/voxforge_whisper_activation_cosine_drift.png)

**Whisper-tiny 层级漂移统计汇总**

| alpha | Mean Relative L2 | Max Relative L2 | Mean Cosine Drift | Max Cosine Drift | Mean JS Divergence | Max JS Divergence |
|---:|---:|---:|---:|---:|---:|---:|
| -1.0000 | 2.2088 | 6.2660 | 0.5326 | 1.2340 | 0.1181 | 0.3279 |
| 0.5000 | 0.7270 | 1.2677 | 0.3686 | 0.9438 | 0.0476 | 0.1707 |
| 1.0000 | 0.8141 | 1.5782 | 0.4429 | 0.9959 | 0.1535 | 0.3741 |

**Whisper-tiny relative L2 最大的层**

| 层 | alpha | Relative L2 | Cosine Drift | JS Divergence |
|---|---:|---:|---:|---:|
| `model.encoder.layers.0.fc2` | -1.0000 | 6.2660 | 0.7304 | 0.3279 |
| `model.encoder.layers.2.fc2` | -1.0000 | 5.3774 | 0.7710 | 0.1973 |
| `model.encoder.layers.1.fc2` | -1.0000 | 4.6837 | 0.5768 | 0.3142 |
| `model.encoder.layers.1.self_attn.out_proj` | -1.0000 | 4.6816 | 0.7941 | 0.2586 |
| `model.encoder.layers.2.self_attn.out_proj` | -1.0000 | 4.3255 | 0.8187 | 0.2881 |
| `model.encoder.layers.3.self_attn.out_proj` | -1.0000 | 4.3243 | 0.8923 | 0.2151 |
| `model.encoder.layers.0.self_attn.out_proj` | -1.0000 | 3.4543 | 0.4195 | 0.2656 |
| `model.encoder.layers.2.self_attn.v_proj` | -1.0000 | 2.9808 | 0.9807 | 0.1663 |
| `model.encoder.layers.1.self_attn.v_proj` | -1.0000 | 2.7017 | 0.9481 | 0.1451 |
| `model.encoder.layers.3.self_attn.v_proj` | -1.0000 | 2.1151 | 0.9475 | 0.0352 |

Whisper-tiny 的层级漂移结果显示，encoder 前几层的 feed-forward `fc2` 和 self-attention 输出投影层对输入非线性较敏感。这与 Transformer 的结构特点一致：attention 和 feed-forward 模块交替堆叠，早期 encoder 表示一旦发生方向和尺度偏移，会影响后续全部 token 表示，并最终影响 decoder 的文本生成。

## 7. 任务一阶段性结论

任务一得到以下结论：

1. 输入相关非线性误差会显著破坏神经网络推理精度。CIFAR-10 中，ResNet20 从 clean 条件下的 92.59% 下降到最差 10.00%，MobileNetV2 从 94.05% 下降到最差 9.65%，接近随机猜测。
2. 非线性强度与任务性能退化之间存在明显阈值效应。模型并不是随 `alpha` 线性缓慢退化，而是在某些弱扰动区间内就出现急剧性能下降。
3. 不同网络结构的敏感性差异明显。MobileNetV2 在 `alpha=±0.1` 附近已经严重退化，ResNet20 相对更稳健。
4. 单层敏感性具有明显结构位置差异。ResNet20 中 stage 起始层和部分中后段卷积层对非线性误差最敏感，例如 `layer3.0.conv1`、`layer1.0.conv1`、`layer2.0.conv1`。
5. 误差会在网络中逐层累积，并表现出不同机制。负 `alpha` 更容易导致幅值尺度失控，正 `alpha` 更明显破坏特征方向。
6. 语音识别任务比图像分类更加脆弱。Whisper-tiny 在 VoxForge 上 `alpha=±0.1` 即出现 WER 大幅恶化，说明 encoder-decoder ASR 的自回归生成过程会放大前端表示偏移。

任务一直接为任务二提供依据：非线性感知训练应覆盖多种 `alpha` 强度，不能只针对单点；训练和验证应关注高敏感层、正负 alpha 方向差异和 clean-robust trade-off。

## 8. 任务二：非线性感知训练

### 8.1 任务二实验目标

任务二面向训练阶段，研究显式加入非线性映射模型后，神经网络对推理非线性扰动的适应能力。与任务一只在推理阶段注入非线性误差不同，任务二将非线性映射纳入训练闭环，重点比较三类策略：

| 方法 | 训练阶段设置 | 目的 |
|---|---|---|
| Clean Training | 训练阶段不加入非线性 | clean baseline |
| NAT Fine-tuning | 先得到 clean checkpoint，再加入非线性扰动继续微调 | 低成本硬件适配 |
| NAT From Scratch | 模型从随机初始化开始就在训练阶段加入非线性扰动 | 端到端鲁棒训练 |

本轮实验覆盖两个任务：

| 数据集 | 任务 | 模型 | 主要指标 |
|---|---|---|---|
| CIFAR-10 | 图像分类 | ResNet20 | Top-1 Accuracy / Robust Accuracy |
| VoxForge | 语音识别 | CRNN-CTC | WER / CER / CTC Loss |

当前报告对应首轮完整闭环实验：CIFAR-10 使用 60 epoch clean/scratch 和 20 epoch fine-tuning；VoxForge 使用小规模固定子集 `120/30/30` 完成 ASR 训练闭环。代码已经支持将 CIFAR-10 扩展到 160/200 epoch，将 VoxForge 扩展到更大子集和 50/100 epoch。

### 8.2 Alpha 采样策略

任务二实现了统一的 alpha scheduler：

| 调度器 | 形式 | 用途 |
|---|---|---|
| NoAlphaScheduler | `alpha=0` | clean baseline |
| FixedAlphaScheduler | 固定 alpha | 单硬件点适配 |
| UniformAlphaScheduler | `alpha ~ Uniform(low, high)` | 随机非线性训练 |
| CurriculumAlphaScheduler | 扰动范围随 epoch 逐步增大 | 改善从头训练稳定性 |

旧版首轮闭环实验使用过如下全局随机 alpha 训练：

```text
CIFAR-10: alpha_train ~ Uniform(-0.2, 0.2)
VoxForge: alpha_train ~ Uniform(-0.1, 0.1)
```

VoxForge 使用更保守范围，是因为任务一中语音模型对 `alpha=±0.1` 已表现出明显敏感性。该设置现在只作为历史基线；正式主协议必须在每个矩阵算子输入端、每次 forward 调用独立采样 `alpha_{l,c}~Uniform(-1,1)`。

### 8.3 工程实现

任务二新增与复用代码如下：

| 文件 | 作用 |
|---|---|
| `src/nonlinear.py` | 非线性映射、forward pre-hook 注入器 |
| `src/alpha_scheduler.py` | 训练阶段 alpha 采样策略 |
| `src/training.py` | checkpoint、CSV、JSON、随机种子等通用工具 |
| `src/cifar_models.py` | CIFAR-10 dataloader 与模型构建 |
| `scripts/train_cifar_task2.py` | CIFAR-10 clean/NAT 训练、评估和 alpha sweep |
| `src/voxforge_data.py` | VoxForge streaming 子集缓存、字符词表、collate |
| `src/asr_models.py` | CRNN-CTC 语音识别模型 |
| `scripts/train_voxforge_task2.py` | VoxForge clean/NAT 训练、评估和 alpha sweep |
| `scripts/make_task2_figures.py` | 根据 CSV 生成任务二图表 |

CIFAR-10 主模型使用 ResNet20。训练阶段每个 batch 根据 alpha scheduler 采样一个 alpha，并通过 `NonlinearityInjector` 将非线性映射注入到目标算子的输入端。

除常规按 clean validation accuracy 选择 checkpoint 外，本轮还增加了 robust validation 选择方式：

```text
selection_score = mean(Acc_val(alpha=-0.2), Acc_val(alpha=0), Acc_val(alpha=0.2))
```

该设置用于验证：NAT 训练是否需要使用硬件鲁棒性指标，而不是只用 clean validation accuracy 选模型。

VoxForge 使用 Hugging Face 数据集 `ciempiess/voxforge_spanish`，字段包括：

```text
audio
normalized_text
duration
speaker_id
```

本轮为了快速完成可复现实验闭环，使用固定随机种子抽取小规模子集：

```text
train=120, valid=30, test=30
max_duration=5s
max_text_length=110 characters
```

语音模型采用轻量 CRNN-CTC：

```text
waveform
  -> 80-bin log-mel spectrogram
  -> 2-layer Conv2d subsampling
  -> 2-layer BiGRU
  -> Linear CTC head
  -> CTC loss / greedy decoding
```

为了避免 CTC 早期全部输出 blank，分类头初始化时对 blank 类加入轻微负 bias。VoxForge 评估输出 WER、CER、CTC loss 和解码样例。

## 9. 任务二旧版首轮实验配置

本节保留的是旧版全局/窄范围 alpha NAT 闭环数据，用于说明训练管线、收敛曲线和历史基线；它不是每算子每次独立随机 `alpha~Uniform(-1,1)` 的最终主协议结果。

### 9.1 CIFAR-10 旧版配置

| 方法 | 初始化 | 训练 alpha | Epoch | Checkpoint 选择 |
|---|---|---|---:|---|
| Clean | 随机初始化 | 无 | 60 | clean validation accuracy |
| NAT Fine-tuning | Clean best checkpoint | `Uniform(-0.2, 0.2)` | 20 | clean validation accuracy |
| NAT From Scratch | 随机初始化 | `Uniform(-0.2, 0.2)` | 60 | clean validation accuracy |
| NAT FT Robust-selected | Clean best checkpoint | `Uniform(-0.2, 0.2)` | 20 | `alpha=-0.2,0,0.2` 平均验证精度 |

CIFAR-10 测试 alpha 网格：

```text
[-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5]
```

### 9.2 VoxForge 旧版配置

| 方法 | 初始化 | 训练 alpha | Epoch | 子集规模 |
|---|---|---|---:|---|
| Clean | 随机初始化 | 无 | 30 | `120/30/30` |
| NAT Fine-tuning | Clean best checkpoint | `Uniform(-0.1, 0.1)` | 10 | `120/30/30` |
| NAT From Scratch | 随机初始化 | `Uniform(-0.1, 0.1)` | 30 | `120/30/30` |

VoxForge 测试 alpha 网格：

```text
[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
```

## 10. 任务二 CIFAR-10 结果

### 10.1 收敛曲线

![CIFAR-10 Training Loss](../outputs/task2/figures/cifar_train_loss.png)

![CIFAR-10 Validation Accuracy](../outputs/task2/figures/cifar_val_accuracy.png)

从训练曲线看，NAT fine-tuning 由于从 clean checkpoint 初始化，起始 loss 明显低于从头训练，20 epoch 内即可保持接近 clean baseline 的验证精度。NAT from scratch 在 60 epoch 内也能达到接近 clean 的验证精度，说明将非线性注入训练并不会阻止模型学习 clean 判别特征。

### 10.2 Alpha-Accuracy 曲线

![CIFAR-10 Accuracy under Inference Nonlinearity](../outputs/task2/figures/cifar_alpha_accuracy.png)

![CIFAR-10 Accuracy Drop](../outputs/task2/figures/cifar_accuracy_drop.png)

CIFAR-10 的主要现象是：所有方法在强非线性 `|alpha|>=0.3` 下仍然明显退化；NAT 的收益主要集中在训练范围附近，尤其是 `alpha=-0.2` 和 `alpha=-0.1`。

### 10.3 CIFAR-10 汇总

| 方法 | Epoch | Clean Acc | Avg Robust Acc | Worst Acc | Max Drop |
|:---|---:|---:|---:|---:|---:|
| Clean | 60 | 0.8939 | 0.3141 | 0.1000 | 0.7939 |
| NAT FT | 20 | 0.8941 | 0.3122 | 0.1079 | 0.7862 |
| NAT Scratch | 60 | 0.8965 | 0.2812 | 0.0935 | 0.8030 |
| NAT FT RobustSel | 20 | 0.8832 | 0.3175 | 0.1000 | 0.7832 |

下图汇总了任务二的两类基线结果。CIFAR-10 中，NAT fine-tuning 可以基本保持 clean accuracy，但平均鲁棒精度提升有限；VoxForge 中，CER 指标显示 NAT fine-tuning 对字符级识别更有帮助。两者共同说明：NAT 是可行的训练闭环，但是否提升鲁棒性取决于训练扰动范围、checkpoint 选择指标和任务本身的误差放大机制。

![统一图 3：CIFAR-10 与 VoxForge NAT 基线对比](../outputs/unified_paper/figures/fig03_nat_baseline.png)

![CIFAR-10 Summary](../outputs/task2/figures/cifar_method_summary.png)

普通 NAT fine-tuning 保持了 clean accuracy，但平均鲁棒精度没有明显提升，说明只按 clean validation accuracy 选 checkpoint 会偏向 clean 性能。NAT FT Robust-selected 牺牲约 `1.09%` clean accuracy，但平均鲁棒精度从 `0.3122` 提升到 `0.3175`，最大精度下降从 `0.7862` 降到 `0.7832`。NAT from scratch 在当前 60 epoch 预算下 clean accuracy 不差，但平均鲁棒精度最低，说明从头鲁棒训练需要更长训练预算或 curriculum alpha。

### 10.4 CIFAR-10 Alpha 逐点数据

| Alpha | Clean | NAT FT | NAT FT RobustSel | NAT Scratch |
|---:|---:|---:|---:|---:|
| -0.50 | 0.1305 | 0.1151 | 0.1191 | 0.1004 |
| -0.30 | 0.1685 | 0.1630 | 0.1936 | 0.1025 |
| -0.20 | 0.3271 | 0.3319 | 0.4562 | 0.2165 |
| -0.10 | 0.7106 | 0.7229 | 0.8287 | 0.7794 |
| 0.00 | 0.8939 | 0.8941 | 0.8832 | 0.8965 |
| 0.10 | 0.7290 | 0.7334 | 0.5584 | 0.6536 |
| 0.20 | 0.2285 | 0.2142 | 0.1798 | 0.2037 |
| 0.30 | 0.1185 | 0.1090 | 0.1046 | 0.0997 |
| 0.50 | 0.1000 | 0.1079 | 0.1000 | 0.0935 |

逐点结果显示，robust-selected fine-tuning 在负 alpha 区域收益明显：`alpha=-0.2` 从 clean baseline 的 `0.3271` 提升到 `0.4562`，`alpha=-0.1` 从 `0.7106` 提升到 `0.8287`。但它对正 alpha 区域有牺牲，说明当前单一 `Uniform(-0.2,0.2)` 训练和三点 robust validation 仍然不能同时覆盖两个方向的最优鲁棒性。

## 11. 任务二 VoxForge 结果

### 11.1 收敛曲线

![VoxForge Training Loss](../outputs/task2/figures/voxforge_train_loss.png)

![VoxForge Validation WER](../outputs/task2/figures/voxforge_val_wer.png)

VoxForge 的训练损失显示 clean 和 NAT scratch 都能从约 5 降到约 2.08；NAT fine-tuning 从 clean checkpoint 出发，10 epoch 后训练 loss 进一步降到 `1.9903`。但由于训练集和验证集都很小，WER 仍接近 1，因此报告同时使用更细粒度的 CER 作为主要辅助指标。

### 11.2 WER/CER 曲线

![VoxForge WER under Inference Nonlinearity](../outputs/task2/figures/voxforge_alpha_wer.png)

![VoxForge CER under Inference Nonlinearity](../outputs/task2/figures/voxforge_alpha_cer.png)

VoxForge 的 WER 曲线变化范围较窄，主要因为小规模 CTC ASR 还没有充分学到词级边界；CER 更能反映字符级识别改善。NAT fine-tuning 在所有测试 alpha 的 CER 上均优于 clean baseline，说明训练阶段加入非线性扰动能改善语音识别模型的字符级鲁棒性。

### 11.3 VoxForge 汇总

| 方法 | Epoch | Train/Val/Test | Clean WER | Clean CER | Avg Robust WER | Avg Robust CER | Worst WER | Worst CER |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|
| Clean | 30 | 120/30/30 | 0.9956 | 0.7920 | 0.9884 | 0.7985 | 0.9956 | 0.8246 |
| NAT FT | 10 | 120/30/30 | 0.9913 | 0.7669 | 0.9891 | 0.7817 | 1.0000 | 0.8079 |
| NAT Scratch | 30 | 120/30/30 | 0.9956 | 0.8755 | 0.9934 | 0.8847 | 0.9956 | 0.9165 |

![VoxForge WER Summary](../outputs/task2/figures/voxforge_method_summary.png)

![VoxForge CER Summary](../outputs/task2/figures/voxforge_cer_summary.png)

NAT fine-tuning 的 clean CER 从 `0.7920` 降到 `0.7669`，平均鲁棒 CER 从 `0.7985` 降到 `0.7817`，说明在当前小规模 ASR 设置下，fine-tuning 是最有效的策略。NAT from scratch 的 CER 明显高于 clean 和 NAT fine-tuning，说明从头训练需要更大数据规模、更长 epoch 或课程式 alpha 调度。WER 接近 1，不能单独作为当前小规模实验的唯一判断依据；CER、CTC loss 和解码样例更能反映模型是否已经学习到字符级声学模式。

### 11.4 VoxForge Alpha 逐点数据

**WER**

| Alpha | Clean | NAT FT | NAT Scratch |
|---:|---:|---:|---:|
| -0.20 | 0.9782 | 0.9825 | 0.9913 |
| -0.10 | 0.9913 | 0.9869 | 0.9913 |
| -0.05 | 0.9913 | 0.9869 | 0.9956 |
| 0.00 | 0.9956 | 0.9913 | 0.9956 |
| 0.05 | 0.9913 | 0.9913 | 0.9913 |
| 0.10 | 0.9869 | 0.9869 | 0.9956 |
| 0.20 | 0.9913 | 1.0000 | 0.9956 |

**CER**

| Alpha | Clean | NAT FT | NAT Scratch |
|---:|---:|---:|---:|
| -0.20 | 0.8112 | 0.8053 | 0.9165 |
| -0.10 | 0.7970 | 0.7870 | 0.8922 |
| -0.05 | 0.7970 | 0.7728 | 0.8847 |
| 0.00 | 0.7920 | 0.7669 | 0.8755 |
| 0.05 | 0.7903 | 0.7627 | 0.8747 |
| 0.10 | 0.7711 | 0.7544 | 0.8722 |
| 0.20 | 0.8246 | 0.8079 | 0.8680 |

### 11.5 解码样例

以下样例来自 NAT fine-tuning 模型在 `alpha=0` 下的 greedy CTC 解码。模型仍处于小规模训练状态，但已经能输出与参考文本相关的部分字符片段。

| Reference | Hypothesis |
|---|---|
| de unas parras artificiales cuyas hojas parecían retazos de terciopelo | unse a eaesasas |
| a las que se acogían grupos de personas para embadurnar | a oa as |
| estaban ocupados por señoras | estan ocupors |
| unas líneas negras y oblicuas semejantes a cuerdas | uas ca eas |
| estaban ocupados por señoras | esan ocupors |

## 12. Fine-tuning 与 From Scratch 对比

### 12.1 收敛性

| 数据集 | 对比结论 |
|---|---|
| CIFAR-10 | fine-tuning 从 clean checkpoint 出发，20 epoch 内即可保持 clean accuracy；scratch 需要 60 epoch 才接近 clean baseline。 |
| VoxForge | fine-tuning 的起点 loss 更低，10 epoch 后 CER 优于 clean；scratch 在 30 epoch 小数据预算下仍明显不足。 |

Fine-tuning 的核心优势是部署成本低：已有 clean 模型只需少量硬件感知微调即可适配目标非线性范围。从头训练的优势理论上是端到端适应硬件误差族，但它需要更长训练预算，而且早期训练更容易受非线性扰动影响。

### 12.2 泛化性

CIFAR-10 显示 NAT 的泛化收益并不是自动发生的。普通 NAT fine-tuning 只按 clean validation accuracy 选模型，平均鲁棒精度没有提升；改用 robust validation 后，负 alpha 区域鲁棒性显著改善，但正 alpha 区域下降。这说明 NAT 至少包含三个关键设计点：

1. 训练 alpha 分布是否覆盖目标硬件误差。
2. checkpoint 选择指标是否包含非线性鲁棒性。
3. 是否需要区分正 alpha 和负 alpha 的误差方向。

VoxForge 中 NAT fine-tuning 在 CER 上更稳定地优于 clean baseline，说明 ASR 模型在字符级输出上能从非线性扰动训练中获益。但 WER 仍受小规模训练限制，后续需要扩大数据和 epoch 才能验证词级泛化。

### 12.3 与任务一发现的对应关系

任务一发现负 alpha 更容易造成幅值漂移，正 alpha 更明显破坏特征方向；任务二 CIFAR-10 robust-selected fine-tuning 主要改善负 alpha 区域，但牺牲正 alpha 区域。这说明训练和模型选择过程中不仅需要覆盖 `|alpha|` 范围，也需要显式考虑 `alpha` 方向。

任务一发现 stage 起始层和中后部卷积层更敏感；任务二当前使用整网均匀注入训练，尚未区分敏感层。后续可以基于任务一结果设计 sensitivity-guided NAT，对高敏感层施加更强扰动、更强一致性约束或单独校准。

任务一发现 ASR 对非线性更脆弱；旧版任务二在 VoxForge 中采用更小训练范围 `Uniform(-0.1,0.1)`，并优先报告 CER。这说明跨任务复用 NAT 思路时，训练 alpha 范围和评价指标必须适配任务特点。但正式主协议不能缩小范围，必须在 `[-1,1]` 上进行每算子每次独立随机注入。

## 13. 局限性与后续改进

当前任务一和任务二已经形成完整实验链条，但仍有以下限制：

1. CIFAR-10 任务二当前为首轮预算，clean/scratch 使用 60 epoch，低于计划中的 160/200 epoch。
2. VoxForge 任务二当前使用 `120/30/30` 小子集，足以验证训练管线和 CER 趋势，但不足以得到高质量 ASR WER。
3. NAT from scratch 在两个任务上都没有表现出稳定优势，后续应加入 curriculum NAT 和更长训练预算。
4. CIFAR-10 robust-selected fine-tuning 说明 checkpoint 选择很关键，但三点验证仍偏向负 alpha，后续可采用更多 alpha 点或 clean-robust 加权目标。
5. 当前非线性只注入 Conv/Linear 等显式算子输入，GRU 内部矩阵乘法没有逐门控注入；若要更贴近硬件部署，需要将 RNN/Transformer 内部投影拆成显式 Linear。
6. 任务一中 Whisper-tiny 是推理敏感性模型，任务二中 VoxForge 使用 CRNN-CTC 训练模型，两者 ASR 架构不同，因此 ASR 结论应理解为同一数据任务上的趋势互证，而不是严格同模型训练对照。

建议下一阶段从任务三鲁棒性增强角度继续：

```text
Loss = CE/NLL + lambda * consistency(clean_logits, nonlinear_logits)
alpha curriculum: 0 -> target range
robust checkpoint selection: mean metric over validation alpha grid
direction-aware training: separate positive-alpha and negative-alpha adapters
sensitivity-guided NAT: stronger constraints on high-sensitivity layers
BatchNorm / activation range recalibration for nonlinear deployment
```

## 14. 工程文件与产物清单

### 14.1 任务一工程文件

| 类型 | 文件路径 | 作用 |
|---|---|---|
| 非线性注入工具 | `src/nonlinear.py` | 定义非线性函数、forward pre-hook 注入器和激活记录器 |
| 指标工具 | `src/metrics.py` | 计算 accuracy、WER、CER、relative L2、cosine drift 等指标 |
| 绘图工具 | `src/plotting.py` | 生成曲线图、热力图和激活分布图 |
| CIFAR 实验脚本 | `scripts/run_cifar_task1.py` | 运行 CIFAR-10 alpha 扫描、单层敏感性和逐层漂移分析 |
| VoxForge Wav2Vec2/CTC 脚本 | `scripts/run_voxforge_task1.py` | 预留 Hugging Face CTC ASR 模型实验入口 |
| VoxForge Whisper 脚本 | `scripts/run_voxforge_whisper_task1.py` | 运行 VoxForge + Whisper-tiny alpha 扫描和层级漂移分析 |
| 图表重绘脚本 | `scripts/make_task1_figures.py` | 根据 CSV 结果重新生成图表 |
| 任务一报告 | `docs/task1_technical_report.md` | 任务一原始技术报告 |

### 14.2 任务一主要输出文件

| 类型 | 文件路径 |
|---|---|
| CIFAR alpha 扫描表 | `outputs/task1/tables/cifar_accuracy_alpha.csv` |
| CIFAR 鲁棒性汇总表 | `outputs/task1/tables/cifar_accuracy_summary.csv` |
| ResNet20 单层敏感性表 | `outputs/task1/tables/cifar10_resnet20_layer_sensitivity.csv` |
| ResNet20 激活漂移表 | `outputs/task1/tables/cifar10_resnet20_activation_drift.csv` |
| VoxForge Whisper WER/CER 表 | `outputs/task1/tables/voxforge_whisper_wer_alpha.csv` |
| VoxForge Whisper 激活漂移表 | `outputs/task1/tables/voxforge_whisper_activation_drift.csv` |
| CIFAR accuracy 曲线 | `outputs/task1/figures/cifar_accuracy_alpha.png` |
| ResNet20 单层敏感性热力图 | `outputs/task1/figures/cifar10_resnet20_layer_sensitivity_heatmap.png` |
| VoxForge WER 曲线 | `outputs/task1/figures/voxforge_whisper_wer_alpha.png` |

### 14.3 任务二工程文件

| 类型 | 路径 |
|---|---|
| 任务二规划 | `docs/task2_experiment_plan.md` |
| 任务二原始技术报告 | `docs/task2_technical_report.md` |
| CIFAR 训练脚本 | `scripts/train_cifar_task2.py` |
| VoxForge 训练脚本 | `scripts/train_voxforge_task2.py` |
| 任务二绘图脚本 | `scripts/make_task2_figures.py` |
| 统一论文图脚本 | `scripts/make_unified_paper_figures.py` |
| CIFAR 表格 | `outputs/task2/tables/cifar_training_logs.csv`, `cifar_alpha_eval.csv`, `cifar_method_summary.csv` |
| VoxForge 表格 | `outputs/task2/tables/voxforge_training_logs.csv`, `voxforge_alpha_eval.csv`, `voxforge_method_summary.csv`, `voxforge_decode_examples.csv` |
| 图表目录 | `outputs/task2/figures/` |
| 统一论文图目录 | `outputs/unified_paper/figures/` |
| Checkpoint 目录 | `outputs/task2/checkpoints/` |

### 14.4 综合报告

| 类型 | 路径 |
|---|---|
| 任务一与任务二综合技术报告 | `docs/task1_task2_integrated_technical_report.md` |

## 15. 复现方式

### 15.1 安装依赖

```powershell
python -m pip install -r requirements-task1.txt
```

### 15.2 任务一 CIFAR-10 主实验

```powershell
python scripts\run_cifar_task1.py `
  --models cifar10_resnet20,cifar10_mobilenetv2_x1_0 `
  --max-samples 10000 `
  --batch-size 512 `
  --workers 0 `
  --analysis-model cifar10_resnet20 `
  --analysis-max-layers 22 `
  --analysis-samples 2048 `
  --analysis-batch-size 256 `
  --activation-keep-batches 1 `
  --output-dir outputs\task1
```

### 15.3 任务一 VoxForge Whisper 主实验

```powershell
python scripts\run_voxforge_whisper_task1.py `
  --streaming `
  --max-samples 32 `
  --max-new-tokens 48 `
  --alphas=-1.0,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.0 `
  --skip-activation-analysis `
  --output-dir outputs\task1
```

### 15.4 任务一图表重绘

```powershell
python scripts\make_task1_figures.py --output-dir outputs\task1 --analysis-model cifar10_resnet20
```

### 15.5 任务二图表重绘

```powershell
python scripts\make_task2_figures.py --output-dir outputs\task2
```

### 15.6 统一论文图重绘

```powershell
python scripts\make_unified_paper_figures.py
```

任务二训练脚本已经保存训练日志、alpha sweep、method summary、decode examples 和 checkpoint，可通过 `scripts/train_cifar_task2.py` 与 `scripts/train_voxforge_task2.py` 复现实验或扩展训练预算。

## 16. 综合结论

任务一和任务二共同说明：

1. 输入相关非线性误差是存算一体神经网络部署中的关键风险。它会在 CIFAR-10 上将高精度分类模型推到接近随机猜测，在 VoxForge ASR 上则会使 WER/CER 快速恶化。
2. 非线性误差的破坏不是简单均匀噪声效应，而是与输入幅值、网络结构位置和误差方向相关。负 alpha 更容易引起幅值失控，正 alpha 更容易破坏特征方向。
3. 模型鲁棒性与参数量没有简单正相关。任务一中 MobileNetV2 参数量大于 ResNet20，但对弱非线性更加敏感。
4. 高敏感层集中在 stage 起始层、中后部卷积层和 Transformer encoder 早期投影层。后续训练和校准应考虑 layer-wise sensitivity。
5. 非线性感知训练是可行的，但不是只要加入扰动就必然提升所有鲁棒指标。任务二 CIFAR-10 普通 NAT fine-tuning 保持 clean accuracy，却没有明显提升平均鲁棒精度；robust validation 能改善负 alpha 区域，但带来正 alpha 与 clean 性能 trade-off。
6. Fine-tuning 是当前最稳定的部署适配策略。CIFAR-10 中 fine-tuning 能保持 clean accuracy；VoxForge 中 fine-tuning 能降低 clean CER 和平均鲁棒 CER。
7. From scratch 在当前预算下不如 fine-tuning，尤其在 VoxForge 小数据 ASR 中更明显。它需要更大数据、更长训练和 curriculum alpha 才能充分体现端到端鲁棒训练优势。
8. 实际部署中，建议采用“任务一敏感性分析 + 任务二 fine-tuning + robust validation”的流程：先定位敏感 alpha、敏感层和误差方向，再用覆盖目标硬件误差范围的 NAT 微调，并用 alpha 网格上的鲁棒验证指标选择 checkpoint。

因此，本文最终结论可以概括为：

```text
非线性误差敏感性分析揭示了存算一体输入非理想响应如何破坏模型表示；
非线性感知训练提供了将该硬件误差纳入训练闭环的机制。
二者结合后，可以从推理失效分析走向部署前硬件适配。
当前最实际的方案是 NAT fine-tuning + robust validation；
更进一步的鲁棒模型需要 curriculum alpha、敏感层约束和 clean/nonlinear 一致性训练。
```

## 17. 参考资料

1. CIFAR-10 official dataset: https://www.cs.toronto.edu/~kriz/cifar.html
2. PyTorch CIFAR models: https://github.com/chenyaofo/pytorch-cifar-models
3. VoxForge: http://www.voxforge.org/
4. Hugging Face VoxForge Spanish dataset: https://huggingface.co/datasets/ciempiess/voxforge_spanish
5. OpenAI Whisper tiny model: https://huggingface.co/openai/whisper-tiny
6. IBM AIHWKit analog hardware-aware toolkit: https://github.com/IBM/aihwkit
7. Graves, A. et al. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.
8. He, K. et al. Deep Residual Learning for Image Recognition.
