# 任务一技术报告：非线性误差敏感性分析

> **重要修正：固定 alpha 曲线不是官方主协议。**
> 最新确认后，任务一主协议应为每个目标矩阵算子输入端、每次 forward 调用独立采样
> `alpha_{l,c} ~ Uniform(-1,1)`。因此，固定 `alpha` vs accuracy 曲线只保留为解释
> 方向性和层敏感性的辅助诊断；主结果应使用随机 alpha 场的 mean / worst / std、
> 单层随机注入和逐层 drift 统计。统一定义见
> [per_occurrence_random_alpha_protocol.md](per_occurrence_random_alpha_protocol.md)。

## 1. 任务目标

本任务面向存算一体芯片部署场景，研究模拟域乘累加运算中输入相关非线性失真对神经网络推理精度的影响。实验将赛题给定的非线性映射模型嵌入到可进行矩阵计算的算子输入端，包括 `Conv1d`、`Conv2d`、`Linear` 等层，模拟激活值进入存算阵列前受到器件非理想响应影响的情况。分析重点包括三部分：第一，系统评估不同非线性强度 `alpha` 对整网推理精度的影响；第二，定位单层非线性误差对最终任务指标的敏感程度；第三，通过中间层输出统计分析单层输出分布偏移和误差在网络中的逐层累积行为。

## 2. 非线性误差模型

赛题给出的非线性误差模型可表示为：

```text
u = x / max(|x|)
f(u) = alpha * u^3 + (1 - alpha) * u
x' = max(|x|) * f(u)
```

其中，`x` 表示理想输入激活值，`x'` 表示经过非线性失真后的实际输入值，`alpha` 表示非线性强度。将其展开可得：

```text
x' = (1 - alpha) * x + alpha * x^3 / max(|x|)^2
e(x) = x' - x = alpha * (x^3 / max(|x|)^2 - x)
```

由上式可见，该误差并不是独立同分布的随机噪声，而是与输入幅值强相关的系统性非线性失真。当 `alpha=0` 时，映射退化为理想线性关系；当 `alpha` 偏离 0 时，激活值的幅值和方向都会被改变，并可能在深层网络中逐层累积。

![图1 不同 alpha 下的三次非线性映射曲线](../outputs/task1/figures/nonlinearity_curves.png)

图1展示了归一化输入 `u` 在不同 `alpha` 下的映射关系。可以看到，`alpha` 越远离 0，曲线偏离理想直线 `f(u)=u` 越明显。正负 `alpha` 对中间幅值激活的影响方向不同，因此后续实验分别测试正向和负向非线性。

## 3. 实验设计

### 3.1 数据集与模型

本任务选择图像分类和语音识别两个任务进行敏感性分析。图像分类使用 CIFAR-10，语音识别使用 VoxForge Spanish 镜像数据集。模型方面，CIFAR-10 采用已有预训练 CNN 权重，VoxForge 采用轻量 Transformer ASR 模型 Whisper-tiny。

**表1 CIFAR-10 实验模型配置**

| 模型 | 类型 | 参数量 | 非线性注入算子数 | 测试样本数 | 评价指标 |
|---|---:|---:|---:|---:|---|
| `cifar10_resnet20` | CNN/ResNet | 272,474 | 22 | 10,000 | Top-1 Accuracy / Loss |
| `cifar10_mobilenetv2_x1_0` | CNN/MobileNetV2 | 2,236,682 | 53 | 10,000 | Top-1 Accuracy / Loss |

**表2 VoxForge 实验模型配置**

| 数据集 | 模型 | 类型 | 参数量 | 非线性注入算子数 | 样本数 | 评价指标 |
|---|---|---|---:|---:|---:|---|
| `ciempiess/voxforge_spanish` | `openai/whisper-tiny` | Encoder-Decoder Transformer ASR | 37,760,640 | 67 | 32 | WER / CER |

说明：VoxForge 部分最初也预留了 Wav2Vec2/CTC 模型脚本，但大型模型在当前环境中下载和加载耗时较长。因此本阶段采用 Whisper-tiny 完成 VoxForge 任务一的实验闭环。Whisper-tiny 虽不是 VoxForge 专门微调模型，但能够代表 Transformer 语音识别模型在 VoxForge 数据上的非线性推理敏感性。

### 3.2 非线性注入位置

实验采用 PyTorch forward pre-hook 在目标算子的输入端注入非线性，不修改模型结构和预训练权重。对于任意目标层 `module`，原始前向传播为：

```text
y = module(x)
```

注入非线性后变为：

```text
x_nonlinear = nonlinearity(x, alpha)
y = module(x_nonlinear)
```

目标算子包括：

```text
Conv1d
Conv2d
ConvTranspose1d
ConvTranspose2d
Linear
```

CIFAR-10 模型主要涉及 `Conv2d` 和 `Linear`；Whisper 模型主要涉及 `Conv1d` 和 `Linear`。这种实现方式的优势是可以对不同模型统一注入硬件误差模型，同时保持原有预训练权重不变。

### 3.3 评价指标

CIFAR-10 图像分类使用：

```text
Top-1 Accuracy
Cross Entropy Loss
Accuracy Drop = Acc(alpha=0) - Acc(alpha)
Worst Accuracy = min_alpha Acc(alpha)
Discrete Robust AUC = mean_alpha Acc(alpha)
```

VoxForge 语音识别使用：

```text
WER = Word Error Rate
CER = Character Error Rate
```

需要注意，WER 允许大于 1，因为当插入错误较多时，编辑距离可以超过参考文本词数。

层级误差传播分析使用：

```text
relative_l2 = ||h'_l - h_l||_2 / ||h_l||_2
cosine_drift = 1 - cos(h'_l, h_l)
JS divergence = clean 激活分布与 nonlinear 激活分布之间的 Jensen-Shannon divergence
mean shift = mean(h'_l) - mean(h_l)
std ratio = std(h'_l) / std(h_l)
```

其中 `h_l` 表示 clean 模型第 `l` 层输出，`h'_l` 表示非线性注入后第 `l` 层输出。

## 4. 整网推理精度衰减分析

### 4.1 CIFAR-10 整网精度趋势

在 CIFAR-10 中，对所有目标算子输入同时注入非线性，扫描 `alpha ∈ {-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0}`。结果如图2、图3和表3所示。

![图2 CIFAR-10 上不同 alpha 对整网 Top-1 Accuracy 的影响](../outputs/task1/figures/cifar_accuracy_alpha.png)

图2展示了 ResNet20 和 MobileNetV2-x1.0 在不同 `alpha` 下的精度变化。两种模型在 `alpha=0` 时均保持正常预训练精度；当 `alpha` 偏离 0 后，精度迅速下降。特别是 MobileNetV2，即使在 `|alpha|=0.1` 的弱非线性条件下也出现严重退化。

![图3 CIFAR-10 上不同 alpha 对整网 Accuracy Drop 的影响](../outputs/task1/figures/cifar_accuracy_drop.png)

图3将精度变化转换为相对 clean 模型的 accuracy drop。可以看到，非线性误差导致的退化并非线性增长，而是存在明显阈值效应：在弱非线性区间，ResNet20 尚能保持部分性能；一旦超过该区间，模型迅速接近随机猜测水平。

**表3 CIFAR-10 整网 alpha 扫描结果**

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

从表3可以看出，ResNet20 的 clean accuracy 为 0.9259，MobileNetV2 的 clean accuracy 为 0.9405。在强非线性条件下，两者均下降到约 0.10，接近 CIFAR-10 十分类随机猜测水平。MobileNetV2 的弱扰动鲁棒性明显更差：`alpha=-0.1` 时 accuracy 仅为 0.1836，`alpha=0.1` 时为 0.2191；而 ResNet20 在相同条件下仍分别保持 0.8332 和 0.7997。

![图4 CIFAR-10 模型 clean、平均鲁棒和最坏精度汇总](../outputs/task1/figures/cifar_accuracy_summary.png)

**表4 CIFAR-10 鲁棒性汇总**

| 模型 | Clean Accuracy | Mean Accuracy | Worst Accuracy | Max Accuracy Drop | Discrete Robust AUC | 参数量 | 注入算子数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cifar10_mobilenetv2_x1_0` | 0.9405 | 0.1810 | 0.0965 | 0.8440 | 0.1810 | 2,236,682 | 53 |
| `cifar10_resnet20` | 0.9259 | 0.3171 | 0.1000 | 0.8259 | 0.3171 | 272,474 | 22 |

表4说明，参数量更大的 MobileNetV2 并没有表现出更强的非线性容忍能力。相反，它的平均鲁棒精度明显低于 ResNet20。这表明网络结构、算子类型、层间归一化和残差连接等因素比单纯参数量更能影响非线性误差鲁棒性。

### 4.2 VoxForge 语音识别整网趋势

VoxForge 部分在 Spanish VoxForge 数据上抽取 32 条样本，使用 Whisper-tiny 进行推理。实验扫描 `alpha ∈ {-1.0, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1.0}`。结果如图5和表5所示。

![图5 VoxForge Whisper-tiny 上不同 alpha 对 WER 的影响](../outputs/task1/figures/voxforge_whisper_wer_alpha.png)

**表5 VoxForge Whisper-tiny alpha 扫描结果**

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

从表5可以看出，Whisper-tiny 在 clean 条件下的 WER 为 0.3293，CER 为 0.1102。当 `alpha=±0.1` 时，WER 已超过 1.0；当 `|alpha| >= 0.2` 时，模型基本进入失效状态。语音识别任务对非线性误差更加敏感，原因可能在于 encoder-decoder ASR 的误差传播链条更长：前端声学表示受到扰动后，不仅影响 encoder 表示，还会通过 decoder 的自回归生成过程进一步放大。

## 5. 单层敏感性分析

整网精度曲线只能说明模型整体对非线性误差敏感，但不能回答“哪些层最脆弱”。因此在 ResNet20 上进一步进行单层敏感性分析：每次只对一个目标层的输入注入非线性，其余层保持理想线性状态，记录最终分类精度下降。实验样本数为 2,048。

![图6 ResNet20 单层非线性敏感性热力图](../outputs/task1/figures/cifar10_resnet20_layer_sensitivity_heatmap.png)

图6中，横轴为 `alpha`，纵轴为 ResNet20 中的目标层，颜色表示单层注入非线性造成的 accuracy drop。颜色越深，表示该层越敏感。

**表6 ResNet20 单层非线性注入造成的 Accuracy Drop**

| 层 | 层编号 | alpha=-1.0 | alpha=-0.5 | alpha=0.5 | alpha=1.0 |
|---|---:|---:|---:|---:|---:|
| `conv1` | 0 | 0.0898 | 0.0200 | 0.0205 | 0.3008 |
| `layer1.0.conv1` | 1 | 0.0762 | 0.0063 | 0.1133 | 0.8120 |
| `layer1.0.conv2` | 2 | 0.0645 | 0.0195 | 0.0371 | 0.6011 |
| `layer1.1.conv1` | 3 | 0.0137 | 0.0034 | 0.0332 | 0.3877 |
| `layer1.1.conv2` | 4 | 0.0239 | 0.0093 | 0.0142 | 0.0996 |
| `layer1.2.conv1` | 5 | 0.0552 | 0.0142 | 0.0527 | 0.6626 |
| `layer1.2.conv2` | 6 | 0.0498 | 0.0117 | 0.0347 | 0.3530 |
| `layer2.0.conv1` | 7 | 0.2515 | 0.0776 | 0.1597 | 0.7939 |
| `layer2.0.conv2` | 8 | 0.3286 | 0.0879 | 0.1855 | 0.7798 |
| `layer2.0.downsample.0` | 9 | 0.1479 | 0.0303 | 0.0410 | 0.2603 |
| `layer2.1.conv1` | 10 | 0.1807 | 0.0454 | 0.0298 | 0.3076 |
| `layer2.1.conv2` | 11 | 0.0518 | 0.0146 | 0.0166 | 0.1260 |
| `layer2.2.conv1` | 12 | 0.2261 | 0.0327 | 0.0269 | 0.4546 |
| `layer2.2.conv2` | 13 | 0.0474 | 0.0156 | 0.0127 | 0.0903 |
| `layer3.0.conv1` | 14 | 0.1782 | 0.0615 | 0.1396 | 0.8145 |
| `layer3.0.conv2` | 15 | 0.0820 | 0.0371 | 0.0391 | 0.6060 |
| `layer3.0.downsample.0` | 16 | 0.0176 | 0.0107 | 0.0039 | 0.0107 |
| `layer3.1.conv1` | 17 | 0.1255 | 0.0405 | 0.0386 | 0.7026 |
| `layer3.1.conv2` | 18 | 0.0420 | 0.0156 | 0.0352 | 0.4214 |
| `layer3.2.conv1` | 19 | 0.0103 | 0.0044 | 0.0410 | 0.4902 |
| `layer3.2.conv2` | 20 | 0.0034 | 0.0005 | 0.0190 | 0.6650 |
| `fc` | 21 | 0.0005 | 0.0005 | 0.0005 | -0.0005 |

**表7 ResNet20 单层敏感性按 alpha 汇总**

| alpha | Mean Accuracy Drop | Max Accuracy Drop | 最高敏感层 |
|---:|---:|---:|---|
| -1.0 | 0.0939 | 0.3286 | `layer2.0.conv2` |
| -0.5 | 0.0254 | 0.0879 | `layer2.0.conv2` |
| 0.5 | 0.0498 | 0.1855 | `layer2.0.conv2` |
| 1.0 | 0.4427 | 0.8145 | `layer3.0.conv1` |

**表8 alpha=1.0 下最敏感的 ResNet20 层**

| 排名 | 层 | 单层注入后 Accuracy | Accuracy Drop |
|---:|---|---:|---:|
| 1 | `layer3.0.conv1` | 0.1089 | 0.8145 |
| 2 | `layer1.0.conv1` | 0.1113 | 0.8120 |
| 3 | `layer2.0.conv1` | 0.1294 | 0.7939 |
| 4 | `layer2.0.conv2` | 0.1436 | 0.7798 |
| 5 | `layer3.1.conv1` | 0.2207 | 0.7026 |
| 6 | `layer3.2.conv2` | 0.2583 | 0.6650 |

单层敏感性实验表明，不同层对非线性误差的容忍能力差异显著。各 stage 的开始位置通常更敏感，例如 `layer1.0.conv1`、`layer2.0.conv1`、`layer3.0.conv1`。这些层处于特征尺度或语义阶段变化的边界，输入分布被扰动后会影响后续整个 stage。相比之下，最终 `fc` 层在单层注入下几乎不敏感，说明分类头单独受到该映射影响时不会像中间卷积层那样引起大规模特征重构错误。

## 6. 单层输出分布偏移

为了观察非线性误差对中间激活分布的影响，实验记录 clean 模型和 nonlinear 模型在代表层上的输出分布。选取浅层 `conv1`、中层 `layer2.1.conv2` 和末层 `fc` 进行展示。

![图7 ResNet20 conv1 层在 alpha=1.0 下的激活分布偏移](../outputs/task1/figures/cifar10_resnet20_activation_hist_conv1_alpha_1.png)

图7显示，浅层卷积输出在非线性注入后已经出现分布压缩和形状偏移，但仍保留部分 clean 分布轮廓。

![图8 ResNet20 layer2.1.conv2 层在 alpha=1.0 下的激活分布偏移](../outputs/task1/figures/cifar10_resnet20_activation_hist_layer2_1_conv2_alpha_1.png)

图8显示，中层特征分布偏移更加明显。这说明前面层的非线性误差已经传播到中层，并与当前层输入非线性叠加。

![图9 ResNet20 fc 层在 alpha=1.0 下的激活分布偏移](../outputs/task1/figures/cifar10_resnet20_activation_hist_fc_alpha_1.png)

图9显示，接近分类输出的特征分布已经明显偏离 clean 状态。由于分类决策依赖 logits 之间的相对大小和方向，末层特征分布偏移会直接导致类别预测错误。

## 7. 误差逐层累积行为

在全网非线性注入条件下，记录每个目标层的输出，并与 clean 模型对应层输出比较。图10和图11分别展示 relative L2 和 cosine drift 的逐层变化。

![图10 ResNet20 全网非线性注入后的逐层 relative L2 漂移](../outputs/task1/figures/cifar10_resnet20_activation_relative_l2.png)

![图11 ResNet20 全网非线性注入后的逐层 cosine drift](../outputs/task1/figures/cifar10_resnet20_activation_cosine_drift.png)

**表9 ResNet20 全网注入后的逐层误差统计汇总**

| alpha | Mean Relative L2 | Max Relative L2 | Mean Cosine Drift | Max Cosine Drift | Mean JS Divergence | Max JS Divergence | Mean Mean Shift | Max Mean Shift | Mean Std Ratio | Max Std Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1.0000 | 41.4098 | 315.4180 | 0.5000 | 0.9858 | 0.3913 | 0.6727 | -6.4390 | 0.0022 | 35.6950 | 222.7964 |
| -0.5000 | 3.6615 | 13.3577 | 0.3490 | 0.9952 | 0.1893 | 0.5725 | -0.4343 | 0.0011 | 3.9545 | 11.1321 |
| 0.5000 | 0.8014 | 1.1193 | 0.3971 | 1.0029 | 0.1250 | 0.3319 | 0.0469 | 0.1570 | 0.4445 | 0.7409 |
| 1.0000 | 1.0042 | 1.7283 | 0.6552 | 1.0056 | 0.2518 | 0.5185 | 0.0720 | 0.2836 | 0.3632 | 1.3626 |

**表10 ResNet20 relative L2 最大的层**

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
| `layer2.2.conv1` | -1.0000 | 14.8709 | 0.2925 | 0.4836 |
| `layer2.0.conv2` | -1.0000 | 13.8335 | 0.5808 | 0.4027 |

表9和表10揭示了两个重要现象。第一，`alpha=-1.0` 会导致极大的幅值漂移，平均 relative L2 达到 41.4098，最大值达到 315.4180，说明放大型非线性会在深层网络中严重累积。第二，`alpha=1.0` 的 relative L2 虽然没有负 alpha 那么极端，但平均 cosine drift 达到 0.6552，说明特征方向被明显破坏。也就是说，负 alpha 主要体现为幅值尺度失控，正 alpha 则更明显地改变特征方向；二者都会导致最终分类边界失效。

## 8. VoxForge Transformer 层级误差传播

除最终 WER/CER 外，实验还对 Whisper-tiny 的前 32 个目标层进行了层级漂移分析。该分析使用 1 条语音样本进行 clean/nonlinear 对齐比较，目的是观察 Transformer ASR 模型中哪些算子输出更容易受到输入非线性的影响。

![图12 Whisper-tiny 全网非线性注入后的逐层 relative L2 漂移](../outputs/task1/figures/voxforge_whisper_activation_relative_l2.png)

![图13 Whisper-tiny 全网非线性注入后的逐层 cosine drift](../outputs/task1/figures/voxforge_whisper_activation_cosine_drift.png)

**表11 Whisper-tiny 层级漂移统计汇总**

| alpha | Mean Relative L2 | Max Relative L2 | Mean Cosine Drift | Max Cosine Drift | Mean JS Divergence | Max JS Divergence |
|---:|---:|---:|---:|---:|---:|---:|
| -1.0000 | 2.2088 | 6.2660 | 0.5326 | 1.2340 | 0.1181 | 0.3279 |
| 0.5000 | 0.7270 | 1.2677 | 0.3686 | 0.9438 | 0.0476 | 0.1707 |
| 1.0000 | 0.8141 | 1.5782 | 0.4429 | 0.9959 | 0.1535 | 0.3741 |

**表12 Whisper-tiny relative L2 最大的层**

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

## 9. 工程实现与复现方式

本任务相关代码和输出如下：

**表13 任务一工程文件**

| 类型 | 文件路径 | 作用 |
|---|---|---|
| 非线性注入工具 | `src/nonlinear.py` | 定义非线性函数、forward pre-hook 注入器和激活记录器 |
| 指标工具 | `src/metrics.py` | 计算 accuracy、WER、CER、relative L2、cosine drift 等指标 |
| 绘图工具 | `src/plotting.py` | 生成曲线图、热力图和激活分布图 |
| CIFAR 实验脚本 | `scripts/run_cifar_task1.py` | 运行 CIFAR-10 alpha 扫描、单层敏感性和逐层漂移分析 |
| VoxForge Wav2Vec2/CTC 脚本 | `scripts/run_voxforge_task1.py` | 预留 Hugging Face CTC ASR 模型实验入口 |
| VoxForge Whisper 脚本 | `scripts/run_voxforge_whisper_task1.py` | 运行 VoxForge + Whisper-tiny alpha 扫描和层级漂移分析 |
| 图表重绘脚本 | `scripts/make_task1_figures.py` | 根据 CSV 结果重新生成图表 |
| 任务一报告 | `docs/task1_technical_report.md` | 当前技术报告 |

**表14 任务一主要输出文件**

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

复现实验环境可通过如下命令安装：

```powershell
python -m pip install -r requirements-task1.txt
```

CIFAR-10 主实验命令：

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

VoxForge Whisper 主实验命令：

```powershell
python scripts\run_voxforge_whisper_task1.py `
  --streaming `
  --max-samples 32 `
  --max-new-tokens 48 `
  --alphas=-1.0,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.0 `
  --skip-activation-analysis `
  --output-dir outputs\task1
```

## 10. 结论

通过 CIFAR-10 和 VoxForge 两类任务的实验，可以得到以下结论。

首先，输入相关非线性误差会显著破坏神经网络推理精度。CIFAR-10 中，ResNet20 从 clean 条件下的 92.59% 下降到最差 10.00%，MobileNetV2 从 94.05% 下降到最差 9.65%，接近随机猜测。VoxForge 语音识别中，Whisper-tiny 的 clean WER 为 0.3293，而 `alpha=±0.1` 时 WER 已超过 1.0，说明 ASR 模型对输入非线性尤其敏感。

其次，非线性强度与任务性能退化之间存在明显阈值效应。模型并不是随 `alpha` 线性缓慢退化，而是在某些弱扰动区间内就出现急剧性能下降。MobileNetV2 在 `alpha=0.1` 时精度已经从 94.05% 降至 21.91%，说明轻量化结构可能更容易受到存算硬件非理想性的影响。

第三，单层敏感性具有明显结构位置差异。ResNet20 中 stage 起始层和部分中后段卷积层对非线性误差最敏感，例如 `layer3.0.conv1`、`layer1.0.conv1`、`layer2.0.conv1`。这说明后续鲁棒训练不应只做整网均匀扰动，而应考虑 layer-wise sensitivity。

第四，误差会在网络中逐层累积，并表现出不同机制。负 `alpha` 更容易导致幅值尺度失控，表现为 relative L2 大幅增长；正 `alpha` 则更明显破坏特征方向，表现为 cosine drift 增大。二者最终都会导致分类边界或解码过程失效。

因此，任务一的敏感性分析为后续非线性感知训练和鲁棒性增强提供了直接依据：后续方法应覆盖多种 `alpha` 强度，关注高敏感层，并引入分布校准、敏感层约束或特征一致性训练来抑制非线性误差的逐层累积。

## 11. 参考资料

1. CIFAR-10 official dataset: https://www.cs.toronto.edu/~kriz/cifar.html
2. PyTorch CIFAR models: https://github.com/chenyaofo/pytorch-cifar-models
3. VoxForge: http://www.voxforge.org/
4. Hugging Face VoxForge Spanish dataset: https://huggingface.co/datasets/ciempiess/voxforge_spanish
5. OpenAI Whisper tiny model: https://huggingface.co/openai/whisper-tiny
6. IBM AIHWKit analog hardware-aware toolkit: https://github.com/IBM/aihwkit
