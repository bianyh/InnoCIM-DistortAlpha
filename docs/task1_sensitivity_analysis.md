# 任务一：非线性误差敏感性分析

## 1. 目标与实验问题

本任务研究存算一体芯片中输入相关非线性失真对神经网络推理精度的影响。实验将赛题给定的三次非线性映射嵌入到矩阵计算算子的输入激活中，重点回答三个问题：

1. 不同非线性强度 `alpha` 对整网推理精度的影响趋势是什么？
2. 单个层独立受到非线性扰动时，哪些层对最终精度最敏感？
3. 当全网算子同时受到非线性扰动时，误差如何在网络中逐层累积？

非线性函数为：

```text
u = x / max(|x|)
f(u) = alpha * u^3 + (1 - alpha) * u
x' = max(|x|) * f(u)
```

等价地：

```text
x' = (1 - alpha) * x + alpha * x^3 / max(|x|)^2
e(x) = x' - x = alpha * (x^3 / max(|x|)^2 - x)
```

因此该误差不是随机噪声，而是与输入幅值相关的系统性失真。实验实现中使用 `max(|x|).detach().clamp_min(1e-12)`，避免除零并避免推理分析阶段引入不必要的 `max` 梯度。

## 2. 实验对象

### 2.1 CIFAR-10 图像分类

数据集使用本地 `data/cifar-10-batches-py`，测试集共 10,000 张图像。模型直接采用 `chenyaofo/pytorch-cifar-models` 的 CIFAR-10 预训练权重：

| 模型 | 类型 | 参数量 | 非线性注入算子数 | 测试样本数 |
|---|---:|---:|---:|---:|
| `cifar10_resnet20` | CNN/ResNet | 272,474 | 22 | 10,000 |
| `cifar10_mobilenetv2_x1_0` | CNN/MobileNetV2 | 2,236,682 | 54 | 10,000 |

CIFAR-10 实验分为：

- 整网 alpha 扫描：对所有 `Conv2d`、`Linear` 输入同时注入非线性。
- 单层敏感性：每次只对 ResNet20 的一个目标层注入非线性，其余层保持理想。
- 逐层误差累积：全网注入非线性，记录各层输出与 clean 输出的差异。
- 激活分布偏移：选择浅层、中层、末层绘制 clean/nonlinear 激活分布。

### 2.2 VoxForge 语音识别

VoxForge 使用 Hugging Face 上的 Spanish VoxForge 镜像 `ciempiess/voxforge_spanish`，采用 streaming 方式抽取 32 条样本进行敏感性扫描。模型采用 `openai/whisper-tiny`，属于 encoder-decoder Transformer ASR 模型。

| 数据集 | 模型 | 类型 | 参数量 | 注入算子数 | 样本数 | 指标 |
|---|---|---|---:|---:|---:|---|
| `ciempiess/voxforge_spanish` | `openai/whisper-tiny` | Transformer ASR | 37,760,640 | 67 | 32 | WER/CER |

说明：最初尝试使用 VoxForge/Wav2Vec2 大模型分支，但大型模型下载和加载耗时过长。因此本阶段采用 Whisper-tiny 完成 VoxForge 任务一的可复现实验闭环。该结果用于敏感性趋势分析；正式提交时可直接用同一脚本扩大样本数或替换为 VoxForge 专门微调的 Wav2Vec2/Conformer 权重。

## 3. 实现方法

### 3.1 算子级非线性注入

实现文件：

- `src/nonlinear.py`
- `scripts/run_cifar_task1.py`
- `scripts/run_voxforge_whisper_task1.py`

注入方式采用 PyTorch forward pre-hook，不修改模型结构和预训练权重。对目标层输入 `x` 执行：

```python
x_nonlinear = nonlinearity(x, alpha)
y = module(x_nonlinear)
```

覆盖的算子类型包括：

```text
Conv1d
Conv2d
ConvTranspose1d
ConvTranspose2d
Linear
```

其中 CIFAR-10 主要使用 `Conv2d/Linear`，Whisper 使用 `Conv1d/Linear`。

### 3.2 指标设计

CIFAR-10 使用：

```text
Top-1 Accuracy
Cross Entropy Loss
Accuracy Drop = Acc(alpha=0) - Acc(alpha)
Worst Accuracy
Discrete Robust AUC = mean_alpha Acc(alpha)
```

VoxForge 使用：

```text
WER = Word Error Rate
CER = Character Error Rate
```

层级分析使用：

```text
relative_l2 = ||h'_l - h_l||_2 / ||h_l||_2
cosine_drift = 1 - cos(h'_l, h_l)
JS divergence between activation histograms
mean shift
std ratio
zero ratio shift
```

其中 `h_l` 是 clean 模型第 `l` 层输出，`h'_l` 是非线性注入后对应层输出。

## 4. 图表输出

所有图表保存在：

```text
outputs/task1/figures
```

核心图表如下：

| 图表 | 文件 |
|---|---|
| 非线性映射曲线 | `outputs/task1/figures/nonlinearity_curves.png` |
| CIFAR-10 accuracy-alpha 曲线 | `outputs/task1/figures/cifar_accuracy_alpha.png` |
| CIFAR-10 accuracy drop 曲线 | `outputs/task1/figures/cifar_accuracy_drop.png` |
| CIFAR-10 clean/mean/worst accuracy 汇总 | `outputs/task1/figures/cifar_accuracy_summary.png` |
| ResNet20 单层敏感性热力图 | `outputs/task1/figures/cifar10_resnet20_layer_sensitivity_heatmap.png` |
| ResNet20 逐层 relative L2 漂移 | `outputs/task1/figures/cifar10_resnet20_activation_relative_l2.png` |
| ResNet20 逐层 cosine drift | `outputs/task1/figures/cifar10_resnet20_activation_cosine_drift.png` |
| VoxForge Whisper WER-alpha 曲线 | `outputs/task1/figures/voxforge_whisper_wer_alpha.png` |
| VoxForge Whisper relative L2 漂移 | `outputs/task1/figures/voxforge_whisper_activation_relative_l2.png` |
| VoxForge Whisper cosine drift | `outputs/task1/figures/voxforge_whisper_activation_cosine_drift.png` |

## 5. 整网推理精度衰减结果

### 5.1 CIFAR-10

全测试集 10,000 张图像上的 Top-1 accuracy 如下：

| alpha | ResNet20 | MobileNetV2-x1.0 |
|---:|---:|---:|
| -1.0 | 0.1138 | 0.0965 |
| -0.8 | 0.1149 | 0.1006 |
| -0.6 | 0.1082 | 0.1001 |
| -0.4 | 0.1186 | 0.0983 |
| -0.2 | 0.5041 | 0.1144 |
| -0.1 | 0.8332 | 0.1836 |
| 0.0 | 0.9259 | 0.9405 |
| 0.1 | 0.7997 | 0.2191 |
| 0.2 | 0.2031 | 0.1000 |
| 0.4 | 0.1000 | 0.1000 |
| 0.6 | 0.1000 | 0.1000 |
| 0.8 | 0.1007 | 0.1000 |
| 1.0 | 0.1000 | 0.1000 |

汇总指标：

| 模型 | clean accuracy | mean accuracy across alpha | worst accuracy | max accuracy drop |
|---|---:|---:|---:|---:|
| ResNet20 | 0.9259 | 0.3171 | 0.1000 | 0.8259 |
| MobileNetV2-x1.0 | 0.9405 | 0.1810 | 0.0965 | 0.8440 |

主要观察：

1. 当 `alpha=0` 时，两种模型都保持正常预训练精度，说明数据加载、归一化和权重加载正确。
2. 两种模型在较强非线性下都退化到接近随机猜测水平，即 CIFAR-10 十分类的约 10% accuracy。
3. ResNet20 对弱扰动更稳健：`alpha=-0.1` 时仍有 83.32%，`alpha=0.1` 时为 79.97%。
4. MobileNetV2 对弱扰动高度敏感：`alpha=-0.1` 降至 18.36%，`alpha=0.1` 降至 21.91%。
5. 这说明参数量更大不必然代表更强非线性鲁棒性；MobileNetV2 的 depthwise/pointwise 结构和更深的算子链条可能导致输入失真快速累积。

### 5.2 VoxForge Whisper

VoxForge Spanish 32 条样本上的 ASR 结果如下：

| alpha | WER | CER |
|---:|---:|---:|
| -1.0 | 1.0000 | 1.0000 |
| -0.5 | 1.0000 | 1.0000 |
| -0.2 | 1.0017 | 0.9820 |
| -0.1 | 1.1304 | 0.6764 |
| 0.0 | 0.3293 | 0.1102 |
| 0.1 | 1.1492 | 1.0188 |
| 0.2 | 1.0000 | 1.0000 |
| 0.5 | 1.0000 | 1.0000 |
| 1.0 | 1.0000 | 1.0000 |

主要观察：

1. Clean Whisper-tiny 在该 32 条 VoxForge Spanish 样本上的 WER 为 0.3293，CER 为 0.1102。
2. 当 `alpha=±0.1` 时，WER 已超过 1.0，说明生成式 ASR 对算子输入失真非常敏感。
3. 当 `|alpha| >= 0.2` 时，WER/CER 基本进入饱和失效状态。
4. ASR 的误差表现比 CIFAR 更剧烈，原因可能包括：encoder-decoder 结构误差会影响后续自回归解码；一旦早期 token 或声学表示偏移，后续文本生成会进一步放大误差。

## 6. 单层输出分布偏移与单层敏感性

### 6.1 ResNet20 单层敏感性

单层敏感性在 ResNet20 上进行，每次只对一个层输入注入非线性，样本数为 2,048。`alpha=1.0` 下最敏感层如下：

| 排名 | 层 | alpha | 单层注入后 accuracy | accuracy drop |
|---:|---|---:|---:|---:|
| 1 | `layer3.0.conv1` | 1.0 | 0.1089 | 0.8145 |
| 2 | `layer1.0.conv1` | 1.0 | 0.1113 | 0.8120 |
| 3 | `layer2.0.conv1` | 1.0 | 0.1294 | 0.7939 |
| 4 | `layer2.0.conv2` | 1.0 | 0.1436 | 0.7798 |
| 5 | `layer3.1.conv1` | 1.0 | 0.2207 | 0.7026 |
| 6 | `layer3.2.conv2` | 1.0 | 0.2583 | 0.6650 |

按 alpha 分组的平均/最大单层精度下降：

| alpha | mean accuracy drop | max accuracy drop |
|---:|---:|---:|
| -1.0 | 0.0939 | 0.3286 |
| -0.5 | 0.0254 | 0.0879 |
| 0.5 | 0.0498 | 0.1855 |
| 1.0 | 0.4427 | 0.8145 |

结论：

1. 单层注入下，`alpha=1.0` 的正向压缩型非线性对 ResNet20 的单层破坏最强。
2. 各 stage 的第一个卷积层通常更敏感，例如 `layer1.0.conv1`、`layer2.0.conv1`、`layer3.0.conv1`。这些层处于特征尺度变换或 stage 开始位置，输入分布变化会影响后续整个 stage。
3. 单层敏感性与整网曲线不完全相同：整网中 `alpha=-1.0` 会产生巨大累积幅值漂移；单层中 `alpha=1.0` 对局部特征方向破坏更直接。

### 6.2 激活分布偏移

激活分布图已输出到：

```text
outputs/task1/figures/cifar10_resnet20_activation_hist_*.png
```

代表层包括：

```text
conv1
layer2.1.conv2
fc
```

从分布图和统计指标看：

1. 浅层 `conv1` 的 clean 与 nonlinear 分布已经出现明显形状变化，但仍保留部分结构。
2. 中层卷积的分布偏移明显加大，说明前面层的误差已经与当前层输入非线性叠加。
3. 最终 `fc` 附近 logits/分类特征方向显著偏移，导致预测类别接近随机。

## 7. 误差逐层累积行为

ResNet20 全网注入非线性后的逐层漂移统计如下：

| alpha | mean relative L2 | max relative L2 | mean cosine drift | max cosine drift | mean JS divergence |
|---:|---:|---:|---:|---:|---:|
| -1.0 | 41.4098 | 315.4180 | 0.5000 | 0.9858 | 0.3913 |
| -0.5 | 3.6615 | 13.3577 | 0.3490 | 0.9952 | 0.1893 |
| 0.5 | 0.8014 | 1.1193 | 0.3971 | 1.0029 | 0.1250 |
| 1.0 | 1.0042 | 1.7283 | 0.6552 | 1.0056 | 0.2518 |

相对 L2 最大的层主要出现在后部 stage：

| 层 | alpha | relative L2 | cosine drift | JS divergence |
|---|---:|---:|---:|---:|
| `layer3.2.conv1` | -1.0 | 315.4180 | 0.2655 | 0.6727 |
| `layer3.1.conv2` | -1.0 | 174.6765 | 0.4410 | 0.6013 |
| `layer3.0.conv2` | -1.0 | 86.6685 | 0.6674 | 0.6062 |
| `layer2.2.conv2` | -1.0 | 59.3146 | 0.8920 | 0.5776 |
| `layer3.1.conv1` | -1.0 | 46.1393 | 0.6608 | 0.5768 |

结论：

1. `alpha=-1.0` 会造成极大的幅值漂移，relative L2 在深层可达数百倍，说明放大型非线性会在网络后部显著累积。
2. `alpha=1.0` 的 relative L2 没有负 alpha 那么极端，但 cosine drift 均值达到 0.6552，说明特征方向被严重改变，同样会破坏分类边界。
3. 误差累积并非简单线性叠加，而是经过卷积、BatchNorm、ReLU、残差分支后在特定 stage 突增。
4. 后部 stage 对最终决策更直接，分布偏移和方向漂移会迅速传导到分类头。

## 8. 复现实验命令

安装依赖：

```powershell
python -m pip install -r requirements-task1.txt
```

CIFAR-10 全量 alpha 扫描、ResNet20 单层敏感性和逐层漂移：

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

仅从已有 CSV 重新生成 CIFAR 图表：

```powershell
python scripts\make_task1_figures.py --output-dir outputs\task1 --analysis-model cifar10_resnet20
```

VoxForge Whisper alpha 扫描：

```powershell
python scripts\run_voxforge_whisper_task1.py `
  --streaming `
  --max-samples 32 `
  --max-new-tokens 48 `
  --alphas=-1.0,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.0 `
  --skip-activation-analysis `
  --output-dir outputs\task1
```

VoxForge Whisper 层级漂移分析：

```powershell
python scripts\run_voxforge_whisper_task1.py `
  --streaming `
  --max-samples 8 `
  --max-new-tokens 48 `
  --alphas=-1.0,-0.5,0,0.5,1.0 `
  --drift-alphas=-1.0,0.5,1.0 `
  --analysis-max-layers 32 `
  --output-dir outputs\task1
```

## 9. 阶段性结论

1. 算子输入非线性会导致显著推理精度退化。CIFAR-10 中 ResNet20 从 92.59% 降到最差 10.00%，MobileNetV2 从 94.05% 降到最差 9.65%。
2. 非线性强度与精度退化呈明显非线性关系。弱扰动区间已经会造成精度大幅下降，强扰动区间直接进入随机猜测或 ASR 失效状态。
3. 不同网络结构的敏感性差异明显。MobileNetV2 在 `|alpha|=0.1` 附近已严重退化，ResNet20 相对更稳健。
4. 单层敏感性集中在 stage 开始位置和后部决策相关层，说明结构边界和特征尺度转换位置是非线性误差的高风险区域。
5. 误差在网络中存在逐层累积和阶段性突增。负 alpha 主要造成幅值漂移，正 alpha 更明显破坏特征方向，两类误差都能导致最终决策失效。
6. 语音识别任务比图像分类更加脆弱。Whisper-tiny 在 VoxForge 上 `alpha=±0.1` 即出现 WER 大幅恶化，说明 encoder-decoder ASR 的自回归生成过程会放大前端表示偏移。

## 10. 后续任务衔接

任务一的结果为任务二和任务三提供直接依据：

1. 非线性感知训练应覆盖一组 alpha，而不是只针对单点 alpha。
2. 训练时应重点关注 stage 起始层和后部敏感层。
3. 可以设计 sensitivity-guided NAT：对高敏感层施加更强训练扰动或更强特征一致性约束。
4. 可以引入 BatchNorm recalibration、激活范围校准、知识蒸馏等方法缓解逐层分布偏移。
5. 语音模型需要单独考虑解码误差放大，不能仅用分类任务结论替代。

## 11. 参考资料

1. CIFAR-10 official dataset: https://www.cs.toronto.edu/~kriz/cifar.html
2. PyTorch CIFAR models: https://github.com/chenyaofo/pytorch-cifar-models
3. VoxForge: http://www.voxforge.org/
4. Hugging Face VoxForge Spanish dataset: https://huggingface.co/datasets/ciempiess/voxforge_spanish
5. OpenAI Whisper tiny model: https://huggingface.co/openai/whisper-tiny
6. IBM AIHWKit analog hardware-aware toolkit: https://github.com/IBM/aihwkit
7. Rasch et al., hardware-aware training for analog in-memory computing, Nature Communications 2023.
