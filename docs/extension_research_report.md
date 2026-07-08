# 拓展研究报告：结构规模、噪声机制、量化叠加与非线性误差

日期：2026-07-08  
项目：InnoCIM-DistortAlpha

## 0. 实验入口与设置

本报告按如下顺序完成三项拓展研究：

1. 网络结构与参数量对非线性误差影响；
2. 对比分析随机高斯噪声注入与非线性失真对模型鲁棒性的不同影响机制；
3. 结合量化误差与非线性误差对模型推理精度的影响进行分析。

新增实验脚本：

```powershell
python scripts\run_extension_research.py `
  --output-dir outputs\extension_research `
  --test-subset 4096 `
  --batch-size 256 `
  --random-repeats 3 `
  --drift-batches 8
```

主要输出：

| 类型 | 路径 |
|---|---|
| 结构扫描表 | `outputs/extension_research/tables/structure_summary.csv` |
| 结构 alpha 曲线 | `outputs/extension_research/tables/structure_alpha_sweep.csv` |
| 高斯噪声与非线性精度表 | `outputs/extension_research/tables/mechanism_accuracy.csv` |
| logits 漂移表 | `outputs/extension_research/tables/mechanism_logit_drift.csv` |
| 量化与非线性叠加表 | `outputs/extension_research/tables/quantization_nonlinearity.csv` |
| 误差分解表 | `outputs/extension_research/tables/quantization_error_decomposition.csv` |
| 图表目录 | `outputs/extension_research/figures/` |

实验使用 CIFAR-10 测试集前 4096 张样本。结构扫描使用 `chenyaofo/pytorch-cifar-models` 的 CIFAR-10 pretrained checkpoint；第 2、3 项使用项目已有 clean ResNet20 checkpoint：

```text
outputs/task2/checkpoints/cifar/cifar_resnet20_clean_best.pt
```

注入位置沿用项目主协议，覆盖 `Conv1d`、`Conv2d`、`ConvTranspose1d`、`ConvTranspose2d`、`Linear` 的输入端，归一化范围为 `per_tensor`。

## 1. 网络结构与参数量对非线性误差影响

### 1.1 已有全量结果回顾

项目原有任务一已经在完整 CIFAR-10 test set 上完成 ResNet20 与 MobileNetV2 的固定 alpha 扫描：

| 模型 | 参数量 | 目标算子数 | Clean Acc | Mean Acc | Worst Acc | Max Drop |
|---|---:|---:|---:|---:|---:|---:|
| ResNet20 | 0.272M | 22 | 92.59% | 31.71% | 10.00% | 82.59% |
| MobileNetV2 x1.0 | 2.237M | 53 | 94.05% | 18.10% | 9.65% | 84.40% |

这个结果已经说明：参数量更大并不自动带来更强非线性鲁棒性。MobileNetV2 x1.0 的参数量约为 ResNet20 的 8.2 倍，但平均鲁棒精度更低，最差精度同样接近 CIFAR-10 随机猜测。

### 1.2 拓展结构扫描

本次拓展在 4096 张测试子集上增加 ResNet56、MobileNetV2 x0.5、ShuffleNetV2 x1.0、VGG11-BN，并统一扫描：

```text
alpha = {-1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1}
```

![结构 alpha 曲线](../outputs/extension_research/figures/structure_accuracy_alpha.png)

![参数量与精度下降](../outputs/extension_research/figures/structure_drop_vs_params.png)

| 模型 | 结构族 | 参数量 | 目标算子数 | Clean Acc | Mean Acc | Mild Mean Acc | Strong Mean Acc | Worst Acc | Max Drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet20 | ResNet | 0.272M | 22 | 92.60% | 41.08% | 58.86% | 10.41% | 10.03% | 82.57% |
| ResNet56 | ResNet | 0.856M | 58 | 94.46% | 44.28% | 65.20% | 10.82% | 10.57% | 83.89% |
| MobileNetV2 x0.5 | MobileNetV2 | 0.700M | 53 | 92.94% | 20.58% | 13.35% | 9.72% | 9.62% | 83.33% |
| MobileNetV2 x1.0 | MobileNetV2 | 2.237M | 53 | 93.55% | 21.34% | 14.83% | 9.79% | 9.69% | 83.86% |
| ShuffleNetV2 x1.0 | ShuffleNetV2 | 1.264M | 57 | 92.60% | 23.62% | 20.07% | 9.94% | 9.69% | 82.91% |
| VGG11-BN | VGG | 9.756M | 11 | 92.53% | 59.68% | 87.30% | 23.85% | 9.69% | 82.84% |

其中：

- `Mild Mean Acc` 是 `alpha in {-0.2, -0.1, 0.1, 0.2}` 的平均准确率；
- `Strong Mean Acc` 是 `|alpha| >= 0.5` 的平均准确率；
- `Max Drop = Clean Acc - Worst Acc`。

### 1.3 结构与参数量影响结论

第一，所有模型在强非线性下都会出现接近随机猜测的最坏情况。6 个模型的 worst accuracy 都在约 `9.6%-10.6%`，说明当 alpha 足够强时，仅依靠常规模型结构和参数规模并不能避免整网失效。

第二，参数量与平均鲁棒性没有简单正相关。本次 6 模型子集的 Spearman 描述性相关中，`params` 与 `mean_accuracy` 的相关系数约为 `0.31`，与 `max_accuracy_drop` 的相关系数约为 `0.14`。样本数很小，不能作为统计定论，但足以说明参数量不是单独可用的鲁棒性解释变量。

第三，同一结构族内部，扩展深度或宽度的收益有限且依赖结构。ResNet56 相比 ResNet20 的 clean accuracy 和 mild mean accuracy 更高，但最坏点仍接近随机猜测。MobileNetV2 从 x0.5 扩到 x1.0 后参数量增加到 3.2 倍左右，mild mean accuracy 只从 `13.35%` 增到 `14.83%`，强扰动下几乎没有改善。

第四，目标算子链条和算子类型比参数量更关键。MobileNetV2 和 ShuffleNetV2 有 53 到 57 个注入算子，且包含大量 depthwise/pointwise 分解路径，非线性误差会在更多算子调用中反复进入。VGG11-BN 参数量最大但注入算子数只有 11，在轻中等 alpha 下显著更稳，mild mean accuracy 达到 `87.30%`。这说明对输入端非线性而言，误差传播的算子链条长度、归一化/激活排列、分支结构和卷积类型，比单纯参数量更能解释鲁棒性差异。

阶段结论：网络参数量不能作为非线性鲁棒性的代理指标。后续模型选择或硬件映射时，应优先关注目标矩阵算子的调用次数、stage 过渡层、depthwise/pointwise 组合、残差路径和激活分布，而不是只看模型大小。

## 2. 随机高斯噪声注入与非线性失真的不同影响机制

### 2.1 精度对比

第 2 项实验使用同一个 ResNet20 clean checkpoint，在目标算子输入端分别注入：

- 相对幅值高斯噪声：`x + N(0, (sigma * max|x|)^2)`；
- 固定 alpha 三次非线性：`f_alpha(u)=alpha*u^3+(1-alpha)*u`；
- 每算子每次调用随机 alpha：`alpha ~ Uniform(-1,1)`。

![高斯噪声与非线性精度对比](../outputs/extension_research/figures/mechanism_gaussian_vs_nonlinear_accuracy.png)

| 扰动类型 | 参数 | Accuracy |
|---|---:|---:|
| Clean | 0 | 89.01% |
| Gaussian | sigma=0.01 | 88.07% |
| Gaussian | sigma=0.02 | 83.72% |
| Gaussian | sigma=0.05 | 46.87% |
| Gaussian | sigma=0.10 | 10.77% |
| Gaussian | sigma=0.20 | 10.42% |
| Gaussian | sigma=0.30 | 9.59% |
| Nonlinear | alpha=-0.1 | 72.36% |
| Nonlinear | alpha=0.1 | 72.56% |
| Nonlinear | alpha=-0.2 | 32.86% |
| Nonlinear | alpha=0.2 | 22.75% |
| Nonlinear | alpha=-1 | 11.01% |
| Nonlinear | alpha=1 | 10.57% |
| Random nonlinear | alpha~U[-1,1] | 12.30% |

### 2.2 Logits 漂移对比

![logits 漂移机制对比](../outputs/extension_research/figures/mechanism_logit_drift.png)

| 扰动 | Logit Relative L2 | Cosine Drift | KL(clean||pert) | Argmax Flip | 子集扰动 Acc |
|---|---:|---:|---:|---:|---:|
| Gaussian sigma=0.05 | 0.775 | 0.368 | 1.963 | 51.90% | 47.66% |
| Gaussian sigma=0.10 | 1.372 | 0.976 | 8.085 | 89.31% | 10.79% |
| Gaussian sigma=0.20 | 80.404 | 0.993 | 24.629 | 90.14% | 10.11% |
| Nonlinear alpha=-0.1 | 0.601 | 0.161 | 1.012 | 25.73% | 71.83% |
| Nonlinear alpha=0.1 | 0.586 | 0.177 | 0.733 | 26.90% | 72.51% |
| Nonlinear alpha=0.2 | 0.972 | 0.719 | 2.833 | 77.64% | 22.22% |
| Random nonlinear | alpha~U[-1,1] | 1.384 | 0.877 | 7.563 | 85.60% | 13.72% |

### 2.3 机制差异

高斯噪声是零均值、元素级随机、输入无关的加性扰动。单层看，它不带固定形状偏置；但在每个目标算子输入端反复注入后，方差会逐层累积，并且会直接降低类别 margin。实验中 `sigma=0.01` 基本可承受，`sigma=0.02` 已出现约 5.3 个百分点下降，`sigma=0.05` 将准确率打到 `46.87%`，`sigma>=0.1` 后接近随机猜测。

三次非线性失真是输入相关、确定性形变。误差项可写成：

```text
x_alpha - x = alpha * M * (u^3 - u)
```

其中 `M=max|x|`，`u=x/M`。它不是在所有位置均匀加噪，而是按激活幅值系统性改变中间区间的响应；在 `u in {-1,0,1}` 时误差为零，在中间幅值区域误差最大。因此，它会改变激活分布形状和特征方向，而不仅仅是提高随机方差。

从结果看，弱固定非线性 `alpha=±0.1` 的 logit relative L2 约 `0.59-0.60`，argmax flip 约 `26%`，准确率仍有 `72%` 左右；但 `alpha=0.2` 已使 cosine drift 增至 `0.719`，argmax flip 达到 `77.64%`。这体现了非线性误差的阈值效应：当表示形状被系统性拉弯到一定程度后，分类边界会快速失效。

随机 per-occurrence alpha 进一步改变了问题性质。固定 alpha 至少在整网中保持同一种方向的形变，而随机 alpha 让每个算子调用面对不同方向、不同强度的曲线。模型很难在一次 forward 内保持一致表征，最终准确率只有 `12.30%`。这也是项目主线中 NAT 难以单独解决严格随机 alpha 的核心原因。

阶段结论：高斯噪声主要体现为方差注入和 margin 侵蚀；非线性失真主要体现为输入相关的系统性特征重映射。二者都能造成精度下降，但机制不同，不能用“加一点随机噪声训练”直接替代非线性鲁棒性评估。

## 3. 量化误差与非线性误差对推理精度的联合影响

### 3.1 实验设置

第 3 项比较三种推理方式：

| 方法 | 含义 |
|---|---|
| `uniform_quant` | 每个目标算子输入先做对称均匀量化，不模拟硬件非线性 |
| `uniform_quant_plus_random_nonlinear` | 先做均匀量化，再经过随机 alpha 三次非线性 |
| `fpc_reference` | 固定点 bit-serial 参考，每个 bit-plane 落在 `{ -1,0,+1 }` 不动点 |

注意：`uniform_quant` 和 `fpc_reference` 在数值精度上都对应 B bit 固定点近似，但计算编码不同。前者把多级量化值直接送入非线性硬件；后者把量化整数拆成 bit-plane，每个 plane 的输入值是三次函数不动点。

### 3.2 精度结果

![量化与非线性叠加精度](../outputs/extension_research/figures/quantization_nonlinearity_accuracy.png)

| Bits | Uniform Quant | Uniform Quant + Random Nonlinear | FPC Reference |
|---:|---:|---:|---:|
| 1 | 10.21% | 10.21% | 10.21% |
| 2 | 19.73% | 11.33% | 19.78% |
| 3 | 70.07% | 12.92% | 70.17% |
| 4 | 85.94% | 12.02% | 85.79% |
| 5 | 88.31% | 14.63% | 88.33% |
| 6 | 89.06% | 14.29% | 89.14% |
| 8 | 88.75% | 13.07% | 88.77% |

Clean baseline 为 `89.01%`，裸随机非线性 baseline 为约 `12.5%`。

### 3.3 误差分解

![量化与非线性误差分解](../outputs/extension_research/figures/quantization_error_decomposition.png)

| Bits | Quantization Drop | Extra Nonlinear Drop After Quantization | Total Drop |
|---:|---:|---:|---:|
| 1 | 78.81% | 0.00% | 78.81% |
| 2 | 69.29% | 8.40% | 77.69% |
| 3 | 18.95% | 57.15% | 76.09% |
| 4 | 3.08% | 73.92% | 76.99% |
| 5 | 0.71% | 73.67% | 74.38% |
| 6 | -0.05% | 74.77% | 74.72% |
| 8 | 0.27% | 75.68% | 75.94% |

这里：

```text
Quantization Drop = Clean Acc - Quant Acc
Extra Nonlinear Drop = Quant Acc - Quant+Nonlinear Acc
Total Drop = Clean Acc - Quant+Nonlinear Acc
```

### 3.4 联合影响结论

第一，单纯量化误差是可控的。B=1 和 B=2 因表示能力不足几乎不可用；B=3 出现明显跃迁，达到 `70.07%`；B=4 达到 `85.94%`；B=5 之后基本接近 clean。也就是说，对这个 ResNet20 checkpoint 而言，5 到 6 bit 的激活输入量化误差已经很小。

第二，普通多级量化不能消除非线性误差。B=5、B=6、B=8 的 `uniform_quant` 已经接近 clean，但一旦这些多级量化值继续经过随机三次非线性，准确率又回到 `13%-15%` 左右。增加 bit 数降低了量化误差，却没有让输入值落在非线性不动点上，因此非线性误差仍然主导。

第三，量化误差和非线性误差不是简单同类误差。低 bit 时，总误差主要来自量化表示不足；高 bit 时，量化误差几乎消失，但非线性额外下降达到约 `74%-76%`。这说明“把激活量化得更精细”不能自然提升随机非线性硬件上的鲁棒性。

第四，FPC 的作用是把联合误差重新约束为量化误差。`fpc_reference` 在各 bit 下几乎与 `uniform_quant` 的精度一致，因为它保留了同样的固定点量化近似；同时它不会遭受 `uniform_quant_plus_random_nonlinear` 的额外非线性崩溃，因为每个 bit-plane 输入只取 `{ -1,0,+1 }`。因此，FPC 不是普通量化，而是“量化 + 面向非线性不动点的计算编码”。

阶段结论：若硬件存在输入端三次非线性，部署分析不能只做量化精度评估。普通量化只能控制数字近似误差，不能控制模拟非线性误差；FPC 这类编码方法才把非线性误差转化为可随 bit 数调节的量化误差。

## 4. 综合结论

本次拓展研究得到三个顺序结论。

第一，网络结构和参数量会影响非线性误差敏感性，但参数量不是主因。所有结构在强非线性下都会接近随机猜测；轻中等扰动下，目标算子链条、结构族、卷积拆分方式和归一化路径更重要。VGG11-BN 参数量最大且目标算子最少，对 mild alpha 更稳；MobileNetV2 和 ShuffleNetV2 目标算子多，轻中等扰动下退化更明显。

第二，高斯噪声与非线性失真的鲁棒性机制不同。高斯噪声是元素级随机方差注入，主要侵蚀 margin；三次非线性是输入相关的系统性曲线重映射，会按激活幅值改变分布形状和特征方向。随机 per-occurrence alpha 又会打破层间形变一致性，因此更接近真实 CIM 随机非理想风险。

第三，量化误差与非线性误差必须联合建模。B=5/6 普通均匀量化已经接近 clean，但在随机非线性硬件中仍然崩溃到约 `14%`。这说明量化精度提高不能替代非线性鲁棒编码。FPC 的关键价值在于把每个 bit-plane 放到 `{ -1,0,+1 }` 不动点上，使剩余误差主要退化为可控的固定点量化误差。

后续建议：

1. 结构维度继续扩展到更多同族模型，例如 ResNet32/44、VGG13/16、MobileNetV2 x1.4，以分离深度、宽度和目标算子数的影响；
2. 在训练侧加入高斯噪声训练、NAT 和二者混合训练的对照，验证机制差异是否能通过训练策略体现；
3. 将量化和非线性叠加实验扩展到权重量化、ADC/DAC 量化与激活 FPC 的组合，形成更接近 CIM 部署链路的联合误差模型。
