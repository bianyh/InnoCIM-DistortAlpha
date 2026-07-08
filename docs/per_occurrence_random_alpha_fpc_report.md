# 每算子每次随机 alpha 协议下的任务一、任务二与任务三 FPC 方法报告

## 0. 核心修正

本报告只采用如下问题定义：

```text
每当一个矩阵计算算子输入端发生非线性失真时，
该次失真的 alpha 在 [-1,1] 范围内随机出现。
同一次推理中，不同层、不同算子、不同调用可以遇到不同 alpha。
```

报告不假设 `E[alpha]=0`，不假设 alpha 服从均匀分布，不读取当前 alpha，也不估计当前 alpha。所有主结论都必须在“每算子每次 alpha 独立未知，范围为 `[-1,1]`”这个约束下成立。

官方非线性函数保持不变：

```python
def nonlinearity(x, alpha=0.0):
    max_val = x.abs().max()
    x = x / max_val
    y = alpha * (x**3) + (1 - alpha) * x
    return y * max_val
```

## 1. 任务一：正确协议下的敏感性分析

### 1.1 实验对象

| 数据集 | 任务 | 模型 | checkpoint | 主指标 |
|---|---|---|---|---|
| CIFAR-10 | 图像分类 | ResNet20 | `outputs/task2/checkpoints/cifar/cifar_resnet20_clean_best.pt` | Top-1 Accuracy |
| VoxForge | 语音识别 | CRNN-CTC | `outputs/task2/checkpoints/voxforge/voxforge_crnn_clean_best.pt` | WER / CER |

注入位置为矩阵计算算子输入端：

```text
Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, Linear
```

主实现为：

```text
src/nonlinear.py::RandomAlphaNonlinearityInjector
```

该注入器在每个目标算子的 forward pre-hook 中重新采样一个 alpha，因此同一次推理中的各层 alpha 不共享。

### 1.2 CIFAR-10 整网随机场结果

结果目录：

```text
outputs/task1_random
```

| eval type | accuracy | worst accuracy | std | 说明 |
|---|---:|---:|---:|---|
| clean | 0.8994 | 0.8994 | 0.0000 | 无失真 |
| per-call random alpha | 0.1318 | 0.1201 | 0.0089 | 每算子每次随机 alpha |
| random MC-1 | 0.1074 | 0.1074 | 0.0000 | 整网重复一次 |
| random MC-3 | 0.1387 | 0.1387 | 0.0000 | 末端概率平均 |
| random MC-5 | 0.1260 | 0.1260 | 0.0000 | 末端概率平均 |

![CIFAR random-field summary](../outputs/task1_random/figures/cifar_random_field_summary.png)

结论：正确协议下，clean checkpoint 从 `0.8994` 降至约 `0.13`，接近随机分类。末端 MC 平均无法解决逐层误差累积。

### 1.3 CIFAR-10 单层敏感性

| layer | accuracy | accuracy drop |
|---|---:|---:|
| `layer2.0.conv2` | 0.5742 | 0.3252 |
| `layer3.0.conv1` | 0.6563 | 0.2432 |
| `layer3.1.conv1` | 0.6777 | 0.2217 |
| `layer2.2.conv1` | 0.7520 | 0.1475 |
| `layer1.1.conv1` | 0.7646 | 0.1348 |
| `layer1.0.conv1` | 0.7754 | 0.1240 |

![CIFAR layer sensitivity](../outputs/task1_random/figures/cifar_layer_random_sensitivity.png)

### 1.4 CIFAR-10 激活漂移

| layer | mean relative L2 | mean cosine drift |
|---|---:|---:|
| `layer2.0.conv1` | 1.29 | 0.12 |
| `fc` | 1.20 | 0.77 |
| `layer3.2.conv2` | 1.13 | 0.79 |
| `layer3.0.conv1` | 0.88 | 0.34 |
| `layer1.0.conv1` | 0.27 | 0.00 |
| `conv1` | 0.15 | 0.00 |

![CIFAR activation drift](../outputs/task1_random/figures/cifar_random_activation_drift.png)

### 1.5 VoxForge 整网随机场结果

结果目录：

```text
outputs/task1_random_voxforge
```

| eval type | WER | CER | 说明 |
|---|---:|---:|---|
| clean | 0.9956 | 0.7945 | CRNN-CTC clean baseline |
| per-call random alpha | 0.9825 | 0.8596 | mean over 3 random-field repeats |
| random MC-1 | 0.9913 | 0.7803 | 小测试集波动 |
| random MC-3 | 1.0000 | 0.9490 | 末端平均不稳定 |

![VoxForge random-field summary](../outputs/task1_random_voxforge/figures/voxforge_random_field_summary.png)

### 1.6 VoxForge 单层敏感性与漂移

| layer | CER | CER increase |
|---|---:|---:|
| `cnn.3` | 0.8413 | 0.0468 |
| `cnn.0` | 0.8170 | 0.0226 |
| `classifier` | 0.7845 | -0.0100 |

| layer | mean relative L2 | mean cosine drift |
|---|---:|---:|
| `cnn.3` | 0.62 | 0.00 |
| `classifier` | 0.35 | 0.10 |
| `cnn.0` | 0.35 | 0.01 |

![VoxForge layer sensitivity](../outputs/task1_random_voxforge/figures/voxforge_layer_random_sensitivity.png)

![VoxForge activation drift](../outputs/task1_random_voxforge/figures/voxforge_random_activation_drift.png)

## 2. 任务二：正确协议下的非线性感知训练

任务二的主训练协议不再使用“每个 batch 一个全局 alpha”。正确实现为：每个训练 forward 中，目标矩阵算子输入端每次调用重新采样 alpha。

主脚本：

```text
scripts/train_cifar_per_occ_nat.py
scripts/train_voxforge_per_occ_nat.py
```

### 2.1 CIFAR-10 结果

| 方法 | 初始化 | epoch | clean acc | random mean acc | random worst acc | random std |
|---|---|---:|---:|---:|---:|---:|
| clean checkpoint eval | clean checkpoint | 0 | 0.8994 | 0.1271 | 0.1074 | 0.0120 |
| per-occ NAT fine-tuning | clean checkpoint | 5 | 0.8711 | 0.1250 | 0.0967 | 0.0277 |
| per-occ NAT scratch | random init | 5 | 0.2520 | 0.1232 | 0.1025 | 0.0141 |

结果目录：

```text
outputs/task2_random_clean_eval
outputs/task2_random_nat_ft
outputs/task2_random_nat_scratch
```

结论：普通 per-occurrence NAT 短训没有显著提升 CIFAR-10 的随机 alpha 场鲁棒性，说明该问题需要结构级机制，而不是只靠常规噪声增强。

### 2.2 VoxForge 结果

| 方法 | 初始化 | epoch | clean WER | clean CER | random mean CER | random worst CER | random std CER |
|---|---|---:|---:|---:|---:|---:|---:|
| clean checkpoint eval | clean checkpoint | 0 | 0.9956 | 0.7945 | 0.8596 | 0.9023 | 0.0409 |
| per-occ NAT fine-tuning | clean checkpoint | 3 | 0.9913 | 0.7803 | 0.8499 | 0.8956 | 0.0326 |
| per-occ NAT scratch | random init | 3 | 1.0000 | 0.9507 | 0.9499 | 0.9532 | 0.0047 |

结果目录：

```text
outputs/task2_random_voxforge_clean_eval
outputs/task2_random_voxforge_nat_ft
outputs/task2_random_voxforge_nat_scratch
```

结论：VoxForge fine-tuning 对 CER 有小幅改善，但仍不能从根本上消除随机 alpha 场带来的退化；scratch 小预算下不可用。

## 3. 任务三：FPC 固定点位串行编码

### 3.1 方法名称

```text
FPC:
Fixed-Point Bit-Serial Coding
固定点位串行编码
```

FPC 解决的问题是：

```text
alpha 在每个算子、每次调用中未知且可能不同；
不能读取 alpha；
不能估计 alpha；
不能假设 alpha 的均值、分布或时间相关性。
```

FPC 不试图抵消 alpha，而是让每一次送入存算阵列的输入脉冲本身落在官方三次映射的固定点上，从而对任意 alpha 恒等。

### 3.2 代数不变量

官方非线性在归一化变量 `u` 上为：

```text
f_alpha(u) = alpha*u^3 + (1-alpha)*u
```

当：

```text
u ∈ {-1, 0, 1}
```

有：

```text
u^3 = u
f_alpha(u) = alpha*u + (1-alpha)*u = u
```

这个等式对任意 `alpha` 成立，不需要任何分布假设。

### 3.3 位串行编码

对一个算子输入张量 `x`，令：

```text
M = max(|x|)
u = clip(x/M, -1, 1)
```

用 B bit 量化幅值：

```text
q = round(|u| * (2^B - 1))
```

将 `q` 展开为二进制 bit-plane：

```text
q = sum_{i=0}^{B-1} b_i * 2^i
```

构造第 `i` 个固定点输入脉冲：

```text
p_i = sign(u) * b_i * M
```

则 `p_i/M` 的元素只可能是 `-1,0,1`，因此：

```text
nonlinearity(p_i, alpha) = p_i, for all alpha in [-1,1]
```

原始输入由位平面加权重构：

```text
x_B = sum_{i=0}^{B-1} (2^i / (2^B - 1)) * p_i
```

对矩阵算子 `W`：

```text
W*x_B + bias
= sum_i c_i * (W*p_i + bias)
```

其中：

```text
c_i = 2^i / (2^B - 1)
sum_i c_i = 1
```

因此 bias 也能正确重构。硬件每次只看到固定点脉冲 `p_i`，所以 alpha 非线性不会改变任何 bit-plane；最终误差只剩 B bit 量化误差，而不是 alpha 失真误差。

### 3.4 与前面被删除方案的根本区别

| 方案类型 | 依赖 | 为什么不用 |
|---|---|---|
| 读取 alpha 后逆补偿 | 当前 alpha 可知 | 违反未知 per-call alpha 约束 |
| 估计一个补偿参数 | alpha 在一段时间内稳定 | 每层每次重采样时不成立 |
| 随机重复平均 | alpha 分布均值 | 原题未给出均值条件，不能使用 |
| 普通 NAT | 训练吸收扰动 | 短训结果显示效果不足 |
| FPC | 官方三次函数固定点 | 对任意 alpha 恒等，不依赖分布 |

### 3.5 工程实现

新增实现：

```text
src/nonlinear.py::BitSerialFixedPointInjector
src/nonlinear.py::fixed_point_bitserial_planes
```

评估脚本：

```text
scripts/evaluate_cifar_bitserial_fixedpoint.py
scripts/evaluate_voxforge_bitserial_fixedpoint.py
```

输出目录：

```text
outputs/task3_fpc_cifar
outputs/task3_fpc_voxforge
```

## 4. FPC 实验结果

### 4.1 CIFAR-10

| 方法 | B bits | accuracy | loss |
|---|---:|---:|---:|
| clean | 0 | 0.8994 | 0.3092 |
| per-call random alpha | 0 | 0.1227 | 9.0274 |
| FPC | 1 | 0.0850 | 9.5280 |
| FPC | 2 | 0.2227 | 6.7022 |
| FPC | 3 | 0.7246 | 1.0962 |
| FPC | 4 | 0.8594 | 0.4443 |
| FPC | 5 | 0.8926 | 0.3338 |
| FPC | 6 | 0.8994 | 0.3128 |
| FPC | 8 | 0.8984 | 0.3117 |

![CIFAR FPC summary](../outputs/task3_fpc_cifar/figures/cifar_fpc_accuracy_summary.png)

![CIFAR FPC bit curve](../outputs/task3_fpc_cifar/figures/cifar_fpc_bit_curve.png)

结论：FPC `B=6` 在不读取 alpha、不假设分布的情况下，将 CIFAR-10 accuracy 从随机场下的 `0.1227` 恢复到 `0.8994`，与 clean 持平。

### 4.2 VoxForge

| 方法 | B bits | WER | CER |
|---|---:|---:|---:|
| clean | 0 | 0.9956 | 0.7945 |
| per-call random alpha | 0 | 0.9825 | 0.8596 |
| FPC | 1 | 1.0000 | 0.9499 |
| FPC | 2 | 0.9956 | 0.8555 |
| FPC | 3 | 0.9825 | 0.7878 |
| FPC | 4 | 0.9956 | 0.7945 |
| FPC | 5 | 0.9913 | 0.7937 |
| FPC | 6 | 0.9956 | 0.7962 |

![VoxForge FPC summary](../outputs/task3_fpc_voxforge/figures/voxforge_fpc_cer_summary.png)

![VoxForge FPC bit curve](../outputs/task3_fpc_voxforge/figures/voxforge_fpc_bit_curve.png)

结论：FPC `B=3` 即可将 VoxForge CER 从随机场下的 `0.8596` 恢复到 `0.7878`，接近 clean CER `0.7945`。当前 ASR baseline 本身较弱，因此 CER 比 WER 更适合作为主指标。

## 5. 方法创新点

1. **分布无关**：不使用 `E[alpha]`、不假设均匀分布、不要求 alpha 稳定。
2. **逐调用有效**：每个 bit-plane 对任意 alpha 恒等，因此适合每层每次 alpha 都不同的情况。
3. **不需要 alpha 估计**：推理时不读取、不反推、不校准 alpha。
4. **误差类型转换**：把不可控的模拟非线性误差转换为可控的 B bit 激活量化误差。
5. **可调开销**：B 越大越接近 clean；CIFAR-10 上 B=4 已恢复到 `0.8594`，B=6 与 clean 持平。
6. **可与任务一敏感性结合**：正式工程实现中可只对敏感层使用较高 B，对低敏感层使用较低 B，以降低计算开销。

## 6. 局限性与后续优化

FPC 的主要代价是每个矩阵算子需要 B 次 bit-plane 执行。后续可做三类优化：

| 优化方向 | 说明 |
|---|---|
| 敏感层自适应 B | 对 `layer2.0.conv2`、`layer3.0.conv1` 等敏感层用 B=6，低敏感层用 B=3/4 |
| 量化感知微调 | 在训练阶段加入 FPC bit-plane 量化，使 B=3/4 达到更高精度 |
| 稀疏 bit-plane 跳过 | 当某一 bit-plane 全零或稀疏度很高时跳过或压缩执行 |

## 7. 可复现实验命令

任务三 CIFAR-10 FPC：

```powershell
python scripts\evaluate_cifar_bitserial_fixedpoint.py `
  --output-dir outputs\task3_fpc_cifar `
  --checkpoint outputs\task2\checkpoints\cifar\cifar_resnet20_clean_best.pt `
  --test-subset 1024 --batch-size 128 --workers 0 `
  --random-repeats 3 --bits "1,2,3,4,5,6,8"
```

任务三 VoxForge FPC：

```powershell
python scripts\evaluate_voxforge_bitserial_fixedpoint.py `
  --output-dir outputs\task3_fpc_voxforge `
  --checkpoint outputs\task2\checkpoints\voxforge\voxforge_crnn_clean_best.pt `
  --hidden-size 96 --train-size 120 --val-size 30 --test-size 30 `
  --max-duration 5.0 --max-text-length 110 --batch-size 16 --workers 0 `
  --random-repeats 3 --bits "1,2,3,4,5,6"
```

