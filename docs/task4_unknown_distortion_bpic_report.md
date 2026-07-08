# 任务四：未知随机失真下的公式无关鲁棒方法 BPIC

## 1. 问题定义

任务四假设我们不知道扰动公式是什么，只知道存算芯片中的模拟输入会发生随机失真：

```text
不知道 f(x) 的具体函数形式；
不知道当前推理、当前层、当前算子调用中出现的失真参数；
不针对赛题三次多项式公式做反函数推导；
只允许通过黑盒方式观察少量输入 anchor 的实际硬件响应。
```

因此，任务四方法不能依赖：

```text
f_alpha(u) = alpha * u^3 + (1 - alpha) * u
```

也不能使用：

```text
u in {-1,0,1} 时 u^3 = u
```

这意味着任务四与任务三 FPC 的定位不同：

| 方法 | 是否依赖赛题公式结构 | 是否需要知道当前扰动参数 | 核心机制 |
|---|---|---|---|
| FPC | 依赖固定点不变量，但不依赖 alpha 数值 | 不需要 | 把输入编码到官方三次映射固定点 |
| BPIC | 不依赖任何显式公式 | 不需要 | 用 pilot anchor 黑盒测量当前响应并做分段线性逆校准 |

## 2. 方法名称

```text
BPIC:
Blind Pilot Inversion Calibration
盲式 pilot 逆校准
```

BPIC 解决的问题是：

```text
当失真函数未知、随机、每个算子调用都可能变化时，
如何不使用公式推导，仍然让实际进入矩阵计算的信号接近期望激活。
```

## 3. 核心思想

对某个矩阵算子输入激活 `x`，BPIC 不直接把 `x` 送入存算阵列，而是先在当前算子调用环境下发送一组已知 pilot anchors：

```text
z_k in [-1, 1], k = 1 ... K
```

硬件返回或可观测到对应响应：

```text
y_k = f_current(z_k)
```

这里 `f_current` 是当前算子调用中真实出现的黑盒失真。BPIC 不知道它的公式，只知道 pilot 输入输出对：

```text
{(z_k, y_k)}_{k=1}^K
```

然后 BPIC 拟合一个分段线性逆映射：

```text
g_current(y_k) ~= z_k
```

对真实激活 `x` 做预失真：

```text
z = g_current(x)
```

硬件实际失真后：

```text
f_current(z) ~= x
```

从而矩阵算子看到的输入接近期望激活。

## 4. 算法流程

对每个目标矩阵算子调用：

1. 计算当前输入张量最大幅度：

```text
M = max(abs(x))
u = clip(x / M, -1, 1)
```

2. 生成 `K` 个均匀 pilot anchors：

```text
z_k = linspace(-1, 1, K)
```

3. 通过当前硬件黑盒测量：

```text
y_k = hardware_distortion_current_call(z_k)
```

4. 按 `y_k` 排序，得到经验逆映射：

```text
y_sorted -> z_sorted
```

5. 对每个输入元素 `u_i` 做一维分段线性插值：

```text
z_i = interp(u_i, y_sorted, z_sorted)
```

6. 将预校准输入送入硬件：

```text
hardware_input = z_i * M
hardware_output = f_current(hardware_input)
```

7. 矩阵算子使用 `hardware_output` 进行计算。

伪代码：

```text
for each operator call:
    M = max(abs(x))
    u = x / M

    anchors = linspace(-1, 1, K)
    observed = black_box_query(anchors)

    inverse = piecewise_linear_inverse(observed, anchors)
    z = inverse(u) * M

    x_actual = black_box_distortion(z)
    y = matrix_operator(x_actual, weight)
```

## 5. 创新点

### 5.1 不使用公式

BPIC 不需要知道失真是三次、多项式、tanh、gamma、sinusoid，还是混合扰动。实验中，公式只用于模拟“硬件黑盒”；BPIC 算法本身只读取 pilot 输入输出对。

### 5.2 每次调用自适应

如果每一层、每一次算子调用的失真都不同，BPIC 会在每次调用时重新采样 pilot 并拟合当前逆映射。因此它适合处理：

```text
layer-varying distortion
operator-call-varying distortion
time-varying distortion
device-drift-induced distortion
```

### 5.3 与训练解耦

BPIC 可以直接应用到已有 checkpoint 上，不要求重新训练模型。它也可以与 FPC、量化训练或 nonlinearity-aware training 组合。

### 5.4 工程可解释

BPIC 的开销来自 `K` 个 pilot anchors，而不是 `K` 次完整矩阵乘法。pilot anchors 是对输入响应曲线的标定，理论上可以由小型校准通路或阵列边缘参考单元完成。

## 6. 适用假设与限制

BPIC 的基本假设是：

1. 当前算子调用期间，pilot 和真实 payload 看到的是同一个或足够接近的失真函数；
2. 失真在 `[-1,1]` 上可由有限个 anchor 近似；
3. 失真响应不是完全随机白噪声，否则任何输入预校准都没有可学习结构；
4. 如果失真强非单调，分段线性逆会出现多值问题，需要更多 anchors 或分支选择策略。

本实验中的 BPIC 使用排序后的分段线性逆映射，对轻度非单调情况也有一定鲁棒性，但它不是任意失真的万能解。

## 7. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 模型 | ResNet20 clean checkpoint |
| 测试集 | 10,000 张 |
| baseline | 未知随机黑盒失真直接推理 |
| 方法 | BPIC |
| pilot anchors | 9, 17, 33, 65 |
| random repeats | 5 |
| BPIC repeats | 3 |

目标算子：

```text
Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, Linear
```

实现文件：

```text
src/nonlinear.py::BlackBoxRandomDistortionInjector
src/nonlinear.py::BlindPilotInverseInjector
scripts/evaluate_cifar_blackbox_pilot.py
```

结果目录：

```text
outputs/paper_task4_bpic_cifar/tables/cifar_bpic_summary.csv
outputs/paper_task4_bpic_cifar/figures/cifar_bpic_family_comparison.png
outputs/paper_task4_bpic_cifar/figures/cifar_bpic_anchor_ablation.png
```

## 8. 黑盒扰动族

为了验证 BPIC 不依赖赛题公式，实验构造了五类黑盒失真：

| family | 说明 | BPIC 是否知道公式 |
|---|---|---|
| contest_cubic | 赛题三次非线性，`alpha in [-1,1]` 随机 | 不知道 |
| gamma | 随机幂律压缩/扩张 | 不知道 |
| tanh | 随机 tanh 饱和型压缩 | 不知道 |
| sinusoid | 随机正弦形变 | 不知道 |
| mixed | 每次算子调用随机选择上述一种失真 | 不知道 |

注意：这些公式只用于实验脚本模拟硬件黑盒。BPIC 算法只拿到 pilot anchors 的观测响应，不使用公式结构。

## 9. 实验结果

![BPIC family comparison](../outputs/paper_task4_bpic_cifar/figures/cifar_bpic_family_comparison.png)

| family | random_acc | random_worst | best_bpic_anchors | best_bpic_acc | best_bpic_worst | absolute_gain | clean_gap |
|:---|---:|---:|---:|---:|---:|---:|---:|
| contest_cubic | 0.1312 | 0.1184 | 65 | 0.8937 | 0.8936 | 0.7625 | 0.0002 |
| gamma | 0.1132 | 0.1060 | 65 | 0.8828 | 0.8802 | 0.7696 | 0.0111 |
| tanh | 0.1048 | 0.1020 | 65 | 0.8936 | 0.8934 | 0.7888 | 0.0003 |
| sinusoid | 0.1277 | 0.1176 | 33 | 0.8935 | 0.8932 | 0.7658 | 0.0004 |
| mixed | 0.1248 | 0.1211 | 65 | 0.8890 | 0.8882 | 0.7643 | 0.0049 |

结论：

1. 裸黑盒随机扰动将 CIFAR-10 clean checkpoint 从 `0.8939` 降到 `0.1048-0.1312`。
2. BPIC 在不知道公式的情况下，将五类扰动全部恢复到 `0.8828-0.8937`。
3. 对赛题三次扰动，BPIC `65` anchors 达到 `0.8937`，几乎等于 clean。
4. 对 mixed 随机族，BPIC `65` anchors 达到 `0.8890`，只比 clean 低约 `0.49%`。

## 10. anchor 数消融

![BPIC anchor ablation](../outputs/paper_task4_bpic_cifar/figures/cifar_bpic_anchor_ablation.png)

| anchors | contest_cubic | gamma | mixed | sinusoid | tanh |
|---:|---:|---:|---:|---:|---:|
| 9 | 0.8900 | 0.6030 | 0.8149 | 0.8832 | 0.8302 |
| 17 | 0.8935 | 0.7787 | 0.8770 | 0.8929 | 0.8893 |
| 33 | 0.8934 | 0.8541 | 0.8826 | 0.8935 | 0.8928 |
| 65 | 0.8937 | 0.8828 | 0.8890 | 0.8935 | 0.8936 |

观察：

1. 对 contest cubic、sinusoid 等较平滑扰动，`K=9` 已经非常有效。
2. 对 gamma 这类曲率变化更明显的失真，anchor 数越多越好，`K=65` 才接近 clean。
3. mixed 扰动族包含多种形态，`K=17` 已达到 `0.8770`，`K=65` 达到 `0.8890`。
4. 默认推荐 `K=33` 或 `K=65`。如果硬件 pilot 开销敏感，可用 `K=17` 作为低开销配置。

## 11. 与 FPC 的关系

BPIC 和 FPC 不是互相替代，而是面向不同问题假设：

| 维度 | FPC | BPIC |
|---|---|---|
| 是否需要知道失真公式 | 需要知道存在固定点结构 | 不需要 |
| 是否读取 alpha | 不读取 | 不读取 |
| 是否每次调用自适应 | 不需要自适应 | 每次调用用 pilot 自适应 |
| 对赛题三次扰动 | 极强 | 极强 |
| 对未知 gamma/tanh/mixed | 不保证 | 实验有效 |
| 主要开销 | B 次 bit-serial 矩阵 partial | K 个 pilot 响应标定 |
| 适合场景 | 公式已知或固定点不变量可信 | 公式未知、器件响应可黑盒测量 |

实际系统可以组合使用：

```text
FPC 用于已知主要非线性模式；
BPIC 用于在线校准未知漂移、温度变化、批次偏差或新器件响应。
```

## 12. 总结

BPIC 提供了一条不依赖赛题三次公式的鲁棒路线。它把“未知随机失真”的问题转化为“当前算子调用的黑盒响应曲线标定”问题，通过 pilot anchors 拟合分段线性逆映射，使真实硬件响应接近期望输入。

实验显示，在 CIFAR-10 ResNet20 全测试集上：

```text
contest_cubic: random 0.1312 -> BPIC 0.8937
gamma:         random 0.1132 -> BPIC 0.8828
tanh:          random 0.1048 -> BPIC 0.8936
sinusoid:      random 0.1277 -> BPIC 0.8935
mixed:         random 0.1248 -> BPIC 0.8890
```

因此，任务四可以提出：

```text
BPIC 是一种公式无关、调用级自适应、黑盒 pilot 校准的鲁棒存算推理方法。
```

它解决了 FPC 没有覆盖的情形：当我们不知道扰动公式、也不能利用固定点不变量时，仍可通过少量在线 pilot 测量恢复接近 clean 的推理精度。
