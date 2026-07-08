# 统一修正：每次失真独立随机 alpha 协议

## 1. 正确协议

根据最新确认，本项目任务一、任务二、任务三的主协议必须统一为：

```text
每当一个矩阵计算算子的输入端发生非线性失真时，
都重新从 [-1,1] 独立随机采样一个 alpha。
```

对一次前向推理中的第 `l` 个目标算子、第 `c` 次调用，应写成：

```text
alpha_{l,c} ~ Uniform(-1, 1)
x'_{l,c} = nonlinearity(x_{l,c}, alpha_{l,c})
```

其中 `nonlinearity` 必须严格保持赛题给出的形式：

```python
def nonlinearity(x, alpha=0.0):
    max_val = x.abs().max()
    x = x / max_val
    y = alpha * (x**3) + (1 - alpha) * x
    return y * max_val
```

这意味着一次推理不是一个标量 `alpha`，而是一个随机 alpha 场：

```text
A = {alpha_{1,1}, alpha_{2,1}, ..., alpha_{L,1}}
```

如果同一个层在循环、共享模块或多次前向中被再次调用，也应重新采样新的 `alpha`。

## 2. 明确排除的错误理解

以下理解不能作为官方主协议：

| 错误理解 | 问题 |
|---|---|
| 一次推理全网共享一个 `alpha` | 低估了层间随机失配，允许做全局估计/补偿 |
| 一个 batch 只采样一个 `alpha` | 仍然是全局扰动，不是每个算子输入端独立失真 |
| 先估计当前 `alpha` 再反向补偿 | 每个算子遇到的 alpha 都可能不同，估计值无法复用于后续算子 |
| 把 `alpha` 范围缩小到 `[-0.2,0.2]` 或 `[-0.1,0.1]` | 不满足官方 `[-1,1]` 主范围 |
| 只画固定 `alpha` vs accuracy 曲线 | 只能作为诊断图，不能代表随机 alpha 场下的主鲁棒性 |

固定 `alpha` 曲线仍有价值，但它的定位应是“辅助敏感性诊断”：用于解释正负方向、强弱趋势和层敏感性。最终主表和主图必须报告每次独立随机 alpha 场下的 Monte Carlo 统计。

## 3. 代码实现要求

主协议对应的注入器为：

```text
src/nonlinear.py::RandomAlphaNonlinearityInjector
```

该注入器在每个目标算子的 `forward_pre_hook` 内执行：

```python
alpha = random.uniform(-1.0, 1.0)
distorted = nonlinearize_tensor(x, NonlinearityConfig(alpha=alpha))
```

因此采样粒度是：

```text
每个被 hook 的 Conv/Linear/ConvTranspose 算子输入张量，每次 forward 调用，重新采样一次 alpha。
```

一次 CIFAR-10 ResNet20 前向的 alpha trace 验证如下：

| operator call | sampled alpha |
|---|---:|
| `conv1` | -0.352334 |
| `layer1.0.conv1` | -0.698302 |
| `layer1.0.conv2` | 0.301869 |
| `layer1.1.conv1` | -0.855127 |
| `layer1.1.conv2` | 0.071764 |
| `layer1.2.conv1` | -0.268622 |
| `layer1.2.conv2` | -0.884002 |
| `layer2.0.conv1` | 0.014871 |
| `layer2.0.conv2` | -0.925009 |
| `layer2.0.downsample.0` | -0.132709 |
| `layer2.1.conv1` | -0.860289 |
| `layer2.1.conv2` | -0.818574 |

该模型一次前向共有 22 个目标算子调用，前 12 个调用的 alpha 全部不同，说明同一次推理中不同层确实遇到不同随机失真。

## 4. 对任务一的修正

任务一主实验应从“固定 alpha sweep”改为“随机 alpha 场敏感性分析”：

| 分析项 | 正确做法 |
|---|---|
| 整网精度衰减 | 多次随机 alpha 场重复推理，报告 mean / worst / std / percentile |
| 单层敏感性 | 每次只打开一个层的随机 alpha 注入，该层每次调用仍独立采样 `alpha~U(-1,1)` |
| 单层输出分布偏移 | 比较 clean 与随机 alpha 场下的激活均值、方差、相对 L2、cosine drift |
| 逐层累积 | 在全网随机 alpha 场下注入，记录各层 drift 随深度放大的趋势 |
| 图表 | Clean vs random-field mean/worst 柱状图、随机重复箱线图、敏感层排序、activation drift 图 |

固定 `alpha={-1,-0.5,0,0.5,1}` 曲线可以放入附录或辅助分析，用来说明三次映射方向性，但主结论必须来自 `RandomAlphaNonlinearityInjector(-1,1)`。

## 5. 对任务二的修正

任务二的 NAT 训练不能再用“每个 batch 一个全局 alpha”的 scheduler 作为主实现。正确训练流程是：

```text
clean forward:
    z_clean = f(x)

random forward:
    z_rand = f(x; A), A={alpha_l}, alpha_l ~ Uniform(-1,1)
```

fine-tuning 与 from scratch 都应使用每算子每次独立随机 alpha 注入。评价也必须用随机 alpha 场的多次重复统计，而不是只在固定 alpha 网格上 sweep。

当前已有的正确脚本入口：

```text
scripts/run_cifar_task1_random.py
scripts/run_voxforge_task1_random.py
scripts/train_cifar_per_occ_nat.py
scripts/train_voxforge_per_occ_nat.py
```

旧脚本 `train_cifar_task2.py`、`train_voxforge_task2.py` 中的全局 alpha scheduler 只能作为历史基线或辅助 ablation，不应作为最终主协议结果。

## 6. 对任务三的修正

在每次失真独立重采样 alpha 的条件下，任何依赖“读取、估计或复用当前 alpha”的方案都不能作为主方法：

| 方法 | 为什么不能作为主方法 |
|---|---|
| 已知 alpha 逆补偿 | 需要知道当前 alpha；但每个算子每次 alpha 都不同且推理未知 |
| 单一候选 gamma 补偿 | 每次算子调用重采样时，一个 gamma 不能代表后续算子的 alpha |
| 固定 gamma 预补偿 | 无法同时抵消独立随机的正负 alpha 场 |

任务三主方法应转为 distribution-free alpha-independent robust training，例如后续新方案应满足：

```text
不读取 alpha
不估计 alpha
不假设 E[alpha]=0
对任意 alpha∈[-1,1] 或端点最坏情形保持稳定
```

核心思想不是估计 alpha，而是让模型对随机 alpha 场本身稳定：

```text
min_theta E_A[L_task(f_theta(x; A), y)]
        + consistency between independent alpha fields
        + clean-teacher distillation
        + activation vulnerability regularization
```

官方非线性可写成：

```text
x_alpha = x + alpha * max(|x|) * (u^3 - u)
```

因此对 alpha 的解析敏感度为：

```text
d x_alpha / d alpha = max(|x|) * (u^3 - u)
```

当 `u in {-1,0,1}` 时，`u^3-u=0`，该位置对任意 alpha 都不敏感。任务三的创新方向应围绕“随机 alpha 场一致性 + 三次非线性固定点表征 + 敏感层约束”展开，而不是围绕“估计当前 alpha”展开。

## 7. 主评估指标

每个数据集至少报告：

| 指标 | CIFAR-10 | VoxForge |
|---|---|---|
| clean 指标 | clean accuracy | clean WER / CER |
| random-field mean | 多个随机 alpha 场的平均 accuracy | 平均 WER / CER |
| random-field worst | 多个随机种子中最差 accuracy | 最差 WER / CER |
| random-field std | accuracy 标准差 | WER / CER 标准差 |
| MC-K 曲线 | K 次随机推理概率平均后的 accuracy | K 次随机推理解码或 logit 平均后的 WER / CER |
| layer sensitivity | 单层随机注入造成的 accuracy drop | 单层随机注入造成的 CER/WER 上升 |

最终报告中，所有主表必须明确写出：

```text
protocol = per-operator per-call random alpha
alpha_{l,c} ~ Uniform(-1,1)
official nonlinearity unchanged
no true-alpha access at inference
```
