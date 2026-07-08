# 任务三 FPC 鲁棒性增强方法完整实验报告

## 1. 实验目标与协议

本报告补齐任务三“固定点位串行编码（FPC, Fixed-Point Bit-Serial Coding）”的完整实验，包括：

1. 已完成训练模型在未知随机非线性扰动下的推理对比；
2. 从零开始的完整失真扰动训练，以及扰动推理对比；
3. `1-bit` 到 `8-bit` 的完整 n-bit 消融；
4. CIFAR-10 全测试集主实验与 VoxForge 小规模跨任务验证。

严格协议如下：

```text
每个矩阵计算算子输入端发生失真时，alpha 独立随机采样自 [-1, 1]。
同一次推理中，不同层、不同算子、不同调用可以遇到不同 alpha。
训练与推理均不读取、不估计、不假设当前 alpha。
```

官方非线性函数保持不变：

```python
def nonlinearity(x, alpha=0.0):
    max_val = x.abs().max()
    x = x / max_val
    y = alpha * (x**3) + (1 - alpha) * x
    return y * max_val
```

最终 FPC 评估脚本已明确使用随机 alpha：

```text
fpc_hardware_alpha_sampling = random_uniform_minus1_1_per_bit_plane
```

实现文件：

```text
src/nonlinear.py
scripts/evaluate_cifar_bitserial_fixedpoint.py
scripts/evaluate_voxforge_bitserial_fixedpoint.py
scripts/train_cifar_per_occ_nat.py
scripts/aggregate_paper_fpc_results.py
scripts/run_final_fpc_random_alpha_evals.ps1
```

## 2. 方法简述

FPC 的核心思想不是估计 `alpha`，也不是对 `alpha` 做均值补偿，而是把进入存算阵列的每一次模拟输入脉冲拆成若干个固定点 bit-plane。对归一化输入：

```text
f_alpha(u) = alpha * u^3 + (1 - alpha) * u
```

当：

```text
u in {-1, 0, 1}
```

有：

```text
u^3 = u
f_alpha(u) = u
```

该等式对任意 `alpha in [-1, 1]` 成立。因此 FPC 将普通激活 `x` 量化并分解为多组 `{-M, 0, +M}` bit-plane，每个 bit-plane 在硬件非线性下保持不变，再把各 bit-plane 的矩阵计算结果按二进制权重数字累加：

```text
M = max(abs(x))
u = x / M
q = round(abs(u) * (2^B - 1))
p_i = sign(u) * bit_i(q) * M

y = sum_i 2^i / (2^B - 1) * op(p_i)
```

因此，未知随机非线性误差被转化为可控的 B-bit 激活量化误差。

## 3. CIFAR-10 主实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 训练集 / 验证集 / 测试集 | 45k / 5k / 10k |
| 模型 | CIFAR-10 ResNet20 |
| 参数量 | 272,474 |
| 注入算子 | `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`, `Linear` |
| 随机扰动 baseline | 每个目标算子调用独立采样 `alpha ~ U[-1,1]` |
| FPC 扰动 | 每个 bit-plane 仍模拟随机 `alpha ~ U[-1,1]` |
| random repeats | 10 |
| n-bit | `B = 1,2,3,4,5,6,7,8` |

已完成训练模型包括：

| run name | 说明 |
|---|---|
| Clean | clean 训练 ResNet20 |
| NAT-FT grid | 早期 alpha-grid / 小范围 NAT fine-tuning checkpoint，仅用于已完成模型对比 |
| NAT-scratch grid | 早期 alpha-grid / 小范围 NAT scratch checkpoint，仅用于已完成模型对比 |
| RobustSel grid | 早期 robust-selection checkpoint，仅用于已完成模型对比 |
| Per-occ NAT best | 本次严格 per-occurrence scratch-full 训练中按 `val_random_accuracy` 选择的 checkpoint |
| Per-occ NAT last | 本次严格 per-occurrence scratch-full 训练结束 checkpoint，作为从零训练主结果 |

注意：`grid` checkpoint 的训练阶段不是本次严格 `[-1,1]` per-occurrence 协议；它们只用于“已有完成训练模型”的推理对比。所有推理评估均严格使用 per-call random alpha in `[-1,1]`。

## 4. 已完成训练模型的扰动推理对比

结果目录：

```text
outputs/paper_fpc_summary/tables/cifar_model_comparison.csv
outputs/paper_fpc_summary/figures/cifar_model_comparison.png
```

下图是 CIFAR-10 主结果的论文级摘要：同一组 checkpoint 下，裸随机非线性推理会降到接近随机分类，而 FPC `B=5` 和 `B=6` 能把精度恢复到 clean 附近。该图突出的是部署结论：FPC 的收益不是来自训练集或 checkpoint 偶然性，而是来自固定点 bit-plane 在官方非线性函数下的端点不变量。

![统一图 4：FPC 恢复随机非线性推理精度](../outputs/unified_paper/figures/fig04_fpc_main_result.png)

![CIFAR model comparison](../outputs/paper_fpc_summary/figures/cifar_model_comparison.png)

| run_name | clean_acc | random_mean_acc | random_worst_acc | fpc_b4_acc | fpc_b5_acc | fpc_b6_acc | fpc_b8_acc | b5_recovery_ratio | b6_recovery_ratio | b8_recovery_ratio |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.8938 | 0.1337 | 0.1184 | 0.8577 | 0.8877 | 0.8935 | 0.8936 | 0.9920 | 0.9996 | 0.9997 |
| NAT-FT grid | 0.8940 | 0.1300 | 0.1151 | 0.8617 | 0.8886 | 0.8922 | 0.8948 | 0.9929 | 0.9976 | 1.0010 |
| NAT-scratch grid | 0.8964 | 0.1355 | 0.1223 | 0.8648 | 0.8889 | 0.8941 | 0.8959 | 0.9901 | 0.9970 | 0.9993 |
| RobustSel grid | 0.8833 | 0.1317 | 0.1184 | 0.8415 | 0.8730 | 0.8819 | 0.8822 | 0.9863 | 0.9981 | 0.9985 |
| Per-occ NAT best | 0.7737 | 0.1622 | 0.1358 | 0.7567 | 0.7698 | 0.7743 | 0.7745 | 0.9936 | 1.0010 | 1.0013 |
| Per-occ NAT last | 0.8659 | 0.1510 | 0.1260 | 0.8275 | 0.8606 | 0.8634 | 0.8665 | 0.9926 | 0.9965 | 1.0008 |

`recovery_ratio` 定义为：

```text
(FPC_B_accuracy - random_mean_accuracy) / (clean_accuracy - random_mean_accuracy)
```

主要结论：

1. 裸随机扰动推理会将 CIFAR-10 ResNet20 从约 `88%-90%` 精度打到 `13%-16%`，接近随机分类。
2. FPC 在 `B=4` 已恢复约 `94%-97%` 的 clean-random 精度损失。
3. FPC 在 `B=5` 时恢复约 `98.6%-99.4%` 的损失。
4. FPC 在 `B=6` 到 `B=8` 时基本贴近 clean，部分结果略高于 clean 是测试随机性与量化正则效应共同造成的微小波动。
5. 结论对 clean checkpoint、历史 NAT checkpoint、严格 per-occurrence scratch-full checkpoint 均成立。

## 5. 从零开始完整失真扰动训练

训练命令核心配置：

```text
script: scripts/train_cifar_per_occ_nat.py
output: outputs/paper_cifar_per_occ_nat_scratch_full_e80
epochs: 80
optimizer: SGD + Nesterov
lr: 0.05, cosine annealing
batch size: 512
loss: CE(clean) + CE(random-field)
random views: 1
alpha: uniform random in [-1,1]
train split: full 45k CIFAR-10 training subset
validation split: 5k CIFAR-10 validation subset
```

训练曲线：

![Per-occurrence NAT scratch training curve](../outputs/paper_fpc_summary/figures/cifar_per_occ_nat_scratch_training_curve.png)

最后 10 个 epoch：

| epoch | train_clean_accuracy | val_clean_accuracy | val_random_accuracy | lr |
|---:|---:|---:|---:|---:|
| 71 | 0.9698 | 0.8138 | 0.1674 | 0.0015 |
| 72 | 0.9723 | 0.8702 | 0.1858 | 0.0012 |
| 73 | 0.9709 | 0.8554 | 0.1660 | 0.0009 |
| 74 | 0.9731 | 0.8524 | 0.1306 | 0.0007 |
| 75 | 0.9735 | 0.8738 | 0.1166 | 0.0005 |
| 76 | 0.9737 | 0.8846 | 0.1092 | 0.0003 |
| 77 | 0.9741 | 0.8816 | 0.1470 | 0.0002 |
| 78 | 0.9741 | 0.8728 | 0.1324 | 0.0001 |
| 79 | 0.9746 | 0.8712 | 0.1102 | 0.0000 |
| 80 | 0.9741 | 0.8708 | 0.1600 | 0.0000 |

训练观察：

| 指标 | 数值 |
|---|---:|
| 最高 `val_clean_accuracy` | 0.8860, epoch 70 |
| 最高 `val_random_accuracy` | 0.2176, epoch 21 |
| `last` checkpoint test clean accuracy | 0.8659 |
| `last` checkpoint random-field mean accuracy | 0.1510 |
| `last` checkpoint random-field worst accuracy | 0.1260 |
| `last` checkpoint FPC B=6 accuracy | 0.8634 |
| `last` checkpoint FPC B=8 accuracy | 0.8665 |

该实验说明：即使从零开始在严格 per-occurrence 随机 alpha 下训练，裸随机扰动推理仍然不稳定；FPC 作为推理编码机制可以把性能恢复到 clean 附近。

## 6. n-bit 消融实验

结果目录：

```text
outputs/paper_fpc_summary/tables/cifar_nbit_long.csv
outputs/paper_fpc_summary/figures/cifar_nbit_ablation.png
```

下图把 `B=1` 到 `B=8` 的 FPC 精度变化压缩成一条工程曲线。核心现象是：`B=1-2` 信息不足，`B=3` 开始出现语义恢复，`B=4` 进入可用区间，`B=5-6` 基本接近 clean。后续效率分析中的推荐配置正是从这条精度曲线推导出来的。

![统一图 5：FPC bit 位宽消融](../outputs/unified_paper/figures/fig05_nbit_ablation.png)

![CIFAR n-bit ablation](../outputs/paper_fpc_summary/figures/cifar_nbit_ablation.png)

| bits | Clean | NAT-FT grid | NAT-scratch grid | Per-occ NAT best | Per-occ NAT last | RobustSel grid |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1006 | 0.1045 | 0.0997 | 0.1000 | 0.1031 | 0.1028 |
| 2 | 0.1970 | 0.1925 | 0.2187 | 0.3643 | 0.3641 | 0.2259 |
| 3 | 0.7058 | 0.7104 | 0.7445 | 0.6399 | 0.7129 | 0.7028 |
| 4 | 0.8577 | 0.8617 | 0.8648 | 0.7567 | 0.8275 | 0.8415 |
| 5 | 0.8877 | 0.8886 | 0.8889 | 0.7698 | 0.8606 | 0.8730 |
| 6 | 0.8935 | 0.8922 | 0.8941 | 0.7743 | 0.8634 | 0.8819 |
| 7 | 0.8930 | 0.8935 | 0.8959 | 0.7753 | 0.8659 | 0.8820 |
| 8 | 0.8936 | 0.8948 | 0.8959 | 0.7745 | 0.8665 | 0.8822 |

消融结论：

1. `B=1` 基本不可用，因为只有符号/零信息，分类精度接近随机。
2. `B=2` 开始恢复部分语义信息，但仍明显不足。
3. `B=3` 出现跃迁，多个 checkpoint 已恢复到 `64%-74%`。
4. `B=4` 达到较高可用性，clean-trained checkpoint 为 `85.77%`。
5. `B=5` 已接近 clean，clean-trained checkpoint 为 `88.77%`。
6. `B=6` 到 `B=8` 基本达到 clean 上限，是推荐的工程配置区间。

## 7. VoxForge 小规模跨任务验证

VoxForge 本地缓存规模较小：

```text
train = 120
val = 30
test = 30
```

因此该部分仅作为跨模态 sanity check，不作为主论文级统计证据。

结果目录：

```text
outputs/paper_fpc_summary/tables/voxforge_model_comparison.csv
outputs/paper_fpc_summary/figures/voxforge_nbit_ablation.png
```

下图给出 VoxForge 小规模跨任务验证。由于 ASR 子集很小，该实验不作为主统计证据；但结果方向与 CIFAR-10 一致：裸随机非线性会抬高 CER，而 FPC `B=3` 或 `B=4` 已可回到对应 clean CER 附近。这说明 FPC 针对的是矩阵算子输入编码问题，而不是 CIFAR-10/ResNet20 的特定经验技巧。

![统一图 8：VoxForge FPC 跨任务验证](../outputs/unified_paper/figures/fig08_voxforge_fpc.png)

![VoxForge n-bit ablation](../outputs/paper_fpc_summary/figures/voxforge_nbit_ablation.png)

| run_name | clean_cer | random_mean_cer | random_worst_cer | fpc_b3_cer | fpc_b4_cer | fpc_b8_cer |
|:---|---:|---:|---:|---:|---:|---:|
| Clean | 0.7945 | 0.8738 | 0.9499 | 0.7878 | 0.7945 | 0.7945 |
| NAT-FT grid | 0.7694 | 0.8561 | 0.9499 | 0.7602 | 0.7678 | 0.7686 |
| NAT-scratch grid | 0.8755 | 0.9228 | 0.9916 | 0.8663 | 0.8747 | 0.8739 |

VoxForge 结果与 CIFAR-10 趋势一致：裸随机扰动使 CER 升高，而 FPC 在 `B=3` 以上即可回到 clean CER 附近。

## 8. 输出文件索引

主要表格：

```text
outputs/paper_fpc_summary/tables/cifar_model_comparison.csv
outputs/paper_fpc_summary/tables/cifar_nbit_long.csv
outputs/paper_fpc_summary/tables/voxforge_model_comparison.csv
outputs/paper_fpc_summary/tables/voxforge_nbit_long.csv
outputs/paper_cifar_per_occ_nat_scratch_full_e80/tables/cifar_training_logs.csv
outputs/paper_cifar_per_occ_nat_scratch_full_e80/tables/cifar_method_summary.csv
```

主要图：

```text
outputs/paper_fpc_summary/figures/cifar_model_comparison.png
outputs/paper_fpc_summary/figures/cifar_nbit_ablation.png
outputs/paper_fpc_summary/figures/cifar_per_occ_nat_scratch_training_curve.png
outputs/paper_fpc_summary/figures/voxforge_nbit_ablation.png
outputs/unified_paper/figures/fig04_fpc_main_result.png
outputs/unified_paper/figures/fig05_nbit_ablation.png
outputs/unified_paper/figures/fig06_fp32_counterexample.png
outputs/unified_paper/figures/fig07_efficiency_tradeoff.png
outputs/unified_paper/figures/fig08_voxforge_fpc.png
```

主要 checkpoint：

```text
outputs/paper_cifar_per_occ_nat_scratch_full_e80/checkpoints/cifar/cifar_per_occ_nat_scratch_full_e80_best.pt
outputs/paper_cifar_per_occ_nat_scratch_full_e80/checkpoints/cifar/cifar_per_occ_nat_scratch_full_e80_last.pt
```

## 9. 总结

FPC 在严格 `alpha in [-1,1]`、每算子每次随机出现、推理时不知道 alpha 的条件下有效。它不依赖 alpha 的均值、分布、方向，也不修改官方非线性函数。实验显示：

1. 裸随机非线性会让 CIFAR-10 ResNet20 从约 `89%` 精度跌到约 `13%-15%`。
2. 已完成训练模型上，FPC `B=5` 可恢复约 `99%` 的 clean-random 精度损失，`B=6` 到 `B=8` 基本贴近 clean。
3. 从零开始完整 per-occurrence 随机失真训练后，裸随机扰动仍只有 `15.10%` mean accuracy，而 FPC `B=8` 达到 `86.65%`，接近该 checkpoint 的 clean `86.59%`。
4. n-bit 消融表明 `B=4` 是可用起点，`B=5` 是较优折中，`B=6-8` 是高精度部署区间。
5. VoxForge 小规模验证呈现相同趋势，说明 FPC 不绑定图像分类模型结构。

因此，FPC 解决的是“未知、随机、逐算子非线性 alpha 导致的推理不稳定”问题，效果是将不可控模拟非线性误差转化为可控 bit 精度误差，并在 5-8 bit 区间保持接近 clean 的推理性能。

## 10. FP32 IEEE754 二进制 bit 串实验

### 10.1 实验定义

用户进一步要求验证“32 位浮点数 FP32 标准 IEEE754 二进制 bit 串”的方案。这里需要先澄清一个重要事实：

```text
IEEE754 的 1-bit sign、8-bit exponent、23-bit mantissa 不是固定点编码。
原始 32 个存储 bit 不能直接作为 32 个线性加权平面输入矩阵乘法。
```

原因是指数位不是数值贡献位，而是控制尾数整体缩放的元数据；符号位也不是可独立线性相加的数值项。因此，若直接把 32 个存储 bit 当作整数 bit-plane 做矩阵累加，得到的是 FP32 存储字的整数表示，不是原始实数张量，矩阵计算没有数学意义。

为保证实验可执行且忠实于 IEEE754 标准，本报告采用如下“IEEE754 数值贡献平面”：

```text
FP32 = (-1)^sign * 2^(exponent - 127) * (1 + mantissa / 2^23)
```

对 normal number，将其拆成：

```text
implicit leading 1 plane + 23 mantissa contribution planes
```

共 24 个可加数值平面。每个平面都由 IEEE754 bit 字段解析而来，并满足：

```text
sum(24 value planes) == original FP32 activation
```

实现文件：

```text
src/nonlinear.py::FP32IEEEBitSerialInjector
scripts/evaluate_cifar_fp32_ieee_bitserial.py
scripts/run_fp32_ieee_cifar_evals.ps1
scripts/aggregate_fp32_ieee_results.py
```

关键工程处理：Conv/Linear 自带 bias。由于 IEEE754 数值平面是直接相加的，每个 partial forward 会重复加入 bias，因此实现中显式扣除 `(plane_count - 1) * bias`。无硬件非线性时，单元测试验证 Conv/Linear 输出可与原始 FP32 forward 完全一致。

### 10.2 FP32 IEEE754 实验结果

结果目录：

```text
outputs/paper_fp32_ieee_summary/tables/cifar_fp32_ieee_model_comparison.csv
outputs/paper_fp32_ieee_summary/figures/cifar_fp32_ieee_model_comparison.png
```

下图把 FP32 IEEE754 value-plane 方案和 FPC 放在同一基准下比较。左侧显示：如果不经过硬件非线性，IEEE754 数值平面分解可以精确还原 clean forward；但一旦每个 value-plane 仍进入随机非线性硬件，精度只恢复到约 `20%-26%`。右侧显示 FP32 IEEE754 的 recovery ratio 远低于 FPC `B=5/6`，因此它验证了“存储 bit 串不等于鲁棒计算编码”这一反例。

![统一图 6：FP32 IEEE754 value-plane 反例](../outputs/unified_paper/figures/fig06_fp32_counterexample.png)

![FP32 IEEE754 bit-string comparison](../outputs/paper_fp32_ieee_summary/figures/cifar_fp32_ieee_model_comparison.png)

| run_name | clean_acc | random_single_acc | fp32_ieee_no_hw_acc | fp32_ieee_random_hw_acc | fp32_recovery_ratio | fp32_clean_gap |
|:---|---:|---:|---:|---:|---:|---:|
| Clean | 0.8938 | 0.1337 | 0.8939 | 0.2072 | 0.0967 | 0.6866 |
| NAT-FT grid | 0.8940 | 0.1300 | 0.8941 | 0.2060 | 0.0994 | 0.6880 |
| NAT-scratch grid | 0.8965 | 0.1355 | 0.8965 | 0.2257 | 0.1185 | 0.6708 |
| RobustSel grid | 0.8833 | 0.1317 | 0.8833 | 0.2008 | 0.0920 | 0.6825 |
| Per-occ NAT best | 0.7737 | 0.1622 | 0.7737 | 0.2621 | 0.1633 | 0.5116 |
| Per-occ NAT last | 0.8658 | 0.1510 | 0.8658 | 0.2399 | 0.1244 | 0.6259 |

### 10.3 FP32 IEEE754 实验结论

1. `fp32_ieee_no_hw_acc` 与 `clean_acc` 基本完全一致，说明 IEEE754 value-plane 分解本身是正确的。
2. 一旦每个 FP32 value-plane 进入硬件前仍遭遇随机非线性，`fp32_ieee_random_hw_acc` 只有 `20%-26%`，远低于 clean。
3. FP32 IEEE754 bit 串没有 FPC 的固定点不变量。每个 value-plane 的归一化值不是 `{ -1, 0, +1 }`，因此官方非线性会继续改变它。
4. FP32 IEEE754 方案能恢复的 clean-random 精度损失只有约 `9%-16%`；FPC `B=5` 已恢复约 `99%`，`B=6-8` 基本完全恢复。
5. FP32 IEEE754 value-plane 需要 24 个矩阵算子 partial forward，计算代价明显高于 FPC 的 5-8 个 partial forward，但鲁棒效果显著更差。

因此，标准 FP32 IEEE754 二进制 bit 串是一种存储格式，不是面向随机模拟非线性的鲁棒计算编码；它不能替代 FPC。

## 11. bit 位数对计算效率的影响

### 11.1 分析口径

本节同时给出两类效率指标：

1. **软件仿真实测时间**：使用 Python hook 在 GPU 上模拟 bit-serial forward，包含 hook、循环、张量调度等软件开销，只用于观察相对趋势。
2. **硬件解析模型**：统计每个矩阵算子需要被调用的次数、串行周期数、激活输入 bit 流量。这更接近存算芯片部署分析。

实验脚本：

```text
scripts/benchmark_bitserial_efficiency.py
```

结果目录：

```text
outputs/paper_efficiency_cifar/tables/cifar_efficiency_summary.csv
outputs/paper_efficiency_cifar/figures/cifar_efficiency_time_vs_bits.png
outputs/paper_efficiency_cifar/figures/cifar_efficiency_activation_bits.png
```

下图给出 FPC 位宽的精度-效率权衡。左侧把 bit 位宽、精度和激活输入流量放在一起，显示 `B=5-6` 是主要拐点；右侧把 clean、FPC 和 FP32 IEEE754 的矩阵调用次数并列，显示 FP32 IEEE754 value-plane 需要 24 个 partial forward，却不能带来相应鲁棒性。综合看，FPC 的工程价值来自“少量 bit-plane + 端点不变量”，而不是单纯把实数拆成更多二进制平面。

![统一图 7：FPC 精度、带宽和串行计算代价](../outputs/unified_paper/figures/fig07_efficiency_tradeoff.png)

![Efficiency time vs bits](../outputs/paper_efficiency_cifar/figures/cifar_efficiency_time_vs_bits.png)

![Efficiency activation bits](../outputs/paper_efficiency_cifar/figures/cifar_efficiency_activation_bits.png)

### 11.2 实测与解析表

| method | bits | mean_elapsed_s | software_time_vs_clean | matrix_passes_per_operator | activation_stream_bits | activation_stream_vs_fp32 | ideal_sequential_throughput_vs_clean | mean_accuracy |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0 | 0.3238 | 1.0000 | 1 | 32 | 1.0000 | 1.0000 | 0.8940 |
| random_single | 0 | 0.3284 | 1.0141 | 1 | 32 | 1.0000 | 1.0000 | 0.1224 |
| fpc | 1 | 0.5458 | 1.6856 | 1 | 1 | 0.0312 | 1.0000 | 0.0923 |
| fpc | 2 | 0.7014 | 2.1660 | 2 | 2 | 0.0625 | 0.5000 | 0.2012 |
| fpc | 3 | 1.0245 | 3.1636 | 3 | 3 | 0.0938 | 0.3333 | 0.7082 |
| fpc | 4 | 1.1496 | 3.5500 | 4 | 4 | 0.1250 | 0.2500 | 0.8547 |
| fpc | 5 | 1.3652 | 4.2159 | 5 | 5 | 0.1562 | 0.2000 | 0.8843 |
| fpc | 6 | 1.5913 | 4.9140 | 6 | 6 | 0.1875 | 0.1667 | 0.8950 |
| fpc | 7 | 1.8213 | 5.6242 | 7 | 7 | 0.2188 | 0.1429 | 0.8936 |
| fpc | 8 | 2.0555 | 6.3475 | 8 | 8 | 0.2500 | 0.1250 | 0.8929 |
| fp32_ieee | 32 | 5.2416 | 16.1865 | 24 | 32 | 1.0000 | 0.0417 | 0.2140 |

### 11.3 效率结论

FPC 的 bit 位数 `B` 带来两个相反方向的工程影响：

```text
矩阵阵列调用次数 / 串行周期数 约为 B
激活输入 bit 流量约为 B / 32 个 FP32 激活流量
```

因此：

1. `B` 越大，量化误差越小，精度越接近 clean。
2. `B` 越大，bit-serial 计算周期越多，若硬件完全串行执行，理想吞吐约按 `1/B` 下降。
3. `B` 越小，激活输入带宽越省，但精度损失明显；`B=1-2` 不可用，`B=3` 仍不足够稳定。
4. `B=4` 是可用起点，激活 bit 流量只有 FP32 的 `12.5%`，但 clean-trained 精度为 `85.47%`，仍有约 `3.9%` clean gap。
5. `B=5` 是较好的折中点，激活 bit 流量为 FP32 的 `15.625%`，精度达到 `88.43%` 左右，已接近 clean。
6. `B=6` 是推荐高精度部署点，激活 bit 流量为 FP32 的 `18.75%`，精度达到 `89.50%`，基本等于 clean。
7. `B=7-8` 继续增加计算周期，但精度提升很小；除非系统对精度极端敏感，否则收益递减。
8. FP32 IEEE754 value-plane 需要 24 个矩阵 partial pass，解析吞吐只有 clean 的 `1/24`，激活 bit 流量仍是 FP32 的 `100%`，同时随机非线性下精度仅约 `21.4%`，因此不是有效方案。

综合精度和效率，建议：

| 场景 | 推荐 bit | 理由 |
|---|---:|---|
| 极低功耗探索 | B=4 | 4 个串行周期，激活流量 12.5%，精度已有明显恢复 |
| 默认部署 | B=5 | 5 个串行周期，精度接近 clean，效率较高 |
| 高精度部署 | B=6 | 基本 clean 级精度，仍只需 FP32 激活流量的 18.75% |
| 极限精度 | B=7-8 | 收益递减，只在精度优先时使用 |
