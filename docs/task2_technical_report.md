# 任务二技术报告：非线性感知训练（Nonlinearity-Aware Training）

> **重要修正：任务二主训练协议不是“每个 batch 一个全局 alpha”。**
> 最新确认后，训练和评估阶段都应在每个矩阵算子输入端、每次 forward 调用独立采样
> `alpha_{l,c} ~ Uniform(-1,1)`。因此旧版 `alpha_scheduler` 的全局采样结果只作为
> 历史基线或 ablation；主实现应使用 `RandomAlphaNonlinearityInjector(-1,1)`。
> 统一定义见
> [per_occurrence_random_alpha_protocol.md](per_occurrence_random_alpha_protocol.md)。

## 1. 引言

任务二面向存算一体芯片中模拟域乘累加的输入相关非线性误差，研究在训练阶段显式加入非线性映射模型后，神经网络对推理非线性扰动的适应能力。与任务一只在推理阶段注入非线性误差不同，任务二将非线性映射纳入训练闭环，重点比较三类策略：

1. **Clean Training**：训练阶段不加入非线性，用作 clean baseline。
2. **NAT Fine-tuning**：先得到 clean checkpoint，再在训练阶段加入非线性扰动继续微调。
3. **NAT From Scratch**：模型从随机初始化开始就在训练阶段加入非线性扰动。

本轮实验覆盖两个任务：

| 数据集 | 任务 | 模型 | 主要指标 |
|---|---|---|---|
| CIFAR-10 | 图像分类 | ResNet20 | Top-1 Accuracy / Robust Accuracy |
| VoxForge | 语音识别 | CRNN-CTC | WER / CER / CTC Loss |

需要说明的是，当前报告对应“首轮完整闭环实验”：CIFAR-10 使用 60 epoch clean/scratch 和 20 epoch fine-tuning；VoxForge 使用小规模固定子集 `120/30/30` 完成 ASR 训练闭环。代码已经支持将 CIFAR-10 扩展到 160/200 epoch，将 VoxForge 扩展到更大子集和 50/100 epoch。

## 2. 技术原理

### 2.1 非线性映射模型

训练和测试阶段均采用赛题给定的三次多项式非线性：

```text
u = x / max(|x|)
f(u) = alpha * u^3 + (1 - alpha) * u
x' = max(|x|) * f(u)
```

其中 `alpha=0` 表示理想线性硬件；`|alpha|` 越大，非线性越强；`alpha` 的正负决定正负输入区域的压缩或放大方向。

实现中对数值稳定性做了处理：

```python
max_val = x.detach().abs().amax().clamp_min(1e-12)
u = x / max_val
y = alpha * (u ** 3) + (1 - alpha) * u
x_nl = y * max_val
```

`max_val` 使用 `detach()`，避免训练过程通过归一化尺度传播不必要的梯度；`clamp_min` 防止全零激活导致除零。

### 2.2 非线性注入位置

非线性误差被注入到矩阵计算算子的输入端，与存算一体芯片中“输入激活值进入模拟乘累加阵列后发生非理想响应”的物理含义一致。当前实现覆盖：

```text
Conv1d
Conv2d
ConvTranspose1d
ConvTranspose2d
Linear
```

CIFAR-10 中主要影响卷积层和全连接分类头；VoxForge 的 CRNN-CTC 中主要影响 log-mel 后端卷积前端和 CTC 线性输出头。

### 2.3 Alpha 采样策略

任务二实现了统一的 alpha scheduler：

| 调度器 | 形式 | 用途 |
|---|---|---|
| NoAlphaScheduler | `alpha=0` | clean baseline |
| FixedAlphaScheduler | 固定 alpha | 单硬件点适配 |
| UniformAlphaScheduler | `alpha ~ Uniform(low, high)` | 随机非线性训练 |
| CurriculumAlphaScheduler | 扰动范围随 epoch 逐步增大 | 改善从头训练稳定性 |

旧版首轮闭环实验曾使用较窄的全局随机 alpha 范围：

```text
CIFAR-10: alpha_train ~ Uniform(-0.2, 0.2)
VoxForge: alpha_train ~ Uniform(-0.1, 0.1)
```

该设置现在只能作为稳定性预实验或历史 ablation。官方主协议必须改为：

```text
对每个目标算子输入端、每次 forward 调用：
alpha_{l,c} ~ Uniform(-1, 1)
```

VoxForge 使用更保守范围的理由可以保留为工程预热策略，但最终主结果不能缩小 alpha 范围，也不能把一个 alpha 共享给整网。

## 3. 工程实现

### 3.1 新增与复用代码

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

### 3.2 CIFAR-10 训练实现

CIFAR-10 主模型使用 ResNet20。训练阶段每个 batch 根据 alpha scheduler 采样一个 alpha，并通过 `NonlinearityInjector` 将非线性映射注入到目标算子的输入端。

除常规按 clean validation accuracy 选择 checkpoint 外，本轮还增加了 robust validation 选择方式：

```text
selection_score = mean(Acc_val(alpha=-0.2), Acc_val(alpha=0), Acc_val(alpha=0.2))
```

该设置用于验证：NAT 训练是否需要使用硬件鲁棒性指标，而不是只用 clean validation accuracy 选模型。

### 3.3 VoxForge 训练实现

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

## 4. 实验配置

### 4.0 Alpha 取值协议修正

根据赛题完整非线性强度范围，最终官方口径应采用：

```text
alpha ~ Uniform(-1, 1)
```

训练阶段的随机非线性误差插入应从 `[-1,1]` 采样，而不是只从 `[-0.2,0.2]` 或 `[-0.1,0.1]` 采样。本文早期任务二实验为了先验证 NAT fine-tuning 与 from scratch 的训练稳定性，使用了较窄 alpha 范围；这些结果可作为弱/中等非线性下的补充分析，但不能作为最终官方范围的唯一结论。

代码已修正：

| 文件 | 修正 |
|---|---|
| `src/alpha_scheduler.py` | 旧版全局 alpha scheduler，仅作为历史 ablation |
| `scripts/train_cifar_task2.py` | 旧版全局 alpha NAT，不作为最终主协议入口 |
| `scripts/train_voxforge_task2.py` | 旧版全局 alpha NAT，不作为最终主协议入口 |
| `scripts/train_cifar_per_occ_nat.py` | CIFAR-10 per-occurrence random alpha NAT 主训练入口 |
| `scripts/train_voxforge_per_occ_nat.py` | VoxForge per-occurrence random alpha NAT 主训练入口 |

后续正式 CIFAR-10 结果应按以下方式重跑：

```powershell
python scripts/train_cifar_per_occ_nat.py --alpha-low -1.0 --alpha-high 1.0 ...
```

VoxForge 已新增 `scripts/train_voxforge_per_occ_nat.py`，正式主结果应使用该脚本，而不是旧版全局 alpha scheduler。

### 4.1 CIFAR-10 旧版首轮实验配置

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

### 4.2 VoxForge 旧版首轮实验配置

| 方法 | 初始化 | 训练 alpha | Epoch | 子集规模 |
|---|---|---|---:|---|
| Clean | 随机初始化 | 无 | 30 | `120/30/30` |
| NAT Fine-tuning | Clean best checkpoint | `Uniform(-0.1, 0.1)` | 10 | `120/30/30` |
| NAT From Scratch | 随机初始化 | `Uniform(-0.1, 0.1)` | 30 | `120/30/30` |

VoxForge 测试 alpha 网格：

```text
[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
```

## 5. CIFAR-10 实验结果

### 5.1 收敛曲线

![CIFAR-10 Training Loss](../outputs/task2/figures/cifar_train_loss.png)

![CIFAR-10 Validation Accuracy](../outputs/task2/figures/cifar_val_accuracy.png)

从训练曲线看，NAT fine-tuning 由于从 clean checkpoint 初始化，起始 loss 明显低于从头训练，20 epoch 内即可保持接近 clean baseline 的验证精度。NAT from scratch 在 60 epoch 内也能达到接近 clean 的验证精度，说明将非线性注入训练并不会阻止模型学习 clean 判别特征。

### 5.2 Alpha-Accuracy 曲线

![CIFAR-10 Accuracy under Inference Nonlinearity](../outputs/task2/figures/cifar_alpha_accuracy.png)

![CIFAR-10 Accuracy Drop](../outputs/task2/figures/cifar_accuracy_drop.png)

CIFAR-10 的主要现象是：所有方法在强非线性 `|alpha|>=0.3` 下仍然明显退化；NAT 的收益主要集中在训练范围附近，尤其是 `alpha=-0.2` 和 `alpha=-0.1`。

### 5.3 CIFAR-10 汇总表

| 方法 | Epoch | Clean Acc | Avg Robust Acc | Worst Acc | Max Drop |
|:---|---:|---:|---:|---:|---:|
| Clean | 60 | 0.8939 | 0.3141 | 0.1000 | 0.7939 |
| NAT FT | 20 | 0.8941 | 0.3122 | 0.1079 | 0.7862 |
| NAT Scratch | 60 | 0.8965 | 0.2812 | 0.0935 | 0.8030 |
| NAT FT RobustSel | 20 | 0.8832 | 0.3175 | 0.1000 | 0.7832 |

![CIFAR-10 Summary](../outputs/task2/figures/cifar_method_summary.png)

结论：

1. **普通 NAT fine-tuning** 保持了 clean accuracy，但平均鲁棒精度没有明显提升，说明只按 clean validation accuracy 选 checkpoint 会偏向 clean 性能。
2. **NAT FT Robust-selected** 牺牲约 `1.09%` clean accuracy，但平均鲁棒精度从 `0.3122` 提升到 `0.3175`，最大精度下降从 `0.7862` 降到 `0.7832`。
3. **NAT from scratch** 在当前 60 epoch 预算下 clean accuracy 不差，但平均鲁棒精度最低，说明从头鲁棒训练需要更长训练预算或 curriculum alpha。

### 5.4 CIFAR-10 Alpha 逐点数据

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

## 6. VoxForge 实验结果

### 6.1 收敛曲线

![VoxForge Training Loss](../outputs/task2/figures/voxforge_train_loss.png)

![VoxForge Validation WER](../outputs/task2/figures/voxforge_val_wer.png)

VoxForge 的训练损失显示 clean 和 NAT scratch 都能从约 5 降到约 2.08；NAT fine-tuning 从 clean checkpoint 出发，10 epoch 后训练 loss 进一步降到 `1.9903`。但由于训练集和验证集都很小，WER 仍接近 1，因此报告同时使用更细粒度的 CER 作为主要辅助指标。

### 6.2 WER/CER 曲线

![VoxForge WER under Inference Nonlinearity](../outputs/task2/figures/voxforge_alpha_wer.png)

![VoxForge CER under Inference Nonlinearity](../outputs/task2/figures/voxforge_alpha_cer.png)

VoxForge 的 WER 曲线变化范围较窄，主要因为小规模 CTC ASR 还没有充分学到词级边界；CER 更能反映字符级识别改善。NAT fine-tuning 在所有测试 alpha 的 CER 上均优于 clean baseline，说明训练阶段加入非线性扰动能改善语音识别模型的字符级鲁棒性。

### 6.3 VoxForge 汇总表

| 方法 | Epoch | Train/Val/Test | Clean WER | Clean CER | Avg Robust WER | Avg Robust CER | Worst WER | Worst CER |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|
| Clean | 30 | 120/30/30 | 0.9956 | 0.7920 | 0.9884 | 0.7985 | 0.9956 | 0.8246 |
| NAT FT | 10 | 120/30/30 | 0.9913 | 0.7669 | 0.9891 | 0.7817 | 1.0000 | 0.8079 |
| NAT Scratch | 30 | 120/30/30 | 0.9956 | 0.8755 | 0.9934 | 0.8847 | 0.9956 | 0.9165 |

![VoxForge WER Summary](../outputs/task2/figures/voxforge_method_summary.png)

![VoxForge CER Summary](../outputs/task2/figures/voxforge_cer_summary.png)

结论：

1. **NAT fine-tuning** 的 clean CER 从 `0.7920` 降到 `0.7669`，平均鲁棒 CER 从 `0.7985` 降到 `0.7817`，说明在当前小规模 ASR 设置下，fine-tuning 是最有效的策略。
2. **NAT from scratch** 的 CER 明显高于 clean 和 NAT fine-tuning，说明从头训练需要更大数据规模、更长 epoch 或课程式 alpha 调度。
3. WER 接近 1，不能单独作为当前小规模实验的唯一判断依据；CER、CTC loss 和解码样例更能反映模型是否已经学习到字符级声学模式。

### 6.4 VoxForge Alpha 逐点数据

**WER：**

| Alpha | Clean | NAT FT | NAT Scratch |
|---:|---:|---:|---:|
| -0.20 | 0.9782 | 0.9825 | 0.9913 |
| -0.10 | 0.9913 | 0.9869 | 0.9913 |
| -0.05 | 0.9913 | 0.9869 | 0.9956 |
| 0.00 | 0.9956 | 0.9913 | 0.9956 |
| 0.05 | 0.9913 | 0.9913 | 0.9913 |
| 0.10 | 0.9869 | 0.9869 | 0.9956 |
| 0.20 | 0.9913 | 1.0000 | 0.9956 |

**CER：**

| Alpha | Clean | NAT FT | NAT Scratch |
|---:|---:|---:|---:|
| -0.20 | 0.8112 | 0.8053 | 0.9165 |
| -0.10 | 0.7970 | 0.7870 | 0.8922 |
| -0.05 | 0.7970 | 0.7728 | 0.8847 |
| 0.00 | 0.7920 | 0.7669 | 0.8755 |
| 0.05 | 0.7903 | 0.7627 | 0.8747 |
| 0.10 | 0.7711 | 0.7544 | 0.8722 |
| 0.20 | 0.8246 | 0.8079 | 0.8680 |

### 6.5 解码样例

以下样例来自 NAT fine-tuning 模型在 `alpha=0` 下的 greedy CTC 解码。模型仍处于小规模训练状态，但已经能输出与参考文本相关的部分字符片段。

| Reference | Hypothesis |
|---|---|
| de unas parras artificiales cuyas hojas parecían retazos de terciopelo | unse a eaesasas |
| a las que se acogían grupos de personas para embadurnar | a oa as |
| estaban ocupados por señoras | estan ocupors |
| unas líneas negras y oblicuas semejantes a cuerdas | uas ca eas |
| estaban ocupados por señoras | esan ocupors |

## 7. Fine-tuning 与 From Scratch 对比

### 7.1 收敛性

| 数据集 | 对比结论 |
|---|---|
| CIFAR-10 | fine-tuning 从 clean checkpoint 出发，20 epoch 内即可保持 clean accuracy；scratch 需要 60 epoch 才接近 clean baseline。 |
| VoxForge | fine-tuning 的起点 loss 更低，10 epoch 后 CER 优于 clean；scratch 在 30 epoch 小数据预算下仍明显不足。 |

Fine-tuning 的核心优势是部署成本低：已有 clean 模型只需少量硬件感知微调即可适配目标非线性范围。从头训练的优势理论上是端到端适应硬件误差族，但它需要更长训练预算，而且早期训练更容易受非线性扰动影响。

### 7.2 泛化性

CIFAR-10 显示 NAT 的泛化收益并不是自动发生的。普通 NAT fine-tuning 只按 clean validation accuracy 选模型，平均鲁棒精度没有提升；改用 robust validation 后，负 alpha 区域鲁棒性显著改善，但正 alpha 区域下降。这说明 NAT 至少包含三个关键设计点：

1. 训练 alpha 分布是否覆盖目标硬件误差。
2. checkpoint 选择指标是否包含非线性鲁棒性。
3. 是否需要区分正 alpha 和负 alpha 的误差方向。

VoxForge 中 NAT fine-tuning 在 CER 上更稳定地优于 clean baseline，说明 ASR 模型在字符级输出上能从非线性扰动训练中获益。但 WER 仍受小规模训练限制，后续需要扩大数据和 epoch 才能验证词级泛化。

## 8. 局限性与后续改进

当前任务二已经完成 clean、NAT fine-tuning、NAT scratch 的完整训练、评估、图表和报告闭环，但仍有以下限制：

1. CIFAR-10 当前为首轮预算，clean/scratch 使用 60 epoch，低于计划中的 160/200 epoch。
2. VoxForge 当前使用 `120/30/30` 小子集，足以验证训练管线和 CER 趋势，但不足以得到高质量 ASR WER。
3. NAT from scratch 在两个任务上都没有表现出稳定优势，后续应加入 curriculum NAT。
4. CIFAR-10 robust-selected fine-tuning 说明 checkpoint 选择很关键，但三点验证仍偏向负 alpha，后续可采用更多 alpha 点或 clean-robust 加权目标。
5. 当前非线性只注入 Conv/Linear 等显式算子输入，GRU 内部矩阵乘法没有逐门控注入；若要更贴近硬件部署，需要将 RNN/Transformer 内部投影拆成显式 Linear。

建议下一阶段从任务三鲁棒性增强角度继续：

```text
Loss = CE/NLL + lambda * consistency(clean_logits, nonlinear_logits)
alpha curriculum: 0 -> target range
robust checkpoint selection: mean metric over validation alpha grid
direction-aware training: separate positive-alpha and negative-alpha adapters
```

## 9. 结论

任务二实验表明：

1. 将非线性映射加入训练阶段是可行的，CIFAR-10 和 VoxForge 都已完成 clean、NAT fine-tuning、NAT scratch 的训练与评估闭环。
2. NAT fine-tuning 是当前最稳定的部署适配策略：CIFAR-10 中能保持 clean accuracy，VoxForge 中能降低 clean CER 和平均鲁棒 CER。
3. 仅加入 NAT 训练不保证鲁棒性提升；CIFAR-10 的普通 NAT fine-tuning 说明 checkpoint 选择若只看 clean accuracy，可能无法选出最鲁棒模型。
4. Robust validation 能改变模型选择方向：CIFAR-10 中 `NAT FT Robust-selected` 将 `alpha=-0.2` accuracy 从 `0.3271` 提升到 `0.4562`，但也带来 clean 和正 alpha 区域的 trade-off。
5. NAT from scratch 在当前预算下不如 fine-tuning，尤其在 VoxForge 小数据 ASR 中更明显；后续需要更大数据、更长训练和 curriculum alpha。

因此，任务二的核心结论不是“NAT 必然提升所有鲁棒指标”，而是：

```text
非线性感知训练提供了把存算一体芯片非理想输入响应纳入训练闭环的机制。
其最终收益取决于训练 alpha 分布、模型选择准则、任务类型和训练预算。
在实际部署中，fine-tuning + robust validation 是更现实的第一阶段硬件适配方案；
from scratch 更适合在有充足训练预算时构建端到端鲁棒模型。
```

## 10. 产物清单

| 类型 | 路径 |
|---|---|
| 任务二规划 | `docs/task2_experiment_plan.md` |
| 任务二技术报告 | `docs/task2_technical_report.md` |
| CIFAR 训练脚本 | `scripts/train_cifar_task2.py` |
| VoxForge 训练脚本 | `scripts/train_voxforge_task2.py` |
| 任务二绘图脚本 | `scripts/make_task2_figures.py` |
| CIFAR 表格 | `outputs/task2/tables/cifar_training_logs.csv`, `cifar_alpha_eval.csv`, `cifar_method_summary.csv` |
| VoxForge 表格 | `outputs/task2/tables/voxforge_training_logs.csv`, `voxforge_alpha_eval.csv`, `voxforge_method_summary.csv`, `voxforge_decode_examples.csv` |
| 图表目录 | `outputs/task2/figures/` |
| Checkpoint 目录 | `outputs/task2/checkpoints/` |

## 11. 参考文献

1. Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. CIFAR-10 technical report.
2. Graves, A. et al. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.
3. He, K. et al. Deep Residual Learning for Image Recognition.
4. VoxForge speech corpus project.
5. PyTorch and torchaudio official documentation.
