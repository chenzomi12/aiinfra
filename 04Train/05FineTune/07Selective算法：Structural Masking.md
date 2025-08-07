<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# Selective算法：Unstructural Masking

> Author by: 李亚鹏

## S-Diff-Pruning

**S-Diff-Pruning**（Structural Diff-Pruning ）：基于上一小节的基础方法U-Diff-Pruning（ Unstructural Diff-Pruning ）拓展而来。

U-Diff-Pruning是独立地决定每个参数是否被剪枝。但模型中的参数通常是有结构的（例如，一个权重矩阵或一个偏置向量）。结构化剪枝的思想是，要剪就一起剪掉整个结构。

这鼓励模型将整个参数块（如某个注意力头的权重矩阵）完整保留或完整剪枝，实验证明这种结构化的方法性能更好。

S-Diff-Pruning将参数根据结构进行分组，例如一个层的所有权重为一组。除了已有的掩码$z_τ$，额外引入一个组内共享掩码$z_τ^g$。

一个参数最终被保留，当且仅当它自身的掩码$z_(τ,i)$和它所属的组的共享掩码$z_τ^g$都为“开启”状态。正则项修正如下：
$$
\mathbb{E}\left[\left\|\boldsymbol{\delta}_\tau\right\|_0\right]=\sum_{j=1}^G\sum_{i\in g(j)}\mathbb{E}\left[\mathbb{1}\{\mathbf{z}_{\tau,i}\cdot\mathbf{z}_\tau^g>0\}\right]
$$

## SPT

Adapter、LoRA等方法通常在所有下游任务中都对模型相同的位置（例如，所有Transformer块的self-attention部分）插入可训练参数，而忽略了不同任务可能需要调整模型不同部分这一事实。

**SPT**，Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning：不应对所有任务都调整相同的参数，而应该根据具体任务，自适应地选择在哪里以及如何分配有限的可训练参数预算。

SPT的核心思想是：首先识别出对于特定下游任务最重要的（即最“敏感”的）参数，然后根据一个固定的参数预算，智能地分配微调资源。

理想情况下，一个参数$w$的敏感度$s$可以定义为：只微调这一个参数能给任务损失(loss)带来多大的降低。但为模型中所有参数进行这种完整的计算，成本是巨大的。

为此，SPT使用一阶泰勒展开来近似loss的变化，即参数$w_n$的敏感度$s_n=g_n△w_n$，其中$g_n$为梯度。

进一步的，将参数更新 $△w_n$近似为单步梯度下降的结果，即$△w_n=ε g_n$，其中ε为学习率。

由此，$s_n=g_n^2$ 。由于对于所有参数来说学习率一致，因此在比较中忽略学习率。

 因此，一个参数的敏感度可以直接用它在任务数据上的梯度的平方来衡量。梯度越大，说明该参数对当前任务的损失影响越大，调整它可能带来的收益也越大，因此它就越“敏感”。

实际操作中，选取部分训练样本，进行参数梯度的计算，从而近似得到所有参数的敏感度。

对于可训练参数预算的分配，SPT综合考虑了非结构化与结构化微调策略。

具体来说，首先依据敏感度得分 $S$，选出 $Top-τ$ 的高敏感参数。

结构化策略：以矩阵为微调单位，若矩阵W中的高敏感参数数量大于阈值$σ_{opt}$，则对整个矩阵进行微调，具体方法为使用 LoRA或 Adapter，引入小的、新的可训练模块。（如$W_{up}$和$W_{down}$）

非结构化策略：若矩阵W中的高敏感参数数量小于阈值$σ_opt$，只更新掩码$M$标记的高敏感参数，其中$g_W$为$W$的梯度。
$$
\boldsymbol{W}^{\prime}=\left\{\begin{array}{ll}\boldsymbol{W}+\boldsymbol{W}_\mathrm{down}\boldsymbol{W}_\mathrm{up}&\quad\mathrm{if}\quad\sum_{j=0}^{d_\mathrm{in}\times d_\mathrm{out}}\boldsymbol{M}^j\geq\sigma_\mathrm{opt}\\\boldsymbol{W}-\epsilon\boldsymbol{g}_W\odot\boldsymbol{M}&\quad\mathrm{otherwise}\end{array}\right.
$$

SPT自适应地结合了结构化和非结构化调优粒度，对应实现了更强的表示能力和更高的灵活性，并将训练参数控制在了预算内。

## Reference

> 1. Diff-Pruning：Parameter-Efficient Transfer Learning with Diff Pruning， https://arxiv.org/abs/2012.07463
> 2. SPT：Sensitivity-aware visual parameter-efficient fine-tuning，https://arxiv.org/abs/2303.08566
