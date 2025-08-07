<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# LoRA变体算法

> Author by: 李亚鹏

## AdaLoRA

LoRA及其一些变体方法，通常为模型中所有需要修改的权重矩阵分配相同数量的可训练参数（例如相同的秩 r）。因此，调优性能不是最优的。

**AdaLoRA**，Adaptive Budget Allocation LoRA：将有限的参数预算（budget）动态地、不均匀地分配给那些更重要的权重矩阵，从而在总参数量不变或更少的情况下，达到更好的性能。

基于奇异值分解（Singular Value Decomposition ，SVD）的参数化 ：与LoRA不同， AdaLoRA 采用了另一种受SVD启发的参数化形式。它将增量$\Delta$分解为左右正交矩阵$P$和$Q$，以及奇异值对角矩阵$\Lambda$ 。
$$
W=W^{(0)}+\Delta=W^{(0)}+P\Lambda Q
$$
SVD参数化有两大优势：

1. 通过控制对角矩阵$Λ$中非零元素的个数，就可以直接、精确地控制增量矩阵$\Delta$的秩。
2. 在 LoRA 中，如果要降低秩，就需要移除 B 的一列和 A 的一行，被剪掉的参数完全丢失。而在 AdaLoRA 中，剪枝操作只作用于奇异值，对应的奇异向量 P 和 Q 的部分依然保留并继续训练，使训练更稳定。

AdaLoRA 设计了一套新的重要性评分机制及剪枝策略。

评分的基本单元是由$P_{k,i}$、$Q_{k,i}$、$\Lambda_{k,i}$构成的三元组。一个三元组的重要性$S_{k,i}$由$\Lambda_{k,i}$本身的重要性、 $P_{k,i}$中所有参数的平均重要性、$Q_{k,i}$中所有参数的平均重要性构成。（k 代表第 k 个权重矩阵，i 代表第 i 个奇异值分量）
$$
S_{k,i}=s(\lambda_{k,i})+\frac{1}{d_1}\sum_{j=1}^{d_1}s(P_{k,ji})+\frac{1}{d_2}\sum_{j=1}^{d_2}s(Q_{k,ij})
$$
单个参数的重要性$s(w)$采用了考虑不确定性的敏感度分数。

敏感度$I(w)$：定义为“权重与梯度的乘积的绝对值”，即$I(w_{ij})=|w_{ij}\nabla_{w_{ij}}\mathcal{L}|$。

AdaLoRA 引入了指数移动平均 (EMA) 来平滑敏感度，并计算了不确定性项$U^{(t)}$，来衡量近期敏感度的波动。

最终的单参数重要性是平滑后的敏感度与不确定性的乘积，这使得那些持续重要且学习尚不稳定的参数获得更高的分数。
$$
s^{(t)}(w_{ij})=\overline{I}^{(t)}(w_{ij})\cdot\overline{U}^{(t)}(w_{ij})
$$
在训练过程中，AdaLoRA 会周期性地根据当前预算$b^{(t)}$  ，计算三元组重要性分数，保留分数最高的$b^{(t)}$个三元组参数（保持它们的奇异值不变） ，并将其余三元组的奇异值设为零。

为了让整个训练过程更稳定、高效而设计的策略。 AdaLoRA定义了总预算（即保留的奇异值总数）$b^{(t)}$如何随训练步数$t$变化。

调度策略分为三个阶段：

1. **热身阶段** (Warm-up)：训练开始时，设置一个比最终目标预算$b^{(T)}$更高的初始预算$b^{(0)}$  。这允许模型在初期探索更广阔的参数空间，让更多的奇异值分量参与训练，从而更准确地评估它们的重要性。
2. **预算削减阶段** (Budget Pruning)：在热身结束后，采用一个三次函数将预算从$b^{(0)}$ 平滑地降低到最终的目标预算$b^{(T)}$。
3. **微调阶段** (Fine-tuning)：当预算达到$b^{(T)}$后，保持预算不变，继续训练模型，直到收敛。

## DoRA

**DoRA**，Weight-Decomposed Low-Rank：DoRA将预训练的权重分解为量级和方向两部分进行微调，特别是采用LoRA进行方向部分的更新，以有效地减少可训练参数的数量，实现了与FT（Full Fine-tuning）非常相似的学习能力。

![](images\05LoRA Variants01.png)

预训练的权重已经包含了适合下游任务的大量知识，所以仅仅对于量级和方向中的某一个进行较大幅度的更新，就已经足够了。

DoRA限制LoRA只专注于方向$V$适应，同时允许量级幅度$m$可调，与原始方法相比简化了任务，在原始方法中，LoRA需要学习两个幅度的调整。
$$
W^{\prime}=\underline{m}\frac{V+\Delta V}{||V+\Delta V||_c}=\underline{m}\frac{W_0+\underline{BA}}{||W_0+\underline{BA}||_c}
$$

对于两个微调维度量级变化量ΔM以及方向变化量ΔD进行深度的分析：

在中间步骤中，LoRA表现出一致的正斜率趋势，这表明方向和量级的变化之间存在正比关系。

相比之下，全量微调（ Full Fine-tuning，FT）表现出更多样的学习模式，斜率相对为负。FT和LoRA之间的这种区别反映了它们各自的学习能力。LoRA倾向于按比例增加或减少量级和方向更新，但它缺乏进行更细微调整的细微能力。

而DoRA的行为更符合FT的学习模式，能够执行更轻微的方向变化和更显著的量级变化。

![](images\05LoRA Variants02.png)

# Reference

>1. AdaLoRA：Adaptive Budget Allocation For Parameter-Efficient Fine-Tuning， https://arxiv.org/abs/2303.10512
>2. DoRA：Weight-Decomposed Low-Rank Adaptation， https://arxiv.org/abs/2402.09353
