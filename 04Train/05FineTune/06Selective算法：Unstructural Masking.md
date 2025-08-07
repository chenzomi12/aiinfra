<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# Selective算法：Unstructural Masking

> Author by: 李亚鹏

## U-Diff-Pruning

**U-Diff-Pruning**（无结构差分剪枝）：对于每个新任务，学习一个非常稀疏的“差分向量”（diff vector） $δ_τ$ ，这个向量被加到原始参数上，从而使模型适应新任务。
$$
\theta_\tau=\theta+\delta_\tau
$$
为了鼓励稀疏性，最直接的方法是惩罚中非零元素的数量。这个数量由L0范数（ $||δ_τ ||_0 $ ）来度量。因此，优化目标变成了：
$$
\begin{aligned}\min_{\delta_\tau}&L(\mathcal{D}_\tau,f_\tau,\boldsymbol{\theta}+\boldsymbol{\delta}_\tau)+\lambda R(\boldsymbol{\theta}+\boldsymbol{\delta}_\tau)\end{aligned}
$$

$$
R(\boldsymbol{\theta}+\boldsymbol{\delta}_\tau)=\|\boldsymbol{\delta}_\tau\|_0=\sum_{i=1}^d\mathbb{1}\{\boldsymbol{\delta}_{\tau,i}\neq0\}
$$

然而，L0范数是不可导的，所以不能直接用梯度下降法来优化，这是剪枝领域的经典难题。

为了解决L0范数不可导的问题，进行L0范数的可微近似。 

分解差分向量：将$δ_τ$分解为一个掩码向量(mask vector)$z_τ$和一个潜在权重向量 (potential weight vector) $w_τ$的逐元素乘积。
$$
\boldsymbol{\delta}_\tau=\mathbf{z}_\tau\odot\mathbf{W}_\tau
$$
其中， $w_τ$表示一个稠密的、可学习的原始参数向量； $z_τ$表示一个理想情况下由0和1组成的二元掩码，决定了$w_τ$中的哪些元素被“激活”。若$z_(τ,i)=0 $，那么$δ_(τ,i)=0$，即实现了剪枝。

此时的二元掩码$z_τ$仍是不可微分的，为了对其进行松弛，进行Hard-Concrete distribution操作：

先为每个参数引入一个可学习的“概率”参数$α_τ$ ，$α_τ$控制了第$i$个参数是0还是1的倾向。

从一个标准的均匀分布中采样一个随机数$u$：
$$
\mathbf{u}\sim U(\mathbf{0},\mathbf{1})
$$
将$u$转化为一个服从逻辑分布的连续随机变量$s_τ$，其值受到随机噪声$u$和可学习参数$α_τ$的共同影响。
$$
\mathbf{s}_\tau=\sigma\left(\log\mathbf{u}-\log(1-\mathbf{u})+\boldsymbol{\alpha}_\tau\right)
$$
$s_τ$的分布总是在 (0, 1) 开区间内，永远不会正好等于 0 或 1。这对于实现真正的剪枝（即参数正好为0）是不利的。

因此， Hard-Concrete先将$s_τ$拉伸到(l, r)区间，其中$l<0$，$r>1$。再进行裁剪，将[0, 1]区间外的值置为确定的0/1。
$$
\bar{\mathbf{s}}_\tau=\mathbf{s}_\tau\times(r-l)+l
$$

$$
\mathbf{z}_\tau=\min(\mathbf{1},\max(\mathbf{0},\mathbf{\bar{s}}_\tau))
$$

此时，正则项可微，目标函数修正为：
$$
\min_{\boldsymbol{\alpha}_\tau,\mathbf{w}_\tau}\mathbb{E}_{\mathbf{u}\sim U[\mathbf{0},\mathbf{1}]}\left[L(\mathcal{D}_\tau,f_\tau,\boldsymbol{\theta}+\mathbf{z}_\tau\odot\mathbf{w}_\tau)\right]+\lambda\sum_{i=1}^d\sigma\left(\boldsymbol{\alpha}_{\tau,i}-\log\frac{-l}{r}\right)
$$

使用修正的目标函数，即可实现学习稀疏的差分向量的目标。

## SARA

**SARA**，Sparse Low-Rank Adaptation：受到剪枝的启发，在评估参数重要性后，不将影响低的参数剪掉，而是用利用这些暂时不重要的参数进行下游任务的训练。换句话说，就是优化稀疏权重矩阵（不重要参数矩阵）来学习特定任务的知识。

<img src="images\06Selective算法：Unstructural Masking01.png" style="zoom: 25%;" />

模型的大多数参数值都在0的附近。设定一个阈值，权重绝对值低于这个阈值的参数被置为0。实验证明，5e-4到1e-3的阈值下，置0操作对模型原始能力的影响微乎其微，也就是说，绝对值低于阈值的参数是无效（不重要）参数。

实验证明，由于训练过程的随机性而导致的初始无效参数，大部分随着训练时间的推移变得有效，可以利用这些暂时无效的参数来微调预训练的模型。

对潜在有效参数（初始无效参数）进行微调：
$$
\begin{aligned}\nabla P_M&=\nabla P\odot M+\mathbf{0}\odot(1-M)\\P_{new}&=P-\lambda\cdot\nabla P_M\end{aligned}
$$
其中$M$是掩码矩阵，$P_M$为潜在有效参数矩阵。梯度、更新只涉及$P_M$。

SARA微调方法很好的利用了初始无效参数，将这部分看似“无效”的参数重新利用起来，使其在下游任务微调中发挥作用。

## Reference

> 1. Diff-Pruning：Parameter-Efficient Transfer Learning with Diff Pruning， https://arxiv.org/abs/2012.07463
> 2. SARA：Sparse Low-rank Adaptation of Pre-trained Language Models， https://arxiv.org/abs/2311.11696
