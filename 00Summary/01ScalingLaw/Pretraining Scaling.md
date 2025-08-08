<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 预训练Scaling Law

预训练阶段的Scaling Law（缩放定律）主要关注如何提升模型的知识容量。

哪些因素影响大语言模型的知识容量和性能，决定着我们需要在预训练过程中在什么方面进行投入；而这些因素如何影响模型性能，则决定着预训练需要投入多少资源。
本节将以 [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) 这篇论文为主，辅以其他研究成果，介绍影响大语言模型预训练效果的重要因素。


## 什么是大型语言模型缩放定律

“缩放定律”是指神经网络性能随关键因素（如模型参数数量、训练数据集大小和计算成本）变化而呈现出的经验性、可预测的规律。这些定律揭示了在投入不同资源时，模型性能（通常以交叉熵损失衡量）如何以幂律形式平滑且可预测地改善，从而为研究人员和工程师提供高效的超参数选择、模型设计和计算资源分配的指导。

在预训练阶段，大语言模型通常采用自监督学习方式进行训练。自监督意味着不需要对数据进行标注，即可用于训练模型。对于主流的自回归语言模型，预训练的核心任务是根据给定的文本序列，预测下一个可能出现的token（token的概念近似于词）。
在这一阶段，模型性能的核心衡量指标是交叉熵损失。交叉熵损失量化了模型预测与真实标签之间的差异，其数值越低，表示模型预测的准确性越高，性能越好。它量化了模型学习真实语言分布的情况，一定程度反映了模型理解和生成语言的能力，。

![预测下一个token](images/01ScalingLaw04.png)


## 核心发现：模型参数、数据集和计算量对模型性能的影响

Kaplan et al.于2020年发表的论文[Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)是LLM缩放定律领域的奠基性工作。该研究通过在WebText2数据集上训练Transformer模型，首次系统性地揭示了语言模型性能与模型规模、数据集大小和计算量之间的精确幂律关系。

该研究的三大核心发现如下：

**数据集大小 (D) 与损失**：  
在不受其他因素限制下，性能随数据集大小 $D$ 的增加而幂律提升， 

$$
L(D) \approx \left( \frac{D_c}{D} \right)^{\alpha_D}
$$  

其中， $\alpha_D \approx 0.095, D_c=5.4*10^{13}$ 。

**模型规模 (N) 与损失**：  
在不受其他因素限制下，语言模型性能随非嵌入参数数量$N$的增加而幂律提升。非嵌入参数是排除了词汇和位置嵌入的模型参数，是真正在训练中用于学习数据分布的参数。

$$
L(N) \approx \left( \frac{N_c}{N} \right)^{\alpha_N}
$$  

其中， $\alpha_N \approx 0.076, N_c=8.8*10^{13}$ 。

**计算量 (C) 与损失**：  
在不受其他因素限制下，性能随优化分配的训练计算量 $C_{\text{min}}$ 的增加而幂律提升，  

$$
L(C_{\text{min}}) \approx \left( \frac{C_c}{C_{\text{min}}} \right)^{\alpha_C}
$$

其中， $\alpha_C \approx 0.050，C_c=2.3*10^8$ 。

计算量 $C \approx 6NBS$ 。其中 $B$ 表示batch size， $S$ 表示训练步长。 $L(C_{\text{min}})$ 的图与其他两个图的不同在于，其是由多个曲线的最低点相连得出的。这多个曲线就是不同batch size下，模型随训练步长的变化。这里在展示的是要让模型到达指定性能需要最少投入的计算资源。从另一面看， $L(C_{\text{min}})$ 展示了在计算资源固定的前提下，模型性能能收敛到哪里。

![模型性能与计算量、数据集大小、模型规模之间的关系](images/01ScalingLaw01.png)

在模型规模 $N$ 确定的前提下， $C$ 由batch size $B$ 和训练步长 $S$ 影响。

增大batch size能有效降低梯度估计中的随机噪声，使优化路径更为稳定且接近理论最优方向。研究表明存在一个关键阈值——最优batch size，在此阈值之下，批量规模的增长与模型收敛速度呈正相关；而一旦超过此阈值，继续扩大批量带来的性能提升则趋于边际化。
因此，为最大化计算资源利用效率和训练时间价值，应当优先采用接近最优batch size的配置。

![batch size](images/01ScalingLaw07.png)

在充分大batch size下，训练步长与模型性能间存在幂律关系，且此规律在不同参数规模的模型架构中均呈现一致性。

![训练步长](images/01ScalingLaw08.png)

前面的结论成立有一个前提，就是不受其他因素限制，这在实际情况下很难成立。
在资源受限时，模型性能的持续提升依赖于模型规模 $N$ 和数据集大小 $D$ 的同步扩展。如果 $N$ 或 $D$ 固定而另一个持续增加，模型性能会在达到一定水平后呈现边际递减效应，提升速度显著减缓。
建议在给定 $C$ 时，模型与数据应符合如下比例： $N_{opt} \propto C^{0.74}, D_{opt} \propto C^{0.27}$ 。

![模型参数与数据](images/01ScalingLaw03.png)
![模型参数与数据](images/01ScalingLaw05.png)

此外，训练曲线（损失随训练步数的变化）遵循可预测的幂律，其参数大致独立于模型大小。这意味着通过外推早期训练曲线，可以大致预测模型在长时间训练后能达到的损失水平 。   
另一个重要的发现是，大型模型比小型模型更具样本效率。它们能够以更少的优化步骤和更少的数据点达到相同的性能水平 。这意味着大模型能更有效地从数据中学习，从每个数据点中提取更多的信息。

![样本效率](images/01ScalingLaw02.png)


后续论文[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)进一步扩充了模型和数据规模，提出了对最优资源分配策略的修正。其核心发现是，对于给定的计算预算，模型参数和数据集应以大致相等的比例缩放，即 $N_{opt} \propto C^{0.5}, D_{opt} \propto C^{0.5}$ ，而非Kaplan et al.建议的侧重模型参数。

![作者使用三种不同方法的预测最优模型大小，发现当时的大模型训练严重不足。这也导致2021年大家复现GPT3都没有做出特别好的效果。](images/01ScalingLaw06.png)

## 其他因素
Kaplan et al.还讨论了其他可能影响模型性能的因素，包括：学习率、模型结构、上下文长度和数据分布偏移，发现这些因素的影响有限。

**学习率**： 
学习率和batch size是两个密切相关的超参数。通常，较大的batch size需要较大的学习率。
Kaplan et al.发现只要学习率不是太小且衰减不要太快，学习率对性能的影响并不强。

不过后续研究[Predictable Scale: Part I, Step Law – Optimal Hyperparameter Scaling Law in Large Language Model Pre-training](https://arxiv.org/abs/2503.04715)比较了不同的最优学习率和batch size，并提出了自己的改进方案。

![最优学习率和batch size](images/01ScalingLaw09.png)

**模型结构**： 

$$
N \approx 2d_{model}n_{layer}(2d_{attn} + d_{ff})
$$

当固定模型规模时，模型性能对包括深度、宽度、注意力头和前馈维度在内的模型结构的依赖性非常轻微。

![最优学习率和batch size](images/01ScalingLaw10.png)

需要注意的是，上述结论源于dense架构Transformer在下一个token预测任务上的表现。在实际应用中，注意力头的配置对模型性能有影响；对于Mixture of Experts (MOE) 架构，专家数量的选择也会直接影响最终效果。

**上下文长度**：
观察下图，除第一个token外，在不同位置的token的损失随模型增大而减小。这说明更大的模型在预测各个位置的token时都表现更好。第一个token的预测不遵循这一规律，可能因为缺乏前置上下文信息。

进一步分析可得出以下补充结论，可以进一步得出以下观察（论文中未明确提及）：

通过对比Token 4/8 < Token 4/1024，发现在相同绝对位置上，较短上下文中的token损失更小。

通过对比Token 1024/1024 < Token 8/8，发现在相同相对位置上，较长上下文中的token损失更小。

![上下文长度](images/01ScalingLaw11.png)

**数据分布偏移**：
模型的训练数据的分布和测试数据的分布很可能是不一样的，这种情况叫数据分布偏移。此时测试数据属于域外数据。
Kaplan et al.发现模型在域外数据上的性能相比于训练集会出现固定幅度的下降，但整体表现仍大致与其在训练集上的性能成正比。
然而在真实场景下，源于分布偏移的知识缺失依然会对使用体验造成明显影响。

![数据分布偏移](images/01ScalingLaw12.png)


## 参考资料

Scaling Laws for Neural Language Models -Kaplan et al.

Training Compute-Optimal Large Language Models -Hoffmann et al.

Predictable Scale: Part I, Step Law – Optimal Hyperparameter Scaling Law in Large Language Model Pre-training -Li et al. 

Deep Dive into LLMs like ChatGPT -Andrej Karpathy

和张祥雨聊，多模态研究的挣扎史和未来两年的2个“GPT-4时刻” -张小珺Jùn｜商业访谈录

