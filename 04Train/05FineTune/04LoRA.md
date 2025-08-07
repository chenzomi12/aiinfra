<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# LoRA算法

> Author by: 李亚鹏

## 微调原理

**LoRA**，Low-Rank Adaptation：LoRA调整的参数主要是Self-Attention里的四个权重矩阵$W_q$（query)、$W_k$（key)、$W_v(value)$、$W_o$ (output)，以及前馈层的$W_{up}$ 和$W_{down}$ 。 

将权重矩阵记为*W*，$d×d $维，输入特征为$x$。LoRA初始化两个矩阵$A$和$B$，对$A$使用随机高斯初始化，将$B$初始化为0。$A$的维度为$r×d$维，*B* 的维度为$d×r$ 维。其中$r$ 是rank秩，远小于$d$ 。

![](images\04LoRA01.gif)

在前向传播的过程中，输出$h$不再是原始权重矩阵$W_0$，而是加上了$BAx$，并加入参数$α$ 缩放$\Delta W_x$ ，如下式：
$$
h=W_0x+\alpha\Delta Wx=W_0x+\alpha BAx
$$
在参数更新时，LoRA冻结了$W_0$权重，只更新$B$和$A$。微调结束后，保留$A$和$B$作为LoRA矩阵，推理时用。为了减少推理延迟，可以将$BA$矩阵与$W_0$合并。

以query值的计算为例，使用LoRA权重后，计算方式如下：
$$
query=(W_q+\alpha BA)x
$$
LoRA的高效性主要来源于低秩。通过低秩近似，LoRA 能够捕捉到数据的主要变化方向，并在保证模型表达能力的前提下，通过调整少量参数实现对模型的微调。

因此，如何为不同任务选择合适的秩 r 是一个关键问题，r的取值一般为4到64不等。一般来说简单任务需要的秩不大，而高难度任务需要较高的秩。

LoRA 不需要在模型中引入新的模块或网络结构，而是直接对现有的线性层进行调整。这使得 LoRA 的参数效率更高，且不增加模型的计算复杂度。与prompt-base微调算法相比，LoRA 的低秩矩阵分解能够在更深层次上调整模型的表示能力，而不仅仅是影响输入层或局部区域。

## 微调效果

* **训练效率提升：**训练参数量一般低于总体参数的1%，训练速度可提升数倍至数十倍，显存占用大幅降低，消费级显卡即可微调。

* **存储节省：**LoRA模型文件极小，通常仅为几 MB 到几百 MB，便于分发管理。

* **即插即用：**一个基础模型可搭配多个不同LoRA模型文件，实现多种效果。

* **高质量微调：**性能媲美全量微调，且不易“遗忘”。

## 多模态应用

现有多模态模型，依赖一个独立的、预训练的视觉模型（如 ViT）来提取图像特征。

推理时，ViT 和连接器是额外且独立的模块，这增加了计算成本和内存占用。

工作流程是串行的，LLM 必须等待 ViT 完全处理完图像才能开始工作。

<img src="images\04LoRA02.png" style="zoom: 67%;" />

**VoRA**（Vision as LoRA） 提出了一种颠覆性的解决方案：不再依赖外部视觉专家，而是让 LLM 自己“学会”看图。

在LLM的每一层线性层中，都加入专门用于处理视觉信息的低秩适配器（LoRA）。

在训练时，冻结LLM的原始参数，只训练这些新加入的LoRA层和轻量级的图像嵌入层,从而将“视觉”能力，直接集成到 LLM 内部。

将语言能力（在冻结的LLM中）和新学习的视觉能力（在LoRA层中）解耦，从而避免了直接训练整个模型时可能出现的训练不稳定或“灾难性遗忘”问题。

<img src="images\04LoRA03.png" style="zoom: 67%;" />

# Reference

> 1. LoRA：Low-Rank Adaptation of Large Language Models，https://arxiv.org/abs/2106.09685 
> 1. VoRA：Vision as LoRA， https://arxiv.org/abs/2503.20680
