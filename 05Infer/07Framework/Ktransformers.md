# Ktransformers

## 1. Arithmetic Intensity Guided Offloading



## 2. MLA算子的矩阵吸收优化

Multi-head Latent Attention（MLA）是DeepSeek V2中提出的一种Attention变体，其核心思想是，不直接存储完整的、与序列长度线性增长的KV Cache，而是通过一个**可学习的低秩投影**，将历史的Key和Value信息**压缩**成一个尺寸更小、更紧凑的“潜在缓存”（Latent Cache）。而MLA算子，从其计算特征来看，同时解决了这两方面的问题：一方面，通过低秩压缩大幅降低了KV Cache的大小，另一方面，MLA解压缩后的多头注意力机制能够提供较高的计算强度，有助于充分利用GPU的算力资源。很明显，MLA算子是针对现代GPU硬件特点“量体裁衣”定制的一个注意力机制，通过对存储和计算的再平衡，能够充分发挥现代GPU的各项优势。

### 2.1. MLA的计算过程

以Deepseek V2为例，假如给定一个输入向量$h_t \in \mathbb{R}^{B \times L \times 5120}$，其中$B$为batch size，$L$为sequence length，5120是DeepSeek V2中的特征向量维度，MLA的计算过程如下。**这里补一张MLA计算流程图**

#### 2.1.1. Query

在DeepSeek-V2中，Query也采用了低秩压缩的方式。首先，将输入向量投影到一个1536维的低维空间：

$$ c_t^Q = W^{DQ} h_t \in \mathbb{R}^{B \times L \times 1536} $$

然后，将其投影到$\mathbb{R}^{H \times 128}$的多头向量空间上（其中$H=128$是heads数），得到了Q向量的第一部分：

$$ q_t^C = W^{UQ} c_t^Q \in \mathbb{R}^{B \times L \times H \times 128} $$

再将其投影到$\mathbb{R}^{H \times 64}$上并使用RoPE嵌入位置信息，得到Q向量的第二部分：

$$ q_t^R = \mathrm{RoPE}(W^{QR} c_t^Q) \in \mathbb{R}^{B \times L \times H \times 64} $$

将两部分拼接的到最终的Q向量：

$$ q_t = [q_t^C, q_t^R] \in \mathbb{R}^{B \times L \times H \times 192} $$

#### 2.1.2. Key和Value

计算KV向量时，首先需要将输入向量投影为512维的联合压缩表示：

$$ c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{B \times L \times 512} $$

与Query的计算过程类似，Key的第一部分是将$c_t^{KV}$通过投影解压缩到$\mathbb{R}^{H \times 128}$的多头向量空间：

$$ k_t^C = W^{UK} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$

Key的第二部分是将输入向量投影到64维向量空间并施加RoPE嵌入位置信息：

$$ k_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{B \times L \times 64} $$

与Query不同的是，完整的Key是将Key的第二部分广播到每个head后与第一部分拼接得到：
$$
k_t = \begin{bmatrix}

​    k_{t,1}^C & k_t^R \\ 

​    k_{t,2}^C & k_t^R \\

​    \vdots & \vdots \\

​    \end{bmatrix} \in \mathbb{R}^{B \times L \times H \times 192}
$$
也就是说，每个head的RoPE部分是完全相同的，这是MLA中Key共享位置编码的设计。

Value向量的计算较为简单，直接将$c_t^{KV}$解压缩到$\mathbb{R}^{H \times 128}$即可：

$$ v_t = W^{UV} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$

#### 2.1.3. Attention

Attention的计算过程和传统的MHA并无差异。首先计算attention score：
$$
a = \mathrm{softmax}\left(\frac{q_t^\top k_t + \mathrm{Mask}}{\sqrt{192}}\right) = 

\mathrm{softmax}\left(\frac{{q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R + \mathrm{Mask}}{\sqrt{128 + 64}} \right)

\in \mathbb{R}^{B \times L \times H \times L}
$$
计算对V的加权和，并将所有head压平，得到Attention输出：

$$ o = a \cdot v_t \in \mathbb{R}^{B \times L \times H \times 128} \cong \mathbb{R}^{B \times L \times 16384} $$

经过另一个矩阵的投影，就能得到MLA的最终输出：

$$ u = W^O o \in \mathbb{R}^{B \times L \times 5120} $$


``` python
def forward(...):
    bsz, q_len, _ = hidden_states.size()
    
    # 计算Q：先降维再升维，好处是相比直接使用大小为 [5120, 24576] 的矩阵
    # [5120, 1536] * [1536, 24576] 这样的低秩分解在存储空间和计算量上都大幅度降低
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    # 切分 rope 和非 rope 部分
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    
    # 计算KV
    # 一个优化的 MLA KVCache 实现只需要缓存这个 compressed_kv 就行，不过后面实际上展开了
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    # 此处compressed_kv 对应公式中的 c_t^{KV}
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    # 将 MLA 展开成标准 MHA 的形式
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )
    # 因为 kv_b_proj 打包了 W^{UK} 和 W^{UV} 把他们分离出来
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    ...
    # 给需要 rope 的部分加 rope
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    
    # 更新和拼接历史 KVCache，可以看到这里存储的是展开后的 MHA KVCache
    # 其中 q_head_dim 等于 qk_nope_head_dim + qk_rope_head_dim
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # 后续就是标准的 MHA 代码，不再赘述
    ...
```

### 2.2. Cache Compress

相比于缓存KV，缓存c更能够节省显存



### 2.3. Projection Absorption

上述分析和实验结果表明，相比缓存完整的KV Cache，缓存压缩后的KV Cache会带来较大的性能下降。另外一个重要的问题是，当前的CacheCompressed实现实际上并不能缓解KV Cache过大的问题，这是由于在计算MLA的时候，仍然需要存储解压后的完整的KV Cache，这很可能引起内存溢出（Out of Memory, OOM）崩溃。

所幸DeepSeek-V2的论文中提出，可以将KV的解压缩矩阵吸收到Q-projection和Out-projection中，从而可以在不解压缩KV Cache的情况下直接计算最终的Attention结果。

对于K的吸收，在Attention Score的计算公式中，非RoPE部分可以做如下展开：
$$
{q_t^C}^\top k_t^C = (W^{UQ} c_t^Q)^{\top} W^{UK} c_t^{KV} = {c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK} c_t^{KV} = ({c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK}) c_t^{KV}
$$
即通过矩阵乘法结合律，可以改为计算$({c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK})$，避免了解压缩出完整的K矩阵。此外，在原始版本的解压缩的过程中，由于每个token的key都需要与$W^{UK}$相乘才能得到，因此计算量较大；矩阵吸收后，$W^{UK}$只需要对$q_t^C$这一个向量相乘，也大大减少了浮点计算量。

对于V的吸收，情况稍微复杂。为表述的清楚性，我们采用Einstein求和约定描述该过程：

``` python
v_t = einsum('hdc,blc->blhd', W_UV, c_t_KV) *# (1)*

o   = einsum('bqhl,blhd->bqhd', a, v_t)     *# (2)*

u   = einsum('hdD,bhqd->bhD', W_o, o)       *# (3)*

*# 将上述三式合并，得到总的计算过程*

u   = einsum('hdc,blc,bqhl,hdD->bhD', W_UV, c_t_KV, a, W_o)

*# 利用结合律改变计算顺序*

o_  = einsum('bhql,blc->bhqc', a, c_t_KV) *# (4)*

o   = einsum('bhqc,hdc->bhqd', o_, W_UV)  *# (5)*

u   = einsum('hdD,bhqd->bqD', W_o, o)     *# (6)*
```

具体的代码实现如下：
``` python
# Absorbed_CacheCompressed
def forward(hidden_states_q: torch.Tensor, q_position_ids: torch.LongTensor, compressed_kv: torch.Tensor):
    ...
    kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
    q_absorb = kv_b_proj[:, :self.qk_nope_head_dim,:]
    out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]
    
    cos, sin = self.rotary_emb(q_pe)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, q_position_ids)
    
    qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
    # 此处改变了q_nope的计算顺序
    query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
    query_states[:, :, :, self.kv_lora_rank :] = q_pe
    
    ...

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q_nope.dtype)
    # 此处改变了attn_output的计算顺序
    attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
    attn_output = torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)
```

### 2.4. Move Elision

由于Key中存在多头共享旋转位置编码（RoPE）的设计，而在DeepSeek V2源代码中存在RePE和No RoPE矩阵的广播和拼接操作，这会导致大量的显存浪费。因此采用Move Elision（移动省略）节省显存


不过，这样还不能完全发挥出MLA的威力。在原始代码中，query_states和key_states会通过拼接RoPE和非RoPE部分得到：
``` python
def forward(...):
    ...
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    ...
```
当我们采取了上述优化后，此处的拼接过程会产生大量无用的数据拷贝和广播，同时也会占用大量显存空间导致OOM。为此，我们采用MoveElision优化策略，
即省略此处的拼接RoPE部分和非RoPE部分的过程，而是直接分别计算量部分的额Attention Score并相加（考虑$q_t^\top k_t = {q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R$）：
``` python
# Absorbed_CacheCompressed_MoveElision
def forward(...):
    ...
    # qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
    # query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
    # query_states[:, :, :, self.kv_lora_rank :] = q_pe

    # key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, qk_head_dim)
    # key_states[:, :, :, : self.kv_lora_rank] = compressed_kv.unsqueeze(1)
    # key_states[:, :, :, self.kv_lora_rank :] = k_pe

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

    attn_weights = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)
    attn_weights *= self.softmax_scale
    ...
```

## 3. 基于cuda graph的调用优化



## 4. 基于稀疏注意力的长文本优化



## 5. CPU的优化

### 5.1. GGUF格式文件

[资料来源][GGUF]

GGUF 是一种文件格式，用于存储模型，以供 GGML 及基于 GGML 的执行器进行推理使用。

GGUF 是一种二进制格式，其设计旨在实现模型的快速加载与保存，并易于读取。模型通常使用 PyTorch 或其他框架进行开发，然后转换为 GGUF 格式，以便在 GGML 中使用。

它是 GGML、GGMF 和 GGJT 的后继文件格式。其设计目标是通过包含加载模型所需的全部信息来消除歧义。同时，它的设计也具备可扩展性，因此可以在不破坏兼容性的前提下向模型中添加新信息。

#### 技术规范

GGUF 是一个基于现有 GGJT 的格式，但对格式进行了一些修改，使其更具可扩展性且更易于使用。它期望具备以下特性：

- **单文件部署**：模型可以被轻松地分发和加载，并且不需要任何外部文件来提供额外信息。
- **可扩展性**：可以为基于 GGML 的执行器添加新功能，或为 GGUF 模型添加新信息，而不会破坏与现有模型的兼容性。
- **`mmap` 兼容性**：模型可以使用 `mmap` 进行加载，以实现快速的加载和保存。
- **易于使用**：无论使用何种编程语言，只需少量代码即可轻松加载和保存模型，无需外部库。
- **信息完备**：加载模型所需的所有信息都包含在模型文件中，用户无需提供任何额外信息。

GGJT 和 GGUF 之间的关键区别在于，GGUF 对超参数（现在称为元数据）使用了键值（key-value）结构，而不是一个无类型的值列表。这样一来，就可以在不破坏与现有模型兼容性的情况下添加新的元数据，并可以用对推理或模型识别有用的附加信息来注解模型。

### GGUF 命名约定

GGUF 遵循 `<基础名称><尺寸标签><微调><版本><编码><类型><分片>.gguf` 的命名约定，其中每个组件（如果存在）都由 `-` 分隔。此约定的最终目的是为了方便人类用户能够一目了然地获取模型最重要的细节。由于现有 gguf 文件名的多样性，该约定并非旨在可以被程序完美解析。

这些组件是：

- **BaseName (基础名称)**：模型基础类型或架构的描述性名称。
  - 此名称可从 gguf 元数据 `general.basename` 派生，并将空格替换为短横线。
- **SizeLabel (尺寸标签)**：参数权重级别（对排行榜有用），表示为 `<专家数量>x<数量><数量级前缀>`。
  - 此标签可从 gguf 元数据 `general.size_label` 获取（如果可用），或在缺失时进行计算。
  - 在“数量”部分支持使用带单个字母数量级前缀的四舍五入小数，以辅助表示浮点指数，如下所示：
    - **Q**: Quadrillion (千万亿) 参数。
    - **T**: Trillion (万亿) 参数。
    - **B**: Billion (十亿) 参数。
    - **M**: Million (百万) 参数。
    - **K**: Thousand (千) 参数。
  - 可以根据需要附加额外的 `-<属性><数量><数量级前缀>` 来指示其他感兴趣的属性。
- **FineTune (微调)**：模型微调目标的描述性名称（例如 Chat、Instruct 等）。
  - 此名称可从 gguf 元数据 `general.finetune` 派生，并将空格替换为短横线。
- **Version (版本)**：（可选）表示模型的版本号，格式为 `v<主版本号>.<次版本号>`。
  - 如果模型缺少版本号，则假定为 `v1.0`（首次公开发行版）。
  - 此版本号可从 gguf 元数据 `general.version` 派生。
- **Encoding (编码)**：指示应用于模型的权重编码方案。然而，内容的类型、混合和排列方式由用户代码决定，并可能根据项目需求而变化。
- **Type (类型)**：指示 gguf 文件的种类及其预期用途。
  - 如果缺失，则文件默认为一个典型的 gguf 张量模型文件。
  - **LoRA**：表示 GGUF 文件是一个 LoRA 适配器。
  - **vocab**：表示 GGUF 文件仅包含词汇表数据和元数据。
- **Shard (分片)**：（可选）指示并表示模型已被分割成多个分片，格式为 `<分片编号>-of-<分片总数>`。
  - **ShardNum (分片编号)**：此分片在模型中的位置。必须是零填充的 5 位数字。
  - 分片编号总是从 `00001` 开始（例如，第一个分片总是从 `00001-of-XXXXX` 开始，而不是 `00000-of-XXXXX`）。
  - **ShardTotal (分片总数)**：该模型的分片总数。必须是零填充的 5 位数字。

![gguf](asset/gguf.png)

### 5.3. AMX后端

AMX与AVX-512的区别

![amx_intro](asset/amx_intro.png)



ktransformers用多级缓存加速activation和weight

![amx](asset/amx.png)

## 6. 参考文献

[GGUF]: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

