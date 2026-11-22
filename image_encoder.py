"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # 【学生注释】TODO-02：多头拆分
        # 输入形状 [B,N,D]，B批次，N token数量，D总特征维度768
        # 先reshape：把最后一维拆成 heads × head_dim，变成 [B,N,H,head_dim]
        # 再转置交换N和H轴，输出 [B,H,N,head_dim] 符合多头注意力格式
        B, N, D = x.shape
        x = x.reshape(B, N, self.heads, self.head_dim)
        x = x.transpose((0, 2, 1, 3))
        return x

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = (x @ self.qkv_weight.T + self.qkv_bias).astype(np.float32)
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        # TODO-03a：计算注意力分数
        # k做转置，最后两个维度互换，方便和q矩阵相乘
        # Q @ K^T 得到相似度矩阵，除以根号head_dim做缩放，防止数值过大
        k_t = k.swapaxes(-1, -2)
        scores = (q @ k_t / np.sqrt(self.head_dim)).astype(np.float32)

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        # TODO-03b：加权Value + 合并多头
        # 注意力权重和v相乘，得到每个头的输出context
        # 转置把head轴放回，再reshape合并多个头，变回单个token特征 [B,N,D]
        context = (coefficients @ v).astype(np.float32)
        B, H, N, hd = context.shape
        context = context.transpose((0, 2, 1, 3))
        merged = context.reshape(B, N, H * hd)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        # TODO-03c：注意力输出线性投影
        # 使用out的权重和偏置，对合并多头后的特征做线性变换
        update = (merged @ self.out_weight.T + self.out_bias).astype(np.float32)
        return update


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # TODO-04a：第一个残差连接
        # 将原始输入x加上注意力模块输出，残差相加
        residual = (x + update).astype(np.float32)

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        # TODO-04b：前馈MLP + 第二个残差连接
        # 先升维，GELU激活，再降维，最后和前面的residual相加
        hidden = (u @ self.fc1_weight.T + self.fc1_bias).astype(np.float32)
        activated = self.gelu(hidden)
        mlp_update = (activated @ self.fc2_weight.T + self.fc2_bias).astype(np.float32)
        output = (residual + mlp_update).astype(np.float32)
        return output


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01：图片分patch切块
        # 输入图片形状：B,3,224,224，patch尺寸32×32，一共7×7=49个patch
        # 拆分维度，调整轴顺序保证patch遍历顺序：从左到右、从上到下
        # 每个patch内部按照C,H,W顺序展平，3*32*32=3072
        B, C, H, W = images.shape
        patch_size = 32
        num_patch_h = H // patch_size
        num_patch_w = W // patch_size
        # 拆分：B, C, 7,32,7,32
        x = images.reshape(B, C, num_patch_h, patch_size, num_patch_w, patch_size)
        # 调换轴顺序 B,7,7,C,32,32
        x = x.transpose((0, 2, 4, 1, 3, 5))
        # 合并patch网格维度，展平单个patch像素
        x = x.reshape(B, num_patch_h*num_patch_w, C*patch_size*patch_size)
        return x.astype(np.float32)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        # TODO-05a：patch线性投影，映射到768维特征
        patch_tokens = (patches @ self.patch_weight.T).astype(np.float32)

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1)).astype(np.float32)

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        # TODO-05b：拼接CLS token + 加上位置编码
        # 在token维度最前面插入CLS，一共49+1=50个token
        # position编码自动对batch广播相加
        x = np.concatenate([cls, patch_tokens], axis=1)
        x = (x + self.position[None, :, :]).astype(np.float32)
        return x

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit‑vector helper is given.
        # TODO-06：取出CLS、后层归一化、投影到512维，最后归一化成单位向量
        cls_token = tokens[:, 0, :]
        normalized_cls = self.post_norm(cls_token)
        output = (normalized_cls @ self.projection).astype(np.float32)

        return unit_vector(output)