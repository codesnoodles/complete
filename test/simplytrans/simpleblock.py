import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
import math
from transformers.activations import ACT2FN


# 自定义mlp所使用的组件
class pconv(nn.Module):
    """
    使用卷积层实现两个线性层的缩放，在里面不使用跳跃连接
    """

    def __init__(self,
                 nf,
                 nx,
                 resid_gain=None,
                 skip_gain=None,
                 trainable_gains=False,
                 init_type="normal",
                 bias=True):
        super().__init__()
        self.nf = nf
        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)
        if skip_gain is None:
            self.weight = nn.Parameter(torch.empty(nx, nf))
            if init_type == "orth":
                nn.init.orthogonal_(self.weight)
            elif init_type == "id":
                self.weight.data = torch.eye(nx)
            elif init_type == "normal":
                nn.init.normal_(self.weight, std=0.02)
            else:
                raise NotImplementedError
            self.skip = False

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf, )
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


# 自定义mlp
class pmlp(nn.Module):
    """
    原始模型使用了配置文件，看看能否更改成直接写入，然后删除不必要的选择模式。直接使用论文里面的模型。
    intermediate_size:是个输出维度
    embed_dim:是个中间维度
    activations_function:激活函数,可以不管
    lrelu_neg_slope:自定义的leaky_relu里面的参数,已经初始化了
    resid_pdrop:dropout的比例,手动设置为0.1
    mlp_proj_init_std:初始化的一个参数,用来控制初始化分布的方差
    """

    def __init__(self, intermediate_size, embed_dim, activations_function,
                 lrelu_neg_slope, resid_pdrop, mlp_proj_init_std):
        super().__init__()
        embed_dim = embed_dim
        self.c_fc = pconv(intermediate_size, embed_dim, bias=False)
        self.c_proj = pconv(embed_dim, intermediate_size, bias=False)
        if activations_function != "leaky_relu":
            self.act = ACT2FN[activations_function]
        else:
            self.act = LeakyReLU(negative_slope=lrelu_neg_slope)
        self.dropout = nn.Dropout(resid_pdrop)
        if mlp_proj_init_std is not None:
            nn.init.normal_(self.c_proj.weight, std=mlp_proj_init_std)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 自定义attn
class pattn(nn.Module):
    """
    a customisable attn sub block  that can implement shpaed attention and identity value weights
    参数多的离谱了
        max_position_embeddings:生成矩阵维度的参数，不清楚怎么设置的，查看源码解决
        hidden_size:输入attn的矩阵维度，也就是transformer所处理的哪个维度的大小
        num_attention_heads: 头的数量
        scale_attn_weights: 直接翻译是对于注意力的缩放？具体操作是将注意力除了一个矩阵
        scale_attn_by_inverse_layer_idx: 也是对注意力矩阵做缩放用的
        first_layer_value_resid_gain: 第一层的v的残差的增益？这是什么？
        value_resid_gain: v的残差增益
        value_skip_gain: v的跳跃连接的增益，问题是跳跃连接和残差不是一个东西吗？
        val_init_type: val的初始化类型有：id,normal,orth等。
        trainable_value_gains: v的增益
        last_laye_proj_resid_gain:最后一层的增益
        n_layer:层数
        proj_resid_gain: cproj 的增益，如果last_laye_proj_resid_gain是none，那么就使用这个
        proj_skip_gain: cproj的跳跃增益
        proj_init_type: cproj的初始化模式
        trainable_proj_gains:可训练的proj的增益
        key_init_std:k初始化的方差
        query_init_std:q初始化的方差
        val_proj_init_std:v初始化的方差
        attn_pdrop:attn的drop的比例
        resid_pdrop:残差的drop的比例
        attn_mat_skip_gain:注意力矩阵跳跃连接的增益
        trainable_attn_mat_gains: 可训练的跳跃连接的增益
        attn_mat_resid_gain: 注意力矩阵残差增益
        center_attn:中央注意力。。。。
        center_attn_gain:中央注意力增益
        layer_idx:层的id
    """

    def __init__(
        self,
        hidden_size,
        max_position_embeddings,
        num_attention_heads,
        scale_attn_weights,
        scale_attn_by_inverse_layer_idx,
        first_layer_value_resid_gain,
        value_resid_gain,
        value_skip_gain,
        val_init_type,
        trainable_value_gains,
        last_laye_proj_resid_gain,
        n_layer,
        proj_resid_gain,
        proj_skip_gain,
        proj_init_type,
        trainable_proj_gains,
        key_init_std,
        query_init_std,
        val_proj_init_std,
        attn_pdrop,
        resid_pdrop,
        attn_mat_skip_gain,
        trainable_attn_mat_gains,
        attn_mat_resid_gain,
        centre_attn,
        centre_attn_gain,
        layer_idx,
        is_cross_attention=False,
    ):
        super().__init__()
        assert is_cross_attention == False
        max_positions = max_position_embeddings
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads(got `embed_dim`:{self.embed_dim} and `num_heads`:{self.num_heads})."
            )
        self.scale_attn_weights = scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.ql_attn = pconv(2 * self.embed_dim, self.embed_dim)
        if first_layer_value_resid_gain is not None and layer_idx == 0:
            value_resid_gain = first_layer_value_resid_gain
        else:
            value_resid_gain = value_resid_gain
        if (value_skip_gain != 1 or value_resid_gain != 0
                or val_init_type != "id"):
            self.v_attn = pconv(self.embed_dim,
                                self.embed_dim,
                                resid_gain=value_resid_gain,
                                skip_gain=value_skip_gain,
                                trainable_gains=trainable_value_gains,
                                init_type=val_init_type,
                                bias=False)
        else:
            self.v_attn = nn.Identity()
        if (last_laye_proj_resid_gain is not None
                and layer_idx == n_layer - 1):
            proj_resid_gain = last_laye_proj_resid_gain
        else:
            proj_resid_gain = proj_resid_gain
        if proj_skip_gain != 1 or proj_resid_gain != 0 or proj_init_type != "id":
            self.c_proj = pconv(self.embed_dim,
                                self.embed_dim,
                                resid_gain=proj_resid_gain,
                                skip_gain=proj_skip_gain,
                                trainable_gains=trainable_proj_gains,
                                init_type=proj_init_type,
                                bias=False)
        else:
            self.c_proj = nn.Identity()
        self.split_size = self.embed_dim
        query_weight, key_weight = self.qk_attn.weight.data.split(
            self.split_size, dim=1)
        if query_init_std is not None:
            query_weight.normal_(mean=0.0, std=query_init_std)
        if key_init_std is not None:
            key_weight.normal_(mean=0.0, std=key_init_std)
        if val_proj_init_std is not None:
            self.v_attn.weight.data.normal_(mean=0.0, std=val_proj_init_std)
            self.c_proj.weight.data.normal_(mean=0.0, std=val_proj_init_std)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.pruned_heads = set()
        self.attn_mat_resid_gain = nn.Parameter(
            attn_mat_resid_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=trainable_attn_mat_gains)
        self.attn_mat_skip_gain = nn.Parameter(
            attn_mat_skip_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=trainable_attn_mat_gains)
        self.centre_attn = centre_attn
        uniform_causal_attn_mat = torch.ones(
            (max_positions, max_positions),
            dtype=torch.float32) / torch.arange(1, max_positions + 1).view(
                -1, 1)
        self.register_buffer(
            "uniform_causal_attn_mat",
            torch.tril(uniform_causal_attn_mat, ).view(1, 1, max_positions,
                                                       max_positions),
            persistent=False,
        )
        self.centre_attn_gain = nn.Parameter(
            centre_attn_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=trainable_attn_mat_gains and centre_attn_gain != 0,
        )
        self.register_buffer(
            "diag",
            torch.eye(max_positions).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions),
                           dtype=torch.bool)).view(1, 1, max_positions,
                                                   max_positions),
            persistent=False,
        )

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1)**0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device)
        attn_weights = torch.where(causal_mask,
                                   attn_weights.to(attn_weights.dtype),
                                   mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        new_attn_weights = self.attn_mat_resid_gain * attn_weights.type(
            value.dtype)

        if self.centre_attn:  # centre_attn = false 下面的配置没有作用。
            post_sm_bias_matrix = (
                self.attn_mat_skip_gain *
                self.diag[:, :, :key_length, :key_length]
            ) - self.centre_attn_gain * (
                self.uniform_causal_attn_mat[:, :, key_length - query_length:
                                             key_length, :key_length])
            new_attn_weights = new_attn_weights + post_sm_bias_matrix

        new_attn_weights = self.attn_dropout(new_attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            new_attn_weights = new_attn_weights * head_mask

        attn_output = torch.matmul(new_attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1,
                              3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size, )
        return tensor.view(new_shape)


# 自定义block
class pblock():
    pass


# 一个root mean square layer normalization 函数
class RMSNorm(nn.Module):
    """
    输入的参数d是需要进行正则化的张量的最后一个维度大小
    """

    def __init__(self, d, eps=1e-8) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * self.d**(-1.0 / 2)
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


class LeakyReLU(nn.Module):
    # LeakyReLU nonlinearity.
    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_slope: float

    def __init__(self,
                 negative_slope: float = 1e-2,
                 inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input >= 0.0, input, input * self.negative_slope)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "negative_slope={}{}".format(self.negative_slope, inplace_str)


if __name__ == "__main__":

    normlayer = RMSNorm(3)
    input = torch.rand(2, 3)
    output = normlayer(input)
    print(output.size())
    linearlayer = pconv(6, 3)
    x = torch.rand(2, 5, 3)
    y = linearlayer(x)
    print(y.size())
    z = torch.rand(3, 4, 5)
    testmlp = pmlp(7, 5, "leaky_relu", 1e-2, 0.1, 0.1)
    mlpout = testmlp(z)
    print(mlpout.size())
    z = torch.rand(3, 4, 5)
    attn = pattn(5, 5, 4, None, None, None, 1, None, "normal", False, None, 4,
                 1, None, "normal", False, None, None, 0, 0, 0, False, 1,
                 False, 1, 1)
