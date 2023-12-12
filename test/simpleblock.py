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
    """

    def __init__(self,
                 config,
                 max_position_embeddings,
                 hidden_size,
                 num_attention_heads,
                 scale_attn_weights,
                 scale_attn_by_inverse_layer_idx,
                 is_cross_attention=False,
                 layer_idx=None):
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
