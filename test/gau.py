import torch.nn as nn
import torch
import torch.nn.functional as F


class GAU(nn.Module):

    def __init__(self,
                 dim,
                 query_key_dim=128,
                 expansion_factor=2,
                 add_residual=True,
                 dropout=0):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_hidden = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                       nn.SiLU())
        self.to_qk = nn.Sequential(nn.Linear(dim, query_key_dim), nn.SiLU())
        self.gamma = nn.Parameter(torch.zeros(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)
        self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim),
                                    nn.Dropout(dropout))
        self.add_residual = add_residual

    def forward(self, x):
        seq_len = x.shape[-2]
        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)
        Z = self.to_qk(normed_x)
        QK = torch.einsum('...d,h d-> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)
        sim = torch.einsum('b i d,b j d ->b i j', q, k) / seq_len
        A = F.relu(sim)**2
        A = self.dropout(A)
        V = torch.einsum('bij,bjd->bid', A, v)
        V = V * gate
        out = self.to_out(V)
        if self.add_residual:
            out = out + x
        return out


gau = GAU(dim=512, query_key_dim=128, expansion_factor=2)
x = torch.randn(1, 1024, 512)
out = gau(x)
print(out.size())
