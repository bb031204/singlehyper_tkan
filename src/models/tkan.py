import torch
import torch.nn as nn
from .kan_linear import KANLinear


class RKANSubLayer(nn.Module):
    """RKAN 子层：每个时间步维护独立的递归子状态 z，
    并通过 KAN 子层 phi 产生非线性映射输出。

    对应 TKAN 论文公式：
      s_{l,t} = W_{l,x} x_t + W_{l,h} z_{l,t-1}
      o_{l,t} = phi_l(s_{l,t})          # phi 为 KAN 子层
      z_{l,t} = W_{hh} z_{l,t-1} + W_{hz} o_{l,t}
    """

    def __init__(self, input_size: int, hidden_size: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.wx = KANLinear(input_size, hidden_size, grid_size, spline_order)
        self.wh = KANLinear(hidden_size, hidden_size, grid_size, spline_order)
        self.hz = KANLinear(hidden_size, hidden_size, grid_size, spline_order)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.phi = KANLinear(hidden_size, hidden_size, grid_size, spline_order)
        self.z_norm = nn.LayerNorm(hidden_size)

    def forward(self, x_t, h_prev, z_prev):
        s_t = self.wx(x_t) + self.wh(z_prev)
        o_t = self.phi(s_t)
        z_t = self.z_norm(self.hh(z_prev) + self.hz(o_t))
        return o_t, z_t


class TKANCell(nn.Module):
    """TKAN 单元：多个 RKAN 子层 + LSTM-like 门控。

    输出门由多个 RKAN 子层的 phi 输出拼接后线性映射得到：
      r_t = Concat[phi_1(s_{1,t}), ..., phi_L(s_{L,t})]
      o_t = sigma(W_o r_t + b_o)
      c_t = f_t * c_{t-1} + i_t * g_t
      h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int, sub_layers: int = 3,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.sub_layers = nn.ModuleList([
            RKANSubLayer(input_size, hidden_size, grid_size, spline_order)
            for _ in range(sub_layers)
        ])
        self.f_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_map = nn.Linear(hidden_size * sub_layers, hidden_size)

    def forward(self, x_t, h_prev, c_prev, z_prev_list):
        cat = torch.cat([x_t, h_prev], dim=-1)
        f_t = torch.sigmoid(self.f_gate(cat))
        i_t = torch.sigmoid(self.i_gate(cat))
        g_t = torch.tanh(self.g_gate(cat))

        r_list, z_list = [], []
        for i, sub in enumerate(self.sub_layers):
            r_t, z_t = sub(x_t, h_prev, z_prev_list[i])
            r_list.append(r_t)
            z_list.append(z_t)

        r_t = torch.cat(r_list, dim=-1)
        o_t = torch.sigmoid(self.o_map(r_t))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t, z_list


class TKANLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, sub_layers: int = 3,
                 grid_size: int = 5, spline_order: int = 3,
                 return_sequences: bool = False):
        super().__init__()
        self.cell = TKANCell(input_size, hidden_size, sub_layers,
                             grid_size, spline_order)
        self.hidden_size = hidden_size
        self.sub_layers = sub_layers
        self.return_sequences = return_sequences

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        c = x.new_zeros(B, self.hidden_size)
        z_list = [x.new_zeros(B, self.hidden_size) for _ in range(self.sub_layers)]
        outs = []
        for t in range(T):
            h, c, z_list = self.cell(x[:, t, :], h, c, z_list)
            outs.append(h)
        outs = torch.stack(outs, dim=1)
        return outs if self.return_sequences else outs[:, -1, :]
