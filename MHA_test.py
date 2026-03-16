import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.att_dropout = nn.Dropout(attention_dropout)

    def forward(self, X, attention_mask=None):
        
        batch, seq_len, _ = X.size()

        # QKV
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # 分头
        q_state = Q.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k_state = K.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        v_state = V.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # 权重
        attention_weight = torch.matmul(
            q_state, k_state.transpose(-1, -2)
        ) // math.sqrt(self.head_dim)
        
        # mask
        if attention_mask is not None:
            attention_weight.masked_fill(
                attention_mask == 0, float("-1e20")
            )

        # 注意力分数
        score = torch.softmax(attention_weight, -1)

        # dropout
        score = self.att_dropout(score)

        # 乘V
        out_mid = torch.matmul(
            score, v_state
        )

        # 维度
        # b head_num s head_dim
        out_mid = out_mid.transpose(1, 2).contiguous()
        
        out_mid = out_mid.view(batch, seq_len, -1)

        out_put = self.out_proj(out_mid)

        return out_put


x = torch.rand(3, 2, 128)
net = MultiHeadAttention(128, 8)

print(net(x, None).shape)

# attention_mask = (
#     torch.tensor(
#         [
#             [0, 1],
#             [0, 0],
#             [1, 0],
#         ]
#     )
#     .unsqueeze(1)
#     .unsqueeze(2)
#     .expand(3, 8, 2, 2)
# )

# x = torch.rand(3, 2, 128)
# net = MultiHeadAttention(128, 8)
# print(net(x, attention_mask).shape)
