import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim, num_devices, dropout=0.1, use_value_head=False):
        super().__init__()
        hidden_dim = state_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.logits = nn.Linear(hidden_dim, num_devices)
        self.use_value_head = use_value_head

        if self.use_value_head:
            self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, mask=None):
        x = F.relu(self.norm1(self.fc1(state)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)

        logits = self.logits(x)

        # Mask invalid actions
        if mask is not None:
            mask = mask.to(torch.bool)
            logits = logits.masked_fill(~mask, -1e9)

        probs = F.softmax(logits, dim=-1)

        if self.use_value_head:
            value = self.value_head(x)
            return probs, logits, value

        return probs, logits
