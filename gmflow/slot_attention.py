import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    def __init__(self, num_slots, source , target , iters=3, eps=1e-8, hidden_dim=128, spatial_dim=(32, 32)):  # Added spatial_dim
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        if target is not None:
            self.to_q = nn.Linear(target, target)
        else:
            self.to_q = nn.Linear(Source, Source)
                
                
        self.to_k = nn.Linear(source, source)
        self.to_v = nn.Linear(source, source)

        self.gru = nn.GRUCell(Source, Source)

        hidden_dim = max(Source, hidden_dim)

        self.fc1 = nn.Linear(Source, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, Source)

        self.norm_input = nn.LayerNorm(Source)
        self.norm_slots = nn.LayerNorm(Source)
        self.norm_pre_ff = nn.LayerNorm(Source)

        self.spatial_dim = spatial_dim  # Store spatial dimensions

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _,i in range(self.iters):
            
            slots_prev = slots
            if i == 0 and source != target: 
                q = self.to_k(inputs)
            else:
                slots = self.norm_slots(slots)
                q = self.to_q(slots)


            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))


        return slots
