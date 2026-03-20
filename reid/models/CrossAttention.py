import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, num_query=4, embed_dim=512, num_head=8, dropout=0.0):
        '''
        ARGS:
        - num_class: ID num of the current domain
        - num_queries: 4 by default, for head/upper/lower/foot
        - embed_dim: 512 by default, because clip proj_dim = 512, need to calculate the contrastive loss
        - num_head
        - ...

        forward RETURNS:
        - region_feats: (B, num_query, dim)
        '''
        super(CrossAttention, self).__init__()

        assert embed_dim % num_head == 0
        self.num_query = num_query
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.attn_drop = nn.Dropout(dropout)
        self.head_dim = self.embed_dim // self.num_head

        queries = torch.empty(self.num_query, self.embed_dim) # (4, 512)
        nn.init.normal_(queries, std=0.02)
        self.queries = nn.Parameter(queries) 

        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj=  nn.Linear(self.embed_dim,self.embed_dim)
        self.proj_out = nn.Linear(self.embed_dim,self.embed_dim)
        
    def forward(self, label, patch_tokens):

        batch_size, patch, dim = patch_tokens.shape
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1) # (B, 4, 512)


        Q = self.q_proj(queries)
        K = self.k_proj(patch_tokens)
        V = self.v_proj(patch_tokens)
        Qh = self._reshape_heads(Q)              # (B, H, Q, Dh)
        Kh = self._reshape_heads(K)              # (B, H, P, Dh)
        Vh = self._reshape_heads(V)              # (B, H, P, Dh)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1))  # (B, H, Q, P)
        scores = scores / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, Vh)                        # (B, H, Q, Dh)
        out = out.transpose(1, 2).contiguous().view(batch_size, self.num_query, dim)
        region_feats = self.proj_out(out)                  # (B, Q, D)

        attn_avg = attn.mean(dim=1)                        # (B, Q, P

            
        return region_feats, attn_avg

    def _reshape_heads(self, x):
            # (B, L, D) -> (B, H, L, D_head)
            B, L, D = x.shape
            assert self.embed_dim == self.num_head * self.head_dim
            return x.view(B, L, self.num_head, self.head_dim).transpose(1, 2)







