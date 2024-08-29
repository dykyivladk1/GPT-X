import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
    


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, attn_pdrop,
                 resid_pdrop, block_size):
        super(CausalSelfAttention, self).__init__()


        self.attn_proj = nn.Linear(n_embed, n_embed * 3)
        self.fin_proj = nn.Linear(n_embed, n_embed)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)


        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))
        self.n_head = n_head
        self.n_embed = n_embed


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dimensions = x.size()

        queries, keys, values = self.attn_proj(x).split(self.n_embed, -1)


        queries = queries.view(batch_size, seq_len, self.n_head, dimensions // self.n_head)
        keys = keys.view(batch_size, seq_len, self.n_head, dimensions // self.n_head)
        values = values.view(batch_size, seq_len, self.n_head, dimensions // self.n_head)

        queries = torch.transpose(queries, 1, 2)
        keys = torch.transpose(keys, 1, 2)
        values = torch.transpose(values, 1, 2)

        sim_scores = torch.matmul(queries, keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))

        sim_scores = sim_scores.masked_fill_(mask = self.bias[:, :, :seq_len, :seq_len] == 0, value = float('-inf'))

        sim_scores = torch.softmax(sim_scores, dim = -1)    
        sim_scores = self.attn_drop(sim_scores)

        attention = torch.matmul(sim_scores, values)
        attention = torch.transpose(attention, 1, 2)
        attention = attention.contiguous().view(batch_size, seq_len, dimensions)
        attention = self.resid_drop(attention)
        attention = self.fin_proj(attention)

        return attention
    





class Block(nn.Module):
    def __init__(self, n_embed, n_head, resid_pdrop, attn_pdrop, block_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(n_embed),
            CausalSelfAttention(n_embed, n_head, attn_pdrop, resid_pdrop, block_size),
            nn.Dropout(resid_pdrop)
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_embed),
            nn.Linear(n_embed, 4 * n_embed),
            GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(resid_pdrop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
    



class GPT_X(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer,
                 n_head, n_embed, embedding_pdrop = 0.1, resid_pdrop = 0.1,
                 attn_pdrop = 0.1):
        super(GPT_X, self).__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed),
            wpe = nn.Embedding(block_size, n_embed),
            drop = nn.Dropout(embedding_pdrop),
            h = nn.ModuleList([Block(n_embed, n_head, resid_pdrop, attn_pdrop, block_size) for _ in range(n_layer)]),
            ln = nn.LayerNorm(n_embed)
        ))
        self.block_size = block_size

        self.lm_head = nn.Linear(n_embed, vocab_size, bias = False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)
        
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor = None):
        device, seq_len = x.device, x.size(1)

        positions = torch.arange(0, seq_len, dtype = torch.long, device = device).unsqueeze(0)
        token_embeds = self.transformer.wte(x)
        position_embeds = self.transformer.wpe(positions)
        x = self.transformer.drop(token_embeds + position_embeds)
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1) if target is not None else None
        return logits, loss
    

    def set_optimizers_(self, learning_rate = float, betas = tuple, weight_decay = float):

        decay = set()
        no_decay = set()


        need_, no_need_ = (nn.Linear,), (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, need_):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, no_need_):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}


        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, greedy_search=False, top_k=None, do_sample=True):
        for _ in range(max_new_tokens):
            input_sequence = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(input_sequence)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                threshold = torch.topk(logits, top_k)[0][:, -1, None]
                logits = torch.where(logits < threshold, torch.full_like(logits, -float('Inf')), logits)

            probabilities = F.softmax(logits, dim=-1)
            
            if do_sample:
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(probabilities, dim=-1, keepdim=True)

            idx = torch.cat([idx, next_token], dim=1)

        return idx
