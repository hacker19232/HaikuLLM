import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import random

# Hyperparameters
batch_size = 64
block_size = 15
max_iters = 10000
eval_interval = 1000
learning_rate = 1e-2
device = 'cpu'
eval_iters = 200
n_embed = 256
torch.manual_seed(1337)
dropout = 0.2
# Initialize encoding
tokenizer = tiktoken.get_encoding("cl100k_base")

def encode(text):
    enc_text = tokenizer.encode(text)
    unique_tokens = sorted(list(set(enc_text)))
    global var_size, full_to_reduced, reduced_to_full
    var_size = len(unique_tokens)
    print('Vocabulary size (var_size):', var_size)
    
    # Create mapping from full token space to reduced token space
    full_to_reduced = {token: i for i, token in enumerate(unique_tokens)}
    reduced_to_full = {i: token for i, token in enumerate(unique_tokens)}
    
    # Remap the encoded text
    remapped_enc_text = [full_to_reduced[token] for token in enc_text]
    return remapped_enc_text

def decode(encoded_text):
    # Decode using the reduced_to_full mapping
    full_tokens = [reduced_to_full[token] for token in encoded_text]
    return tokenizer.decode(full_tokens)

def somefunc():
    return 0.1

# Read and preprocess text
with open('all_haiku.txt', 'r') as file:
    text = file.read()
    text = text.split("\n")
    random.shuffle(text)
    text = "\n".join(text)

datas = torch.tensor(encode(text), dtype=torch.long)

# Debug: print range of values in datas to ensure they are within bounds
print(f"Data range: {datas.min().item()} to {datas.max().item()}")
print(f"Vocabulary size (var_size): {var_size}")

n = int(0.9 * len(datas))  # 90% for training
train = datas[:n]
test = datas[n:]

def get_batch(split):
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    context = torch.stack([data[i:i+block_size] for i in ix])
    target = torch.stack([data[i+1:i+1+block_size] for i in ix])
    context, target = context.to(device), target.to(device)
    return context, target

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropuout = nn.Dropout(droupout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = torch.bmm(q, k.transpose(1, 2)) / (C**0.5)  # scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return torch.bmm(wei, v)

class MultiHead(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)  # Projection layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.proj(torch.cat([h(x) for h in self.heads], dim=-1))
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.model(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.attention = MultiHead(head_size, n_head)
        self.feedfor = FeedForward()
        self.layer1 = nn.LayerNorm(n_embed)
        self.layer2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.attention(self.layer1(x))
        x = x + self.feedfor(self.layer2(x))
        return x

class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_table = nn.Embedding(var_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=8),
            Block(n_embed, n_head=8),
            Block(n_embed, n_head=8),
        )
        self.lm_head = nn.Linear(n_embed, var_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.embed_table(idx)
        pos_indices = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        pos_emb = self.position_embedding_table(pos_indices)
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_var_size):
        for _ in range(max_var_size):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            char = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, char), dim=1)
        return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
def main():
    model = Bigram().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for i in range(max_iters):
        if i % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {i}: {losses}")

        x, y = get_batch('train')
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    filename = 'llmhaiku.sav'
    pickle.dump(model, open(filename, 'wb'))
