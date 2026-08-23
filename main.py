import torch

text = """hello world
hello transformer
transformer learns language
""" * 100

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [stoi[ch] for ch in s]

def decode(l: list[int]) -> str:
    return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

batch_size = 4
block_size = 8

def get_batch():
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

n_embd = 32

token_embedding_table = torch.randn(vocab_size, n_embd) * 0.02
token_embedding_table.requires_grad_(True)


position_embedding_table = torch.randn(block_size, n_embd) * 0.02
position_embedding_table.requires_grad_(True)

n_head = 4
head_size = n_embd // n_head # 32 // 4 = 8
hidden_size = 4 * n_embd # 4 * 32 = 128

def layer_norm(x, gamma, beta):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + 1e-5)
    return gamma * x_hat + beta


class MultiHeadAttention:
    def __init__(self):
        self.Wq = torch.randn(n_embd, n_embd) * 0.02
        self.Wk = torch.randn(n_embd, n_embd) * 0.02
        self.Wv = torch.randn(n_embd, n_embd) * 0.02
        self.Wo = torch.randn(n_embd, n_embd) * 0.02

        self.Wq.requires_grad_(True)
        self.Wk.requires_grad_(True)
        self.Wv.requires_grad_(True)
        self.Wo.requires_grad_(True)

    def parameters(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]

    def forward(self, x):
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        B, T, C = x.shape
        q = q.reshape(B, T, n_head, head_size)
        k = k.reshape(B, T, n_head, head_size)
        v = v.reshape(B, T, n_head, head_size)
    
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    
        scores = q @ k.transpose(-2, -1)
        scores = scores / (head_size ** 0.5)
    
        mask = torch.tril(torch.ones(T, T))
    
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
        scores = scores - scores.max(dim=-1, keepdim=True).values
    
        exp_scores = torch.exp(scores)
    
        weights = exp_scores / exp_scores.sum(
            dim=-1,
            keepdim=True
        )
    
        out = weights @ v
    
        out = out.transpose(1, 2)
        out = out.reshape(B, T, C)
    
        out = out @ self.Wo
        return out

class MLP:
    def __init__(self):
        self.W1 = torch.randn(n_embd, hidden_size) * 0.02
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, n_embd) * 0.02
        self.b2 = torch.zeros(n_embd)

        self.W1.requires_grad_(True)
        self.b1.requires_grad_(True)
        self.W2.requires_grad_(True)
        self.b2.requires_grad_(True)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = h * (h > 0)  # ReLU activation
        out = h @ self.W2 + self.b2
        return out

class Block:
    def __init__(self):
        self.ln1_gamma = torch.ones(n_embd)
        self.ln1_beta = torch.zeros(n_embd)
        self.ln2_gamma = torch.ones(n_embd)
        self.ln2_beta = torch.zeros(n_embd)

        self.ln1_gamma.requires_grad_(True)
        self.ln1_beta.requires_grad_(True)
        self.ln2_gamma.requires_grad_(True)
        self.ln2_beta.requires_grad_(True)

        self.mha = MultiHeadAttention()
        self.mlp = MLP()

    def parameters(self):
        return [self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta] + self.mha.parameters() + self.mlp.parameters()

    def forward(self, x):
        xn = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        x = x + self.mha.forward(xn)

        xn = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        x = x + self.mlp.forward(xn) 

        return x

n_layer = 3
ln_f_gamma = torch.ones(n_embd)
ln_f_beta = torch.zeros(n_embd)

ln_f_gamma.requires_grad_(True)
ln_f_beta.requires_grad_(True)

W_lm = torch.randn(n_embd, vocab_size) * 0.02
b_lm = torch.zeros(vocab_size)

W_lm.requires_grad_(True)
b_lm.requires_grad_(True)

blocks = [Block() for _ in range(n_layer)]

params = [
    token_embedding_table,
    position_embedding_table,
    ln_f_gamma,
    ln_f_beta,
    W_lm,
    b_lm
]
for block in blocks:
    params += block.parameters()

num_params = sum(p.numel() for p in params)
print("Number of parameters:", num_params)


def forward(idx, targets=None):
    B, T = idx.shape

    tok_emb = token_embedding_table[idx]
    positions = torch.arange(T)
    pos_emb = position_embedding_table[positions]

    x = tok_emb + pos_emb

    for block in blocks:
        x = block.forward(x)

    x = layer_norm(x, ln_f_gamma, ln_f_beta)

    logits = x @ W_lm + b_lm

    if targets is None:
        loss = None
    else:
        shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
        exp_logits = torch.exp(shifted_logits)
        probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

        correct_probs = []
        for b in range(B):
            for t in range(T):
                target_id = targets[b, t]
                correct_probs.append(probs[b, t, target_id])
        correct_probs = torch.stack(correct_probs)

        loss = -torch.log(correct_probs).mean()

    return logits, loss

learning_rate = 1e-1

for step in range(10000):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    x, y = get_batch()
    logits, loss = forward(x, y)

    loss.backward()

    for p in params:
        p.data -= learning_rate * p.grad

    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item()}")


def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] # 逗号是Pytorch分维度写法，表示取所有行，最后block_size列
        logits, _ = forward(idx_cond) # [B, T, C]
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.argmax( 
            probs,
            dim=-1,
            keepdim=True
        )
        idx = torch.cat((idx, next_id), dim=1) # Pytorch API, dim=1表示按列拼接
    return idx # [B, T+max_new_tokens]

idx = torch.tensor([[stoi['h']]], dtype=torch.long) # 为了复用函数，加上Batch那一维
output_idx = generate(idx, max_new_tokens=20)
output_text = decode(output_idx[0].tolist())
print("Generated text:", output_text)