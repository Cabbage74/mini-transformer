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

# x = encode("hello")
# print(x)
# print(decode(x))

data = torch.tensor(encode(text), dtype=torch.long)

# print(data)
# print(data.shape)
# print(data.dtype)

batch_size = 4
block_size = 8

def get_batch():
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

x, y = get_batch()
# print("x:", x)
# print("y:", y)

n_embd = 32

token_embedding_table = torch.randn(vocab_size, n_embd) * 0.02
token_embedding_table.requires_grad_(True)

tok_emb = token_embedding_table[x]
# print("tok_emb:", tok_emb)

position_embedding_table = torch.randn(block_size, n_embd) * 0.02
position_embedding_table.requires_grad_(True)

positions = torch.arange(block_size)

pos_emb = position_embedding_table[positions]
# print("pos_emb:", pos_emb)

x = tok_emb + pos_emb
# print("x:", x)
# print("x.shape:", x.shape)

# Single Head Self Attention
# head_size = 16

# # x
# # ├── Wq → Q
# # ├── Wk → K
# # └── Wv → V
# Wq = torch.randn(n_embd, head_size) * 0.02
# Wq.requires_grad_(True)
# Wk = torch.randn(n_embd, head_size) * 0.02
# Wk.requires_grad_(True)
# Wv = torch.randn(n_embd, head_size) * 0.02
# Wv.requires_grad_(True)

# q = x @ Wq
# k = x @ Wk
# v = x @ Wv

# scores = q @ k.transpose(-2, -1)

# #               K
# #            I   love playing games
# # Q   I      ?     ?      ?      ?
# #    love    ?     ?      ?      ?
# #  playing   ?     ?      ?      ?
# #  games     ?     ?      ?      ?

# # row 2, col 0表示playing 这个位置的 Query，和 I 这个位置的 Key 匹配程度如何

# scores = scores / (head_size ** 0.5)

# mask = torch.tril(torch.ones(block_size, block_size))

# for b in range(batch_size):
#     scores[b][mask == 0] = float('-inf')

# scores = scores - scores.max(dim=-1, keepdim=True).values
# exp_scores = torch.exp(scores)
# weights = exp_scores / exp_scores.sum(
#     dim = -1,
#     keepdim = True
# )

# out = weights @ v

# print("out:", out)
# print("out.shape:", out.shape)

n_head = 4
head_size = n_embd // n_head

W_q = torch.randn(n_embd, n_embd) * 0.02
W_q.requires_grad_(True)
W_k = torch.randn(n_embd, n_embd) * 0.02
W_k.requires_grad_(True)
W_v = torch.randn(n_embd, n_embd) * 0.02
W_v.requires_grad_(True)
Wo = torch.randn(n_embd, n_embd) * 0.02
Wo.requires_grad_(True)

def multi_head_attention(x):
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v

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

    out = out @ Wo
    return out

    # print("q:", q.shape)
    # print("scores:", scores.shape)
    # print("weights:", weights.shape)
    # print("attention out:", out.shape)

    # print(weights[0, 0])

ln1_gamma = torch.ones(n_embd)
ln1_gamma.requires_grad_(True)
ln1_beta = torch.zeros(n_embd)
ln1_beta.requires_grad_(True)

def layer_norm(x, gamma, beta):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + 1e-5)
    return gamma * x_hat + beta

residual = x
xn = layer_norm(x, ln1_gamma, ln1_beta)
attn_out = multi_head_attention(xn)
x = residual + attn_out

print("x:", x)
print("x.shape:", x.shape)

hidden_size = 4 * n_embd

W1 = torch.randn(n_embd, hidden_size) * 0.02
W1.requires_grad_(True)
b1 = torch.zeros(hidden_size)
b1.requires_grad_(True)

W2 = torch.randn(hidden_size, n_embd) * 0.02
W2.requires_grad_(True)
b2 = torch.zeros(n_embd)
b2.requires_grad_(True)

ln2_gamma = torch.ones(n_embd)
ln2_beta = torch.zeros(n_embd)

ln2_gamma.requires_grad_(True)
ln2_beta.requires_grad_(True)

def mlp(x):
    h = x @ W1 + b1
    h = h * (h > 0)  # ReLU activation
    out = h @ W2 + b2
    return out

residual = x
xn = layer_norm(x, ln2_gamma, ln2_beta)
mlp_out = mlp(xn)
x = residual + mlp_out


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

x = tok_emb + pos_emb
for block in blocks:
    x = block.forward(x)

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

x = layer_norm(x, ln_f_gamma, ln_f_beta)


logits = x @ W_lm + b_lm

shifted = logits - logits.max(dim=-1, keepdim=True).values

exp_logits = torch.exp(shifted)

probs = exp_logits / exp_logits.sum(
    dim=-1,
    keepdim=True
)

correct_probs = []
B, T = y.shape
for b in range(B):
    for t in range(T):
        target_id = y[b, t]
        correct_probs.append(probs[b, t, target_id])
correct_probs = torch.tensor(correct_probs)

loss = -torch.log(correct_probs).mean()
print("loss:", loss)

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

x, y = get_batch()
logits, loss = forward(x, y)

learning_rate = 1e-3

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
        idx_cond = idx[:, -block_size:]
        logits, _ = forward(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.argmax(
            probs,
            dim=-1,
            keepdim=True
        )
        idx = torch.cat((idx, next_id), dim=1)
    return idx

idx = torch.tensor([[stoi['h']]], dtype=torch.long)
output_idx = generate(idx, max_new_tokens=20)
output_text = decode(output_idx[0].tolist())
print("Generated text:", output_text)