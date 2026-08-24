import torch

text = """hello world
hello transformer
transformer learns language
"""

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

    def forward(self, x, past_kv=None):
        # Prefill/训练时：x 是 [B, T, C]
        # 使用 KV Cache Decode 时：x 是 [B, 1, C]
        q = x @ self.Wq # [B, 1, C] @ [C, C] = [B, 1, C]
        k = x @ self.Wk
        v = x @ self.Wv

        B, T, C = x.shape
        q = q.reshape(B, T, n_head, head_size)
        k = k.reshape(B, T, n_head, head_size)
        v = v.reshape(B, T, n_head, head_size)
    
        q = q.transpose(1, 2) # 【B, n_head, T, head_size】
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        past_length = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_length = past_k.shape[2]
            k = torch.cat((past_k, k), dim=2) # [B, n_head, past_length + 1, head_size]
            v = torch.cat((past_v, v), dim=2)
    
        scores = q @ k.transpose(-2, -1)
        scores = scores / (head_size ** 0.5)
    
        # mask = torch.tril(torch.ones(T, T))
        if past_kv is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)) # device=x.device确保mask和scores在同一个设备上
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            if T != 1:
                raise ValueError("When past_kv is provided, T must be 1.")
            # 不需要masking，因为每次只生成一个token，scores的shape是[B, n_head, 1, past_length + 1]，只需要计算当前token和之前所有token的注意力分数
    
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
        present_kv = (k, v)
        return out, present_kv

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

    def forward(self, x, past_kv=None):
        xn = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attn_out, present_kv = self.mha.forward(xn, past_kv=past_kv)
        x = x + attn_out

        xn = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        x = x + self.mlp.forward(xn) 

        return x, present_kv

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


def forward(idx, targets=None, past_kvs=None):
    past_length = 0
    if past_kvs is not None:
        if len(past_kvs) != n_layer:
            raise ValueError(f"Expected past_kvs to have length {n_layer}, but got {len(past_kvs)}.")
        past_length = past_kvs[0][0].shape[2]  # 每一层的KV Cache长度应该是一样的

    # Decode阶段T就是1
    B, T = idx.shape

    tok_emb = token_embedding_table[idx]
    # 变成只取最后一个位置
    positions = torch.arange(
        past_length,
        past_length + T,
        device=idx.device,
    )
    if past_length + T > block_size:
        raise ValueError(
            "KV Cache length exceeds block_size."
        )
    pos_emb = position_embedding_table[positions]

    x = tok_emb + pos_emb

    present_kvs = []
    for layer_idx, block in enumerate(blocks):
        layer_past_kv = None
        if past_kvs is not None:
            layer_past_kv = past_kvs[layer_idx]
        x, layer_present_kv = block.forward(x, past_kv=layer_past_kv)
        present_kvs.append(layer_present_kv)

    
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

    return logits, loss, present_kvs

learning_rate = 1e-1

for step in range(10000):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    x, y = get_batch()
    logits, loss, _ = forward(x, y)

    loss.backward()

    for p in params:
        p.data -= learning_rate * p.grad

    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item()}")


def sample_next_token(logits, temperature=1.0, top_k=None):
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        k = min(top_k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, k) # Pytorch API, 返回前k个最大值和索引 [B, k]
        min_top_k_value = top_k_values[:, -1].unsqueeze(-1) # Pytorch API, unsqueeze(-1)表示在最后一维增加一个维度 [B, 1]
        logits = logits.masked_fill(logits < min_top_k_value, float('-inf'))

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1) # Pytorch API, 按照概率分布采样，返回采样的索引 [B, 1]
    return next_token

@torch.inference_mode() # 打上这个装饰器，推理的时候Pytorch就不会计算梯度了，节省计算资源
def generate_without_kv_cache(idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] # 逗号是Pytorch分维度写法，表示取所有行，最后block_size列 [B, block_size]
        logits, _, _ = forward(idx_cond) # [B, T, C]
        logits = logits[:, -1, :]
        next_id = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k
        )
        idx = torch.cat((idx, next_id), dim=1) # Pytorch API, dim=1表示按列拼接
    return idx # [B, T+max_new_tokens]

@torch.inference_mode()
def generate_with_kv_cache(idx, max_new_tokens, temperature=1.0, top_k=None):
    # 由于位置编码是硬编码的，所以 KV Cache 的长度不能超过 block_size，否则位置编码会出错
    # 又由于我没写 KVCache 的 Eviction 策略，比如滑动窗口，所以 KVCache 是无脑增长的
    # 所以这里限制长度，否则 KV Cache 会随着生成超过 block_size
    if idx.shape[1] + max_new_tokens > block_size:
        raise ValueError("The total length of idx and max_new_tokens exceeds block_size.")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    if max_new_tokens == 0:
        return idx

    # Prefill阶段
    logits, _, past_kvs = forward(idx) # [B, T, C]
    next_id = sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        top_k=top_k
    ) # [B, 1]
    idx = torch.cat((idx, next_id), dim=1) # [B, T+1]

    # Decode阶段，每次只有一个向量进入神经网络
    for _ in range(max_new_tokens - 1):
        logits, _, past_kvs = forward(next_id, past_kvs=past_kvs) # [B, 1, C]
        next_id = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k
        )
        idx = torch.cat((idx, next_id), dim=1) # [B, T+2], [B, T+3], ...
    return idx # [B, T+max_new_tokens]


prompt = torch.tensor(
    [[stoi["h"], stoi["e"]]],
    dtype=torch.long,
)

output_without_cache = generate_without_kv_cache(
    prompt.clone(),
    max_new_tokens=6,
    temperature=0,
)

output_with_cache = generate_with_kv_cache(
    prompt.clone(),
    max_new_tokens=6,
    temperature=0,
)

assert torch.equal(
    output_without_cache,
    output_with_cache,
)

print("Full generation KV Cache test passed.")
print("Output:", decode(output_with_cache[0]))