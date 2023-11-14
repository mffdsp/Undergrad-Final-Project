import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtext.transforms import SentencePieceTokenizer
from torch.hub import load_state_dict_from_url

DATA_PATH = "./data/data.txt"

# hyperparameters
block_size = 256  # what is the maximum context length for predictions?
batch_size = 64
max_iters = 8000
eval_interval = 1000
learning_rate = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
weight_decay = 1e-6

# regularization
# batch_size = 128 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 1000
# eval_interval = 100
# learning_rate = 3e-4

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# eval_iters = 200
# n_embd = 384
# n_head = 8
# n_layer = 6
# dropout = 0.1

# # hyperparameters
# batch_size = 4 # how many independent sequences will we process in parallel?
# block_size = 2 # what is the maximum context length for predictions?
# max_iters = 5
# eval_interval = 2
# learning_rate = 3e-4

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# eval_iters = 2
# n_embd = 4
# n_head = 2
# n_layer = 2
# dropout = 0.2
# # ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# # here are all the unique characters that occur in this text
chars = sorted(list(set(text)))


# def data_augmentation(text):
#   augmented_text = []

#   for token in text:
#     # With probability 50%, replace the token with a random token.
#     if random.random() < 0.5:
#       augmented_text.append(random.choice(chars))
#     else:
#       augmented_text.append(token)

#   # With probability 10%, insert a random token into the text.
#   for _ in range(int(0.1 * len(text))):
#     index = random.randint(0, len(augmented_text))
#     augmented_text.insert(index, random.choice(chars))

#   # With probability 10%, remove a random token from the text.
#   for _ in range(int(0.1 * len(text))):
#     index = random.randint(0, len(augmented_text) - 1)
#     augmented_text.pop(index)

#   return ''.join(augmented_text)


# augmented_text = data_augmentation(text)
# print(f"BeforeAugmentation = {len(text)//1e6}M characteres.\AfterAugmentation = {len(augmented_text)//1e6}M characteres.")

vocab_size = len(chars)

# create a mapping from characters to integers (Replace with tiktoken) https://github.com/openai/tiktoken
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string


# xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"

# vocab = torch.load('xlmr.vocab.pt')
# vocab_size = len(vocab)

# def tokenize_text(text):
#     return nltk.word_tokenize(text)

# def encode(s):
#     tokens = tokenize_text(s)
#     return [vocab[token] for token in tokens]

# def decode(l):
#     tokens = [key for key, value in vocab.items() if value in l]
#     return ' '.join(tokens)

# encode(text)


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GenerativeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # Apply log_softmax to logits for better numerical stability
            log_probs = F.log_softmax(logits, dim=-1)

            # Use negative log-likelihood loss
            loss = F.nll_loss(log_probs, targets)

            # # cross_entropy
            # loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GenerativeLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

for iter in range(max_iters):
    print(f"Epoch {iter+1}", end="\r")

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()

        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Texto de entrada
input_text = "Hi Junior, i need some "

# Tokenize o texto de entrada
# input_tokens = encode(input_text)

# print(input_tokens)
# # Converta os tokens para um tensor do Torch
# input_tensor = decode(input_tokens)
# print(input_tensor)

context = torch.tensor(encode(input_text), dtype=torch.long, device=device).unsqueeze(0)

import pickle

# save the iris classification model as a pickle file
model_pkl_file = "test.pkl"

torch.save(model.state_dict(), "GenerativeLanguageModel_TEST.pt")

with open(model_pkl_file, "wb") as file:
    pickle.dump(model, file)

m = model.to(device)

# evaluate model
y_predict = decode(m.generate(context, max_new_tokens=500)[0].tolist())

print(y_predict)
