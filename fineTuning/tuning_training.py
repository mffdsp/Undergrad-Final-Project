import json
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch.nn.functional as F

file_path = "./fineTuning_data.json"
model_path = "../model_5.pt"
DATA_PATH = "../data/data.txt"
EPOCHS = 8000
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def import_json_from_file(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

##############################
##############################


# Train and test splits
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 6000
eval_interval = 250
learning_rate = 1e-4

eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
weight_decay = 1e-6

training_data = import_json_from_file(file_path)
training_data = training_data[:8000]


def import_json_from_file(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


tun_encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
tun_decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

max_length = max(
    len(tun_encode(email["subject"] + " " + email["body"])) for email in training_data
)

# Pré-processamento dos dados e codificação usando encoders personalizados
encoded_inputs = []
print(f"Training with {len(training_data)} mails")

text_input = ""

i = 1
for email in training_data:
    text_input += email["subject"] + "\n" + email["body"] + "\n"

    if i % 1000 == 0:
        print(f"LOADED {i} e-mails so far...", end="\r")
    i += 1
    # encoded_input = tun_encode(email['subject'] + '\n' + email['body'])

    # # Preencher a sequência com zeros até o comprimento máximo
    # padded_input = encoded_input + [0] * (max_length - len(encoded_input))

    # encoded_inputs.extend(padded_input)
    # print(i)
    # i += 1

data = torch.tensor(tun_encode(text_input), dtype=torch.long)
n = int(0.8 * len(data))  # first 80% will be train, rest val
train_data = data[:n]
val_data = data[n:]


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

        # compute attention scores ("affinities") weight aggregation of the past
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
            nn.SILU(),
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

            # # Pesquisas cross_entropy
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


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

model = GenerativeLanguageModel()

## Load no modelo pre-treinado
model.load_state_dict(torch.load(model_path))
model = model.to(device)

layers_to_freeze = ["token_embedding_table", "position_embedding_table"]

for name, param in model.named_parameters():
    if any(l_name in name for l_name in layers_to_freeze):
        param.requires_grad = False
        print("Freezing ", name, " layer.")

input_tensors = torch.tensor(encoded_inputs).to(device)

# https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Realizar o fine-tuning
model.train()
print("\n---")
for epoch in range(EPOCHS):  # número de épocas de treinamento
    print(f"Epoch {epoch+1}", end="\r")

    if epoch % 800 == 0:
        losses = estimate_loss()
        print(f"*Train loss {losses['train']:.4f} *Val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model_after_tuning.pt")


# Função para gerar texto usando o modelo
def generate_text(model, seed_text, max_length):
    model.eval()  # Defina o modelo para o modo de avaliação
    input_ids = torch.tensor(
        tun_encode(seed_text), dtype=torch.long, device=device
    ).unsqueeze(0)

    # Gere o texto
    generated_ids = model.generate(input_ids, max_length)
    generated_text = tun_decode(generated_ids.squeeze().tolist())

    return generated_text


# Exemplo de uso
seed_text = "Important Announcement: Office Closure Dear team"
max_length = 200
generated_text = generate_text(model, seed_text, max_length)[:]

print("Texto gerado:")
print(generated_text)
