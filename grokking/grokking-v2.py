#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

p = 97
x_idx, y_idx = torch.meshgrid(torch.arange(p, device=device,dtype=torch.long), torch.arange(p, device=device,dtype=torch.long), indexing="ij",) # X and Y coordinates should be long for embedding
X = torch.stack([x_idx.flatten(), y_idx.flatten()], dim=1)
y_inv = torch.pow(X[:, 1], p - 2) % p 
Y = (X[:, 0] * y_inv) % p



f = 0.3
print(X.size(0))
split = int(X.size(0) * f)
indices = torch.randperm(X.size(0), device=device,dtype=torch.long)
X_train, Y_train = X[indices[:split]], Y[indices[:split]]
X_val, Y_val = X[indices[split:]], Y[indices[split:]]

class MyNet(nn.Module):
    def __init__(self, p=97):
        super().__init__()
        self.embedder = nn.Embedding(p, 128)
        # 1. Add Learnable Positional Embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, 2, 128) * 0.02)

        # Keep norm_first=True for stability
        dec_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128,
                                             batch_first=True)
        self.dec = nn.TransformerEncoder(dec_layer, num_layers=2)
        self.to_vocab = nn.Linear(128, p)

    def forward(self, x):
        embeddings = self.embedder(x)
        # 2. Add position info before the Transformer
        embeddings = embeddings + self.pos_emb 
        h = self.dec(embeddings).mean(axis=1)
        return self.to_vocab(h)

model = MyNet(p=p).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
loss_fn = nn.CrossEntropyLoss()

steps = 100000
batch_size = 512
train_len = X_train.size(0)

history = []
pbar = tqdm(range(steps), mininterval=1.0)

model.train()
for step in pbar:
    ix = torch.randint(0, train_len, (batch_size,), device=device)
    x_batch, y_batch = X_train[ix], Y_train[ix]

    optimizer.zero_grad(set_to_none=True)
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch) 
    loss.backward()
    optimizer.step()
    if step<10:
        scheduler.step()
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(dim=1) == Y_val).float().mean().item()
            train_acc = (logits.argmax(dim=1) == y_batch).float().mean().item()
            if val_acc>0.99:
                print("GROCk")
                break

            history.append([step, loss.item(), train_acc, val_acc])
            pbar.set_description(f"Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Train Acc : {train_acc:.4f}")
        model.train()


# In[2]:


ckpt = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
}
torch.save(ckpt,  "./checkpoint.pt")


# In[3]:


with open(os.path.join("./history.json"), "w") as f:
    json.dump({"history": history}, f)

step, loss, train_acc, val_acc = zip(*history)
plt.figure(figsize=(6, 4))
plt.plot(step, train_acc, label="train_acc")
plt.plot(step, val_acc, label="val_acc")
plt.grid(True)
plt.legend()
acc_path = os.path.join("./acc.png")
plt.savefig(acc_path, dpi=200, bbox_inches="tight")
plt.close()


# In[4]:


p=97

W = model.to_vocab.weight.detach().cpu().float().numpy()

W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=42,
)
Z = tsne.fit_transform(W)  # [p, 2]

x, y = Z[:, 0], Z[:, 1]
colors = (np.arange(p) % 8)

plt.figure(figsize=(8, 7))
sc = plt.scatter(x, y, c=colors, s=35)
plt.axis("off")

for i in range(p):
    j = (i + 8) % p
    plt.plot([x[i], x[j]], [y[i], y[j]], linewidth=0.6, alpha=0.5)

for i in range(p):
    plt.text(x[i], y[i], str(i), fontsize=8, ha="center", va="center")

plt.title(f"t-SNE of output weights (V={p}), lines: +{8} mod {p}, colors: mod 8")
plt.savefig("./tsne.png", dpi=200, bbox_inches="tight")
plt.close()

Z


# In[ ]:





# In[ ]:




