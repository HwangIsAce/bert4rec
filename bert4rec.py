import torch
from torch import nn, einsum
from einops import rearrange
import numpy as np
from torch.optim import Adam

from dataloader import MyDataLoader
from constants import TRAIN_CONSTANTS

def FeedForward(dim):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim*4),
        nn.GELU(),
        nn.Linear(dim*4, dim)
    )

class Attention(nn.Module):
    def __init__(
            self,
            emb_dim,
            heads,
    ):
        super().__init__()
        self.heads = heads
        self.multi_head_dim = emb_dim // heads
        self.norm = nn.LayerNorm(emb_dim)
        self.to_qkv = nn.Linear(emb_dim, self.multi_head_dim *3)
        self.scale = emb_dim**-0.5
        self.softmax = nn.Softmax(dim = -1)

        self.to_out = nn.Linear(self.multi_head_dim, emb_dim)

    def forward(self, x):  # bias? # dropout?
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q,k,v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q,k)
        attn_map = self.softmax(sim)

        out = einsum('b h i j, b h j d -> b h i d', attn_map, v) 

        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        out = self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        emb_dim,
        heads,
        depth= 2,
    ):
        super().__init__()

        self.layer = nn.ModuleList([])

        for _ in range(depth):
            self.layer.append(nn.ModuleList([
                Attention(emb_dim, heads),
                FeedForward(emb_dim)
            ]))

    def forward(self, x):
        
        for attn, ffn in self.layer:
            attn_out = attn(x)

            x = x + attn_out
            x = x + ffn(x)

        return x

class BERT4REC(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim,
            max_len,
            heads
    ):
        super().__init__()

        self.max_len = max_len

        self.item_embedder = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedder = nn.Embedding(vocab_size,emb_dim)   # random 

        self.transformer = Transformer(emb_dim, heads)


    def forward(self, x):
        # if x.shape[1] > self.max_len :
        #     x = x[:, :self.max_len]
        
        # item embedding
        x = self.item_embedder(x)

        # positional embedding
        positional_tensor = (torch.range(1, x.shape[1]) * torch.ones(x.shape[0], x.shape[1])).long()
        positional_embedding = self.positional_embedder(positional_tensor)
        

        x = x + positional_embedding 

        # transformer
        x = self.transformer(x)

        return x


if __name__ == "__main__":

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    train_constants = TRAIN_CONSTANTS()

    train_loader, valid_loader, test_loader = MyDataLoader()


    model = BERT4REC(train_constants.VOCAB_SIZE, 
                     train_constants.EMB_DIM, 
                     train_constants.MAX_LEN,
                     heads=8)

    def train(model: nn.Module, epoch):
        model.train()

        total_loss = 0.0
        total_iter = 0
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        cross_entropy_loss = nn.CrossEntropyLoss()

        for epoch, batch in enumerate(train_loader):

            input_ids = batch['input_ids'].long()
            labels = batch['labels'].long()
            
            logits = model(input_ids) ## test

            loss = cross_entropy_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iter += 1

        mean_loss = total_loss / total_iter
        print(f"epoch: {epoch+1}, loss: {mean_loss: 1.4f}")
        
    for epoch in range(10):
        train(model, epoch)

    PATH = '/home/jaesung/jaesung/study/bert4rec/output/test.pth'
    torch.save(model.state_dict(), PATH)

        