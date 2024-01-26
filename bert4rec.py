import torch
from torch import nn, einsum
from einops import rearrange

from dataloader import MyDataLoader
from constants import TRAIN_CONSTANTS


class FeedForward(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(train_constants.EMB_DIM)
    
    def forward(self, x):
        x = self.norm(x)


class Attention(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim,
            heads,
    ):
        super().__init__()
        self.heads = heads
        self.multi_head_dim = emb_dim // heads
        self.norm = nn.LayerNorm(emb_dim)
        self.to_qkv = nn.Linear(emb_dim, emb_dim *3)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q,k,v))

        sim = einsum('b h i d, b h j d -> b h i j', q,k)
        import IPython; IPython.embed(colors="Linux"); exit(1)
        


    
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        heads,
        num_layers= 2,
    ):
        super().__init__()

        self.layer = nn.ModuleList([])

        for _ in range(num_layers):
            self.layer.append(nn.ModuleList([
                Attention(vocab_size, emb_dim, heads),
                FeedForward()
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
        self.positional_embedder = nn.Embedding(max_len+1,emb_dim)

        self.transformer = Transformer(vocab_size, emb_dim, heads)

    def forward(self, x):
        if x.shape[1] > self.max_len :
            x = x[:, :self.max_len]
        
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

    for i, batch in enumerate(train_loader):

        input = batch['input_ids'].long()
        
        tt = model(input) ## test 
