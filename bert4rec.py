import torch
from torch import nn

from dataloader import MyDataLoader
from constants import TRAIN_CONSTANTS
 

class BERT4REC(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim,
            max_len
    ):
        super().__init__()

        self.mex_len = max_len

        self.item_embedder = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedder = nn.Embedding(max_len+1,emb_dim)

    def forward(self, x):

        # item embedding
        x = self.item_embedder(x)

        # positional embedding
        positional_tensor = (torch.range(1, x.shape[1]) * torch.ones(x.shape[0], x.shape[1])).long()
        positional_embedding = self.positional_embedder(positional_tensor)

        x = x + positional_embedding 
        
        return x


if __name__ == "__main__":

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    train_constants = TRAIN_CONSTANTS()

    train_loader, valid_loader, test_loader = MyDataLoader()

    model = BERT4REC(train_constants.VOCAB_SIZE, 
                     train_constants.EMB_DIM, 
                     train_constants.MAX_LEN)

    for i, batch in enumerate(train_loader):

        input = batch['input_ids'].long()
        
        tt = model(input) ## test 
