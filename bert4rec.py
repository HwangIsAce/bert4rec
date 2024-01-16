import torch
from torch import nn

from dataloader import MyDataLoader

class BERT4REC(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":

    train_loader, valid_loader, test_loader = MyDataLoader()


    model = BERT4REC()

    for i, batch in enumerate(train_loader):
        
        tt = model(batch) ## test 
        import IPython; IPython.embed(colors="Linux"); exit(1)
