import torch.nn as nn

from utils.init import init_wt_unif


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, dropout_p=0):
        super(MyEmbedding, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.drop = nn.Dropout(p=dropout_p)
        init_wt_unif(self.embed.weight)

    def forward(self, x):
        x = self.embed(x)
        x = self.drop(x)
        return x
