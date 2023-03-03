import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, padding_idx, smoothing=0.0, vocab_size=None):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor, vocab_size: int = None):
        """Basic idea: Use target to build true_dist, and calculate KL divergence with the original x. Each position in the generated sequence

        Args:
            x (torch.Tensor): [bsz, seq_len, vocab]
            target (torch.Tensor): [bsz, seq_len]
            vocab_size: copynet中可能出现动态词表

        Returns:
            [type]: [description]
        """
        if vocab_size is None:
            assert (
                x.size(-1) == self.vocab_size
            ), f"input x shape: {x.shape} last dim is not equal to {self.vocab_size}."
            vocab_size = self.vocab_size
        else:
            assert (
                x.size(-1) == vocab_size
            ), f"input x shape: {x.shape} last dim is not equal to {vocab_size}."

        # reshape to [bsz * seq_len , *]
        x = x.view(-1, vocab_size)
        target = target.view(-1)

        true_dist = x.data.clone()
        # padding and itself are not included, the rest of the vocabulary is filled
        true_dist.fill_(self.smoothing / (vocab_size - 2))
        # The true word in the answer is assigned 1-ε
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())  # return KL divergence between x and processed target


if __name__ == "__main__":
    bsz = 5
    seq_len = 11
    vocab = 30
    x = torch.rand((bsz, seq_len, vocab))
    target = torch.randint(0, vocab, (bsz, seq_len))
    label = LabelSmoothing(padding_idx=0, smoothing=0.2, vocab_size=vocab)
    res = label(x, target)
    _x = 1
