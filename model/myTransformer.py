import math
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        input
            memory: [bsz, src_len, dim]
            src_mask: [bsz, 1, src_len]
            tgt: [bsz, tgt_len, dim]
            tgt_mask: [bsz, tgt_len, tgt_len]
        return
            hidden_states: [bsz, tgt_len, dim]
            prob: [bsz, tgt_len, tgt_vocab]
        """
        hidden_states = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        prob = self.generator(hidden_states)
        return hidden_states, prob


# -------------------- encoder -------------------- #


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x: [bsz, seq_len, hid]
        mean = x.mean(-1, keepdim=True)  # [bsz, seq_len, 1]
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # 注意，x进来直接先LayerNorm
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_conn = ResidualConnection(size, dropout)
        self.feed_conn = ResidualConnection(size, dropout)

        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # 在res-conn中，先layernorm一下，再进lambda x，然后dropout，最后+x
        x = self.res_conn(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.feed_conn(x, self.feed_forward)
        return x


def multi_head_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention' 矩阵和矩阵之间计算相似度"
    # q,k,v: [bsz, 8, seq_len, 64]
    d_k = query.size(-1)
    # score 不一定是方形!! 最后用的是矩阵乘法 value 的size是tgt！
    # [bsz, 8, src_len, 64] @ [bsz, 8, 64, tgt_len] = [bsz, 8, src_len, tgt_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, tiny_value_of_dtype(scores.dtype))
    # 每一行 softmax
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 重点！！[bsz, 8, seq_len, seq_len] @ [bsz, 8, seq_len, 64] = [bsz, 8, seq_len, 64]
    # 矩阵视角：p_attn的一行和value的每列计算内积，得到目标矩阵的一行，即一个单词由其他单词的加权表示
    # 若p_attn是下三角为1的矩阵，在矩阵乘法上，表示为单词i之后的单词权重为0，这就是「看不见后面单词」的本质
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        bsz = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # [bsz, 8, seq_len, 64] 8头注意力机制，8*64=512
        query = self.q_proj(query).view(bsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.k_proj(key).view(bsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.v_proj(value).view(bsz, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = multi_head_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x: [bsz, 8, seq_len, 64] -> [bsz, seq_len, 8, 64] -> [bsz, seq_len, hid]
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.h * self.d_k)
        return self.linear_out(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, padding_idx=0):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.d_model = embedding_dim

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 先取log再exp
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # This is typically used to register a buffer that should not to be considered a model parameter.
        # For example, BatchNorm’s running_mean is not a parameter, but is part of the module’s state.
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 传入的x已经是embedding矩阵
        x = x + self.pe[:, : x.size(1)].detach()
        return self.dropout(x)


# -------------------- decoder -------------------- #


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: [bsz, tgt_len, dim]
        memory: [bsz, src_len, dim]
        src_mask: [bsz, 1, src_len]
        tgt_mask: [bsz, src_len, src_len]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, mem_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        # 注意，和 memory 做attn
        self.mem_attn = mem_attn
        self.feed_forward = feed_forward

        self.res_conn_selfattn = ResidualConnection(size, dropout)
        self.res_conn_srcattn = ResidualConnection(size, dropout)
        self.res_conn_feed = ResidualConnection(size, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Follow Figure 1 (right) for connections.
        注意，这里的src_attn，以x为Q，m为K, V。
        """
        # 在res-conn中，先layernorm一下，再进lambda x，然后dropout，最后+x
        x = self.res_conn_selfattn(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.res_conn_srcattn(x, lambda x: self.mem_attn(x, memory, memory, src_mask))
        return self.res_conn_feed(x, self.feed_forward)


# -------------------- helper -------------------- #


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def subsequent_mask(seq_len):
    "Mask out subsequent positions."
    attn_shape = (1, seq_len, seq_len)
    # 如果 diagonal ==0 保留主对角线以及右上角的值，为1则不保留主对角线的值，为2则对角线上1斜线的值也为0.
    # 包含对角线的下三角为1的矩阵，表示只能看到当前以及之前的词。
    triu_mask = torch.triu(torch.ones(attn_shape), diagonal=1) == 0
    return triu_mask


def make_triu_mask_batch(tgt, pad=0):
    """
    Create a mask to hide padding and future words.
    tgt: [bsz, tgt_len]
    pad: 0
    """
    # tgt: [bsz, seq_len-1]
    tgt_mask = (tgt != pad).unsqueeze(-2)
    # [bsz, seq_len-1] & 包含对角线的下三角为1的矩阵 = [bsz, seq_len-1, seq_len-1], 多了一维！
    # 重要步骤，例如[x1, ... , xn]重复n次成为方阵，然后和下三角矩阵做位与运算。
    # 若原本的tgt_mask中，没有pad，则做不做位与都一样；若有pad，则位与后的矩阵，右下角会提前出现False
    tgt_len = tgt.size(-1)
    triu_mask = subsequent_mask(tgt_len).type_as(tgt_mask.data)
    tgt_mask = tgt_mask & triu_mask
    return tgt_mask


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# -------------------- main -------------------- #


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # Embeddings 部分有小trick，除以了维度的根号
    # PositionalEncoding 奇数偶数位置不同，使用时会自动boardcast
    # Encoder直接把 EncoderLayer 重复N次
    # EncoderLayer 由两个 ResidualConnection 组成，此子模块封装了drop + norm + 残差连接
    # ResidualConnection[0] 是 multihead-attn，[1]是feedforward
    # 因此，输入x先经过positionalencoding，就到了multihead-attn模块
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N),
        Decoder(
            DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout),
            N,
        ),
        nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # [bsz, 1, dim]
        if trg is not None:
            self.trg = trg[:, :-1]  # 除了最后一个词
            self.trg_y = trg[:, 1:]  # 除了第一个词
            self.trg_mask = make_triu_mask_batch(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()


def run_epoch(data_iter, model, loss_compute):
    """
    Standard Training and Logging Function
    注意 这里train和eval共用一个函数。eval的时候没有decode，没有search。
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        hidden_states, prob = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(prob, batch.trg_y)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `rate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(
        model.src_embed[0].d_model,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        """
        vocab_size: 词表大小
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        """
        x: [bsz * seq_len, vocab]
        target: [bsz * seq_len]
        基本思路: 使用target构建true_dist，并与原始的x计算KL散度。生成序列每个位置
        都是在词表上的单分类，target表示答案，true_dist对答案分布做了平滑。
        """
        assert x.size(1) == self.vocab_size
        true_dist = x.data.clone()
        # 每个位置，padding和自身不算在内，词表其余词都填充
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        # 答案中真实的词，赋予1-ε
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())  # 返回x和处理后的target之间的KL散度


def data_gen(V, bsz, total_batch):
    "Generate random data for a src-tgt copy task."
    for i in range(total_batch):
        np.random.seed(i)
        data = torch.from_numpy(np.random.randint(1, V, size=(bsz, 18)))  # 一句话17个单词+[start]
        data[:, 0] = 1  # [start]
        src = data.detach()
        tgt = data.detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion  # LabelSmoothing
        self.opt = opt

    def __call__(self, prob, y):
        """
        x: [bsz, seq_len, vocab] 表示在全词表下的概率分布
        y: [bsz, seq_len]
        norm: 一个batch，target中单词的个数
        流程: 先使用labelsmoothing，将targetY平滑一些，然后使用x logit和平滑后的分布做KL散度。
        除norm是计算梯度时不受到bsz影响(torch更新后应该不用了)，乘则是真实反映bsz下的梯度信息。
        """
        loss = self.criterion(prob.contiguous().view(-1, prob.size(-1)), y.contiguous().view(-1))
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item()  # 优化的时候不能受到batch的影响，返回的时候要加回去


def greedy_decode(model, src, src_mask, max_len_tgt, start_symbol):
    """
    memory: [bsz, src_len, dim]
    src_mask: [bsz, 1, src_len]
    训练模型时，tgt是一次性输入，位置i的监督信号就是词index，预测信号是词表范围的logits
    预测时，是循环输出y，并拼接到y作为下一次的输入。和训练时的目标是一致的，因为训练时使用错位的方法进行。
    模型的输出都是[bsa, seq_len, hid]，最后跟一个Generator（线性层）将维度压缩到词表级别。
    """
    memory = model.encode(src, src_mask)
    bsz = memory.shape[0]
    pred_tgt = torch.ones(bsz, 1).fill_(start_symbol).type_as(src.data)
    pred_probs = torch.ones(bsz, 1).fill_(1.0).type_as(memory.data)
    for i in range(max_len_tgt - 1):
        hidden_states, prob = model.decode(
            memory=memory,
            src_mask=src_mask,
            tgt=pred_tgt,
            tgt_mask=subsequent_mask(pred_tgt.size(1)).type_as(
                src.data
            ),  # 下三角矩阵也是逐渐增大的，运算中不起作用
        )
        prob = prob[:, -1, :]
        next_prob, next_word = torch.max(prob, dim=1)
        pred_tgt = torch.cat([pred_tgt, next_word.type_as(pred_tgt).unsqueeze(-1)], dim=-1)
        pred_probs = torch.cat([pred_probs, next_prob.type_as(pred_probs).unsqueeze(-1)], dim=-1)
    return pred_tgt, pred_probs


def run_first_example():
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(vocab_size=V, padding_idx=0, smoothing=0.1)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(
        model.src_embed[0].d_model,
        1,
        400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )

    for epoch in range(1):
        model.train()
        run_epoch(
            data_gen(V, bsz=32, total_batch=3),
            model,
            SimpleLossCompute(criterion, model_opt),
        )
        model.eval()
        print(
            run_epoch(
                data_gen(V, 29, 5),
                model,
                SimpleLossCompute(criterion, opt=None),
            )
        )

    model.eval()

    # pred
    batch = 13
    src_len = 50
    src = torch.randint(0, V, (batch, src_len)).long()
    src_mask = torch.randint(0, 2, (batch, 1, src_len)).bool()
    pred_tgt, pred_probs = greedy_decode(model, src, src_mask, max_len_tgt=10, start_symbol=1)
    print(pred_tgt)
    print(pred_probs)


if __name__ == "__main__":
    run_first_example()
