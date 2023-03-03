import utils.constants as const


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith("weight_"):
                wt = getattr(lstm, name)
                wt.data.uniform_(-const.rand_unif_init_mag, const.rand_unif_init_mag)
            elif name.startswith("bias_"):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.0)
                bias.data[start:end].fill_(1.0)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=const.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=const.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=const.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-const.rand_unif_init_mag, const.rand_unif_init_mag)
