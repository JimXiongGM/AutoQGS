from collections import Counter


def plot_freq_scatter(lens: list):
    import matplotlib.pyplot as plt

    cc = Counter(lens)
    fig, ax = plt.subplots()
    ax.scatter(cc.keys(), cc.values(), label="freq")
    for x, y in cc.items():
        ax.text(x, y, s="%d" % y, ha="center", va="bottom", fontsize=9)

    ax.set_title("frequency distribution")
    return ax


def cal_freq_dist(lens, orderby="most_common"):
    """
    calculate the cumulative distribution or cumulative distribution
    orderby:
        1. most_common
        2. number_line
    """
    assert orderby in [
        "most_common",
        "number_line",
    ], f"orderby: {orderby} not in [`most_common`,`number_line`]"
    if not lens:
        return {}
    total = len(lens)
    c = Counter(lens)

    if orderby == "most_common":
        c = c.most_common()
    else:
        try:
            c = sorted(c.items(), key=lambda x: int(x[0]))
        except:
            c = sorted(c.items(), key=lambda x: x[0])

    cum = 0
    cum_dict = {}
    for i in c:
        v, freq = i
        cum += freq
        cum_dict[v] = float(cum) / total
    return cum_dict


if __name__ == "__main__":
    ax = plot_freq_scatter(lens=[1, 2, 3, 4, 4, 1, 2])
    # ax.figure.savefig("./123.png")
