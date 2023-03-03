import re


def colorful(text, color="yellow"):
    if color == "yellow":
        text = "\033[1;33;33m" + text + "\033[0m"
    elif color == "grey":
        text = "\033[1;30;47m" + text + "\033[0m"
    elif color == "green":
        text = "\033[1;32;32m" + text + "\033[0m"
    else:
        pass
    return text


def find_all_spans(substr, text):
    """
    given a substring and a text, return char offsets
    """
    res = [(m.start(), m.start() + len(substr)) for m in re.finditer(substr, text)]
    return res


def check_offsets_overlap(offsets):
    """
    check if the offsets have overlap
    if there is overlap, return True
    """
    for item in offsets:
        assert len(item) >= 2
    # sort by start, check if there is overlap
    offsets.sort(key=lambda x: x[0])
    for index in range(len(offsets) - 1):
        if offsets[index][1] > offsets[index + 1][0]:
            return True
    return False


def drop_offsets_overlap(offsets):
    """
    if there is overlap, drop the second one
    """
    for item in offsets:
        assert len(item) >= 2
    offsets.sort(key=lambda x: x[0])
    new_offsets = [offsets[0]]
    p1, p2 = 0, 1
    while p2 < len(offsets):
        # overlap!
        if offsets[p1][1] > offsets[p2][0]:
            p2 += 1
        # ok
        else:
            new_offsets.append(offsets[p2])
            p1 = p2
            p2 += 1
    return new_offsets


def insert_by_offsets(text, func, offsets):
    """
    update text by offsets and func
    """
    if not offsets:
        return text
    assert check_offsets_overlap(offsets) == False
    # insert text in reverse order
    offsets.sort(key=lambda x: x[0])
    last = offsets[-1]
    color = last[2] if len(last) > 2 else "yellow"
    new_srt = func(text[last[0] : last[1]], color) + text[last[1] :]
    for index in list(range(len(offsets) - 1))[::-1]:
        current = offsets[index]
        last = offsets[index + 1]
        color = current[2] if len(current) > 2 else "green"
        new_srt = func(text[current[0] : current[1]], color) + text[current[1] : last[0]] + new_srt
    new_srt = text[: offsets[0][0]] + new_srt
    return new_srt

