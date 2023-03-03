import html
import os
import re

from pip._internal import main
from tqdm.auto import tqdm, trange

from .common_utils import replaces


def _clean_html_special(text):
    text = html.unescape(text)
    escape_map = {
        "%21": "!",
        "%2A": "*",
        "%22": '"',
        "%27": "'",
        "%28": "(",
        "%29": ")",
        "%3B": ";",
        "%3A": ":",
        "%40": "@",
        "%26": "&",
        "%3D": "=",
        "%2B": "+",
        "%24": "$",
        "%2C": ",",
        "%2F": "/",
        "%3F": "?",
        "%25": "%",
        "%23": "#",
        "%5B": "[",
        "%5D": "]",
    }
    return replaces(text=text, maps=escape_map)


patt_tag = re.compile(r"<[^>]+>", re.S)


def _clean_tag(text):
    text = patt_tag.sub("", text)
    return text


patt_white_spaces = re.compile(" |(&nbsp)|(\u3000)+?", re.S)


def _clean_white_spaces(text):
    text = patt_white_spaces.sub(" ", text)
    text = re.sub(" +", " ", text)
    return text


def replaces_special_area_char(text):
    replace_map = {
        "🈶": "有",
        "䃼": "补",
        "⻣": "骨",
        "㖿": "耶",
        "⺠": "民",
    }
    return replaces(text=text, maps=replace_map)


re_no_chn_en_num = (
    "[^(a-zA-Z0-9\u4e00-\u9fa5 )|^(’·°–!\"#$%&'()*+,-./:;<=>?@，。?★、…【】（）：《》？“”‘’！[\\]^_`{|}~)]"
)
re_no_chn_en_num = re.compile(re_no_chn_en_num)


def find_special_char(text):
    res = re.findall(pattern=re_no_chn_en_num, string=text)
    return res


def remove_special_char(text):
    text = re.sub(pattern=re_no_chn_en_num, repl="", string=text)
    return text


def chn_to_simple(text):
    try:
        from opencc import OpenCC
    except ModuleNotFoundError:
        main(["install", "opencc-python-reimplemented"])

    # pip install opencc-python-reimplemented -U
    text = OpenCC("t2s").convert(text)
    return text


# common chinese full-width punctuation
common_zh_punctuations = [
    "。",
    "，",
    "、",
    "；",
    "：",
    "“",
    "”",
    "（",
    "）",
    "《",
    "》",
    "「",
    "」",
    "—",
    "！",
    "【",
    "】",
    "…",
    "‘",
    "’",
]
# quotation marks and comma cannot be matched
common_en_punctuations = [
    ".",
    ",",
    ",",
    ";",
    ":",
    '"',
    '"',
    "(",
    ")",
    "<",
    ">",
    "{",
    "}",
    "-",
    "!",
    "[",
    "]",
    "...",
    "'",
    "'",
]
common_zn_en_punc_map = {zh: en for zh, en in zip(common_zh_punctuations, common_en_punctuations)}
common_en_zh_punc_map = {
    en: zh
    for en, zh in zip(common_en_punctuations, common_zh_punctuations)
    if en not in ['"', "'"] and zh not in ["、"]
}
# all chinese
all_chn = "[\u4e00-\u9fa5]"


def Q2B_char(uchar, excludes=common_zh_punctuations):
    if uchar in set(excludes):
        return uchar
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xFEE0
    if inside_code < 0x0020 or inside_code > 0x7E:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def B2Q_char(uchar, excludes=[]):
    if uchar in set(excludes):
        return uchar
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7E:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xFEE0
    return chr(inside_code)


def Q2B_string(text, excludes=common_zh_punctuations):
    text = [Q2B_char(c, excludes) for c in text]
    text = "".join(text)
    return text


def B2Q_string(text, excludes=[]):
    text = [B2Q_char(c, excludes) for c in text]
    text = "".join(text)
    return text


def clean_html_pipline(text):
    if text:
        text = _clean_html_special(text)
        text = _clean_tag(text)
        text = _clean_white_spaces(text)
        text = chn_to_simple(text)
        text = replaces(text, common_en_zh_punc_map)
        text = Q2B_string(text)
    return text


def save_doc_to_docx(rawpath, target_path):  # doc转docx
    from win32com import client as wc

    word = wc.Dispatch("Word.Application")
    # cannot use relative path, use absolute path
    for path in tqdm(os.listdir(rawpath), ncols=100):
        # find files ending with .doc and not starting with ~$ (~$ is to exclude temporary files)
        if path.endswith(".doc") or path.endswith(".docx") and not path.startswith("~$"):
            # print(i)
            doc = word.Documents.Open(rawpath + path)
            # split the file name and suffix
            rename = os.path.splitext(path)
            # save as .docx
            doc.SaveAs(target_path + rename[0] + ".docx", 12)  # 12表示docx格式
            doc.Close()
    word.Quit()


STOP_WORDs = [
    "",
    "\n",
    "!",
    '"',
    "\"'",
    '"\'"',
    '",',
    '".',
    "#",
    "&",
    "'",
    "'\"",
    "'.",
    "'s",
    "(",
    "(,",
    "(;",
    ")",
    "),",
    ").",
    "):",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "[",
    "]",
    "a",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "she",
    "the",
    "this",
    "to",
    "was",
    "–",
    "†",
    "•",
    "→",
    "=",
    "+",
    "?",
    "_",
    " ",
]