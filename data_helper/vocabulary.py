import json
import os
import tempfile
from collections import Counter, defaultdict

# from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"

# from allennlp.common.util import START_SYMBOL, END_SYMBOL
START_SYMBOL = "@start@"
END_SYMBOL = "@end@"


class Vocab:
    """
    main funtions:
        1. keep raw words count info in order to trim to any frequence when needed.
        2. multi-process build support.
        3. allennlp support.
    """

    def __init__(self, name="defalut"):
        self.name = name
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.max_size = None
        self.min_freq = 1

        self.pad_token = DEFAULT_PADDING_TOKEN
        self.unk_token = DEFAULT_OOV_TOKEN
        self.sos_token = START_SYMBOL
        self.eos_token = END_SYMBOL

        self.raw_counter = defaultdict(int)
        self.reserved = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        self.word2index = dict(zip(self.reserved, range(len(self.reserved))))
        self.index2word = {v: k for k, v in self.word2index.items()}

    def build_from_counter(self, vocab_counter: dict, max_size=None, min_freq=1):
        """
        vocab_counter: {"a": 1, "b": 2, ...}
        """
        self.raw_counter = vocab_counter
        self._add_words(vocab_counter.keys())
        self.trim(max_size=max_size, min_freq=min_freq)

    def build_from_list(self, words: list, max_size=None, min_freq=1):
        vocab_counter = Counter(words)
        self.build_from_counter(vocab_counter=vocab_counter, max_size=max_size, min_freq=min_freq)

    def build_from_counters(self, vocab_counters: list, max_size=None, min_freq=1):
        """
        vocab_counters: [{"a": 1, "b": 2, ...}, {"a": 1, "c": 2, ...}]
        """
        self.raw_counter = defaultdict(int)
        for counter in vocab_counters:
            for word, freq in counter.items():
                self.raw_counter[word] += freq
        self._add_words(self.raw_counter.keys())
        self.trim(max_size=max_size, min_freq=min_freq)

    def _add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word[len(self.index2word)] = word
        assert len(self.word2index) == len(self.index2word)

    def trim(self, max_size: int = None, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        if min_freq <= 1 and (max_size is None or max_size >= len(self.word2index)):
            return
        max_size = max_size - len(self.reserved)
        ordered_words = sorted(((c, w) for (w, c) in self.raw_counter.items()), reverse=True)

        ordered_words = ordered_words[:max_size]
        self.word2index = dict(zip(self.reserved, range(len(self.reserved))))

        for count, word in ordered_words:
            if count < min_freq:
                break
            self.word2index.setdefault(word, len(self.word2index))

        self.index2word = {v: k for k, v in self.word2index.items()}

    def get_vocab_size(self):
        return len(self.index2word)

    @property
    def size(self):
        return self.get_vocab_size()

    def getIndex(self, word, oov_dict={}):
        """
        oov_dict: {"a":0, "b":1}  and vocab size: 100 --> b idx is 101
        """
        if word in self.word2index:
            return self.word2index[word]
        elif word in oov_dict:
            return oov_dict[word] + len(self)
        else:
            return self.UNK

    def getWord(self, idx, oov_dict={}, _done=False):
        """
        if _done, oov_dict is id-to-word
        """
        assert idx >= 0, f"{idx} must >= 0 !"
        if not _done:
            oov_dict = {v: k for k, v in oov_dict.items()}

        if idx < len(self.index2word):
            return self.index2word[idx]
        elif idx < len(self.index2word) + len(oov_dict):
            return oov_dict[idx]
        else:
            return self.unk_token

    def get_raw_count(self, word):
        return self.raw_counter.get(word, 0)

    def to_word_sequence(self, seq, oov_dict={}, stop_at_end=False):
        if stop_at_end and self.EOS in seq:
            seq = seq[: seq.index(self.EOS)]
        oov_dict = {v: k for k, v in oov_dict.items()}
        sentence = [self.getWord(idx, oov_dict=oov_dict, _done=True) for idx in seq]
        return sentence

    def to_index_sequence(self, words, oov_dict={}):
        seq = [self.getIndex(word, oov_dict=oov_dict) for word in words]
        return seq

    def is_oov(self, word):
        return word not in self.word2index

    def save_to_file(self, path):
        content = {
            "name": self.name,
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "raw_counter": self.raw_counter,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f1:
            json.dump(content, f1, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_file(cls, path):
        with open(path) as f1:
            content = json.load(f1)
        vocab = cls(name=content["name"])
        vocab.build_from_counter(
            content["raw_counter"],
            max_size=content["max_size"],
            min_freq=content["min_freq"],
        )
        return vocab

    def save_for_allennlp(self, dirname, namespace=None):
        """
        dirname:
            non_padded_namespaces.txt (*tags *labels)
            {name}.txt
        """
        namespace = namespace if namespace else self.name
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "non_padded_namespaces.txt"), "w", encoding="utf-8") as f1:
            f1.write(f"*tags\n*labels\n{namespace}\n")
        with open(os.path.join(dirname, f"{namespace}.txt"), "w", encoding="utf-8") as f1:
            for i in range(len(self.index2word)):
                f1.write(self.index2word[i] + "\n")

    def to_allennlp(self, namespace="defalut"):
        from allennlp.data import Vocabulary

        tmp = tempfile.TemporaryDirectory(dir="./")
        self.save_for_allennlp(dirname=tmp.name, namespace=namespace)
        # tmp.flush()
        vocab = Vocabulary.from_files(
            directory=tmp.name,
            padding_token=self.pad_token,
            oov_token=self.unk_token,
        )
        tmp.cleanup()
        return vocab

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def __repr__(self) -> str:
        return f"Vocab {self.name}. length: {len(self.word2index)}  min freq: {self.min_freq}."

    def __str__(self) -> str:
        return self.__repr__()


def init_tokenizer():
    """
    usage:
        tokenize = tokenizer()
        tokenize(text="hello")
    """
    from jieba import Tokenizer

    tokenizer = Tokenizer()

    def _tokenize(text):
        words = list(tokenizer.cut(text))
        return words

    return _tokenize

