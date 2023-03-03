def init_nltk(lower=False):
    """
    usage:
        tokenize = init_nltk()
        tokenize(text="hello, this. is good")
    """
    from nltk.tokenize import wordpunct_tokenize

    tokenizer = wordpunct_tokenize

    def _tokenize(text, **kwargs):
        text = text.lower() if lower else text
        words = list(tokenizer(text))
        return words

    return _tokenize


def init_spacy(lower=False, keep_sent=False):
    """
    mode in ["tokenize"]
    usage:
        tokenize = init_spacy(lower=False,keep_sent=False)
        tokenize(text="hello, this. is good")
    """
    import spacy

    nlp = spacy.load("en_core_web_sm")

    def _tokenize(text, lemma=False):
        words = []
        text = str(text).lower() if lower else str(text)
        try:
            if keep_sent:
                words = [sent.text for sent in nlp(text).sents]
            else:
                words = [tok.lemma_ if lemma else tok.text for sent in nlp(text).sents for tok in sent]
        except Exception as e:
            print(text, e)
        return words

    return _tokenize


def init_bert(path="/home/xionggm/pretrained_models/bert-base-uncased"):
    """
    usage:
        tokenize = init_bert()
        tokenize(text="hello")
    trick:
        delete Fast and press F12, you will see the doc.
    use it to tokenize one sentence.
    """
    from transformers import BertTokenizerFast as BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(path)
    print(f"Load successfully: {path}")

    def _tokenize(text, return_tokens=False, **kwargs):
        out = tokenizer.__call__(
            text=text,
            padding=False,
            truncation="longest_first",
            max_length=128,
            stride=0,
            # return_tensors="pt",
            verbose=True,
            add_special_tokens=False,
        )
        if return_tokens:
            words = tokenizer.tokenize(text=text)
            return out, words
        else:
            return out

    return _tokenize


if __name__ == "__main__":
    tokenize = init_nltk()
    x = tokenize(text="hello, this. is good")
    print(x)
