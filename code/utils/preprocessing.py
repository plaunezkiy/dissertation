import Stemmer
import re

stemmer = Stemmer.Stemmer("english")
alphanum = r"[a-zA-Z0-9_-]*"
non_alphanum = r"[^a-zA-Z0-9]"


def preprocess_text(text, join=True):
    # tokenize
    text = text.lower()
    text = re.sub(non_alphanum, "\n", text)
    # print(text.split("\n"))
    tokens = list(filter(lambda token: bool(token), text.split("\n")))
    # normalize
    tokens = list(map(lambda token: stemmer.stemWord(token), tokens))
    if join:
        return " ".join(tokens)
    return tokens
