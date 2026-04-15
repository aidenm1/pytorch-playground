from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that drops HTML and punctuation."""
    return TOKEN_RE.findall(re.sub(r"<[^>]+>", " ", text.lower()))


@dataclass
class Vocabulary:
    stoi: dict[str, int]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_size: int = 30_000) -> "Vocabulary":
        counts = Counter(token for text in texts for token in tokenize(text))
        tokens = sorted(counts, key=lambda token: (-counts[token], token))
        tokens = [token for token in tokens if counts[token] >= min_freq][: max_size - 2]
        return cls({token: idx for idx, token in enumerate([PAD_TOKEN, UNK_TOKEN, *tokens])})

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.stoi.get(token, self.unk_idx) for token in tokenize(text)[:max_length]]
        return ids or [self.unk_idx]

    def __len__(self) -> int:
        return len(self.stoi)


class ReviewDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocabulary, max_length: int = 300):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        self.texts, self.labels, self.vocab, self.max_length = texts, labels, vocab, max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int, str]:
        text = self.texts[index]
        return self.vocab.encode(text, self.max_length), int(self.labels[index]), text


def make_collate_fn(pad_idx: int):
    def collate(batch):
        sequences, labels, texts = zip(*batch)
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(sequence, dtype=torch.long) for sequence in sequences],
            batch_first=True,
            padding_value=pad_idx,
        )
        return padded, lengths, torch.tensor(labels, dtype=torch.long), list(texts)
    return collate


def load_imdb(train_limit: int = 10_000, test_limit: int = 5_000, seed: int = 42):
    """Return shuffled IMDb train/test text and labels. A limit of 0 uses all rows."""
    from datasets import load_dataset

    # Use the fully-qualified Hub repository ID. Recent huggingface_hub releases
    # no longer accept the historical single-segment "imdb" alias everywhere.
    data = load_dataset("stanfordnlp/imdb")
    train, test = data["train"].shuffle(seed=seed), data["test"].shuffle(seed=seed)
    if train_limit:
        train = train.select(range(min(train_limit, len(train))))
    if test_limit:
        test = test.select(range(min(test_limit, len(test))))
    return list(train["text"]), list(train["label"]), list(test["text"]), list(test["label"])
