import torch
import sys
from argparse import Namespace
from types import SimpleNamespace
from pathlib import Path

from src.dataset import ReviewDataset, Vocabulary, load_imdb, make_collate_fn, tokenize
from src.model import SentimentLSTM
from src.utils import get_device


def test_tokenize_removes_html_and_lowercases():
    assert tokenize("<b>A GREAT movie!</b>") == ["a", "great", "movie"]


def test_vocabulary_unknown_and_empty_review():
    vocab = Vocabulary.build(["good movie", "bad movie"], min_freq=1, max_size=10)
    assert vocab.encode("unseen", 5) == [vocab.unk_idx]
    assert vocab.encode("", 5) == [vocab.unk_idx]


def test_batch_and_model_backward_pass():
    texts, labels = ["good movie", "very bad movie"], [1, 0]
    vocab = Vocabulary.build(texts, min_freq=1)
    dataset = ReviewDataset(texts, labels, vocab)
    tokens, lengths, targets, _ = make_collate_fn(vocab.pad_idx)([dataset[0], dataset[1]])
    model = SentimentLSTM(len(vocab), vocab.pad_idx, embedding_dim=8, hidden_dim=6)
    logits = model(tokens, lengths)
    assert logits.shape == (2, 2)
    torch.nn.CrossEntropyLoss()(logits, targets).backward()
    assert model.embedding.weight.grad is not None


def test_imdb_loader_uses_fully_qualified_hub_id(monkeypatch):
    requested = []

    class FakeSplit(dict):
        def __len__(self):
            return len(self["text"])

        def shuffle(self, seed):
            return self

        def select(self, indices):
            indices = list(indices)
            return FakeSplit({key: [values[i] for i in indices] for key, values in self.items()})

    def fake_load_dataset(repo_id):
        requested.append(repo_id)
        return {
            "train": FakeSplit(text=["good", "bad"], label=[1, 0]),
            "test": FakeSplit(text=["great", "awful"], label=[1, 0]),
        }

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    train_texts, _, test_texts, _ = load_imdb(train_limit=1, test_limit=1)
    assert requested == ["stanfordnlp/imdb"]
    assert train_texts == ["good"]
    assert test_texts == ["great"]


def test_copying_config_for_json_does_not_mutate_runtime_path():
    cfg = Namespace(output_dir=Path("results"), epochs=1)
    serialized = vars(cfg).copy()
    serialized["output_dir"] = str(cfg.output_dir)
    assert isinstance(cfg.output_dir, Path)
    assert serialized["output_dir"] == "results"


def test_device_auto_selects_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device().type == "mps"


def test_forcing_unavailable_mps_has_clear_error(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    try:
        get_device("mps")
    except RuntimeError as error:
        assert "no usable MPS device" in str(error)
    else:
        raise AssertionError("expected unavailable MPS to raise")
