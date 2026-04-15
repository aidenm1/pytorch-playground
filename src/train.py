from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ReviewDataset, Vocabulary, load_imdb, make_collate_fn
from .evaluate import evaluate
from .model import SentimentLSTM
from .utils import binary_metrics, get_device, save_json, seed_everything


def run_baseline(train_texts, train_labels, test_texts, test_labels):
    vectorizer = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(x_train, train_labels)
    return binary_metrics(test_labels, model.predict(vectorizer.transform(test_texts)).tolist())


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for inputs, lengths, labels, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


def save_plots(history, metrics, output_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="LSTM confusion matrix")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close(fig)


def main(args=None):
    parser = argparse.ArgumentParser(description="Train an IMDb LSTM sentiment classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="compute device (default: auto-detect CUDA, then Apple MPS, then CPU)")
    parser.add_argument("--skip-baseline", action="store_true")
    cfg = parser.parse_args(args)
    seed_everything(cfg.seed); cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device); print(f"Using device: {device}")
    train_texts, train_labels, test_texts, test_labels = load_imdb(cfg.train_limit, cfg.test_limit, cfg.seed)
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=cfg.seed, stratify=train_labels
    )
    baseline = None if cfg.skip_baseline else run_baseline(tr_texts, tr_labels, test_texts, test_labels)
    vocab = Vocabulary.build(tr_texts)
    collate = make_collate_fn(vocab.pad_idx)
    def loader(texts, labels, shuffle=False):
        return DataLoader(ReviewDataset(texts, labels, vocab, cfg.max_length), batch_size=cfg.batch_size,
                          shuffle=shuffle, collate_fn=collate, pin_memory=device.type == "cuda")
    train_loader, val_loader, test_loader = loader(tr_texts, tr_labels, True), loader(val_texts, val_labels), loader(test_texts, test_labels)
    model = SentimentLSTM(len(vocab), vocab.pad_idx, cfg.embedding_dim, cfg.hidden_dim, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = cfg.learning_rate or (1e-3 if cfg.optimizer == "adam" else 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if cfg.optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=lr)
    history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}
    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device)
        for key, value in (("train_loss", train_loss), ("train_accuracy", train_acc),
                           ("val_loss", val_metrics["loss"]), ("val_accuracy", val_metrics["accuracy"])):
            history[key].append(value)
        print(f"Epoch {epoch + 1}/{cfg.epochs}: train loss={train_loss:.4f}, val loss={val_metrics['loss']:.4f}, val accuracy={val_metrics['accuracy']:.3f}")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"model_state": model.state_dict(), "vocab": vocab.stoi,
                        "config": {"embedding_dim": cfg.embedding_dim, "hidden_dim": cfg.hidden_dim,
                                   "dropout": cfg.dropout, "max_length": cfg.max_length}}, cfg.output_dir / "model.pt")
    checkpoint = torch.load(cfg.output_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, errors = evaluate(model, test_loader, criterion, device)
    # vars(cfg) is the namespace's live dictionary. Copy it before making Path
    # values JSON-safe so cfg.output_dir remains a Path for subsequent writes.
    serialized_config = vars(cfg).copy()
    serialized_config["output_dir"] = str(cfg.output_dir)
    report = {"baseline": baseline, "lstm": test_metrics, "history": history,
              "config": serialized_config}
    save_json(report, cfg.output_dir / "metrics.json")
    (cfg.output_dir / "errors.txt").write_text("\n\n".join(
        f"Actual: {'positive' if x['actual'] else 'negative'}\nPredicted: {'positive' if x['predicted'] else 'negative'}\n{x['review']}" for x in errors
    ), encoding="utf-8")
    save_plots(history, test_metrics, cfg.output_dir)
    print(json.dumps({"baseline": baseline, "lstm": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
