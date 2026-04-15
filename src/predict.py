from __future__ import annotations

import argparse

import torch

from .dataset import Vocabulary
from .model import SentimentLSTM
from .utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Classify a movie review")
    parser.add_argument("text")
    parser.add_argument("--checkpoint", default="results/model.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab, cfg = Vocabulary(checkpoint["vocab"]), checkpoint["config"]
    model = SentimentLSTM(len(vocab), vocab.pad_idx, cfg["embedding_dim"], cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    ids = torch.tensor([vocab.encode(args.text, cfg["max_length"])], device=device)
    lengths = torch.tensor([ids.size(1)])
    with torch.no_grad():
        probabilities = model(ids, lengths).softmax(dim=1)[0]
    prediction = int(probabilities.argmax())
    print(f"{'positive' if prediction else 'negative'} ({probabilities[prediction].item():.1%} confidence)")


if __name__ == "__main__":
    main()
