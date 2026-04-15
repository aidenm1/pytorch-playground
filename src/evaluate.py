from __future__ import annotations

import torch

from .utils import binary_metrics


def evaluate(model, loader, criterion, device, collect_errors: int = 10):
    model.eval()
    total_loss, count, truths, predictions, errors = 0.0, 0, [], [], []
    with torch.no_grad():
        for inputs, lengths, labels, texts in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs, lengths)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            count += labels.size(0)
            preds = logits.argmax(dim=1).cpu().tolist()
            actual = labels.cpu().tolist()
            truths.extend(actual)
            predictions.extend(preds)
            for text, truth, pred in zip(texts, actual, preds):
                if truth != pred and len(errors) < collect_errors:
                    errors.append({"actual": truth, "predicted": pred, "review": text})
    metrics = binary_metrics(truths, predictions)
    metrics["loss"] = total_loss / max(count, 1)
    return metrics, errors

