# Movie Review Sentiment Classification with PyTorch

An educational, end-to-end IMDb sentiment project: text becomes token IDs, learned
embeddings, an LSTM representation, and positive/negative logits. It includes a
TF-IDF + logistic-regression baseline, a manual PyTorch training loop, evaluation,
plots, error analysis, checkpointing, and inference.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m src.train --epochs 3
python -m src.predict "Slow at first, but ultimately a wonderful film."
```

The first training run downloads IMDb. Defaults use 10,000 training and 5,000 test
reviews. For a smoke run use `--train-limit 1000 --test-limit 500 --epochs 1`; use
`--train-limit 0 --test-limit 0` for all 50,000 labeled reviews. The compute device
is selected in this order: NVIDIA CUDA, Apple Metal (MPS), then CPU. Override it
with `--device cpu`, `--device cuda`, or `--device mps`.

Outputs in `results/` include the best `model.pt`, `metrics.json`,
`training_curves.png`, `confusion_matrix.png`, and `errors.txt`.

## Architecture and data flow

IMDb review → lowercase tokenizer → training vocabulary → padded batch →
`Embedding(128)` → packed `LSTM(128)` → dropout → `Linear(128, 2)`.

Only the training split builds the vocabulary. `<PAD>` and `<UNK>` occupy IDs 0
and 1. Packed sequences prevent padding from affecting the final LSTM state.
`CrossEntropyLoss` consumes raw logits, so the model does not apply softmax while
training.

## Experiments

Run controlled comparisons by changing one option at a time:

```bash
python -m src.train --optimizer adam --hidden-dim 64 --output-dir results/adam-h64
python -m src.train --optimizer sgd  --hidden-dim 64 --output-dir results/sgd-h64
python -m src.train --optimizer adam --hidden-dim 128 --output-dir results/adam-h128
python -m src.train --optimizer adam --hidden-dim 128 --dropout 0 --output-dir results/no-dropout
```

Compare `accuracy` and `f1` in each `metrics.json`, while inspecting validation
curves for convergence and overfitting. Seeds, split, architecture, limits, and
learning settings are recorded with every report. Actual scores are intentionally
not claimed in this repository: they depend on the selected sample, epochs, device,
and package versions.

## Evaluation

After each epoch, evaluation uses `model.eval()` and `torch.no_grad()`. Final test
metrics are accuracy, precision, recall, F1, loss, and a confusion matrix. The
baseline and LSTM use the same held-out test sample. Misclassified review examples
are written to `errors.txt`; negation, sarcasm, mixed sentiment, and long reviews
are useful patterns to inspect.

## PyTorch concepts demonstrated

| Concept | Where demonstrated |
|---|---|
| Tensors | Batches, labels, logits, loss |
| `Dataset` | `ReviewDataset` in `src/dataset.py` |
| `DataLoader` | Mini-batches in `src/train.py` |
| `nn.Module` | `SentimentLSTM` |
| `nn.Embedding` | Learned token representation |
| `nn.LSTM` | Packed sequence processing |
| Forward pass | `SentimentLSTM.forward` |
| Loss | `CrossEntropyLoss` |
| Autograd | `loss.backward()` |
| Optimizer | Adam or SGD, followed by `step()` |
| `model.train()` | Training epoch |
| `model.eval()` / `no_grad()` | Evaluation and prediction |
| CPU/GPU | Automatic CUDA/MPS/CPU selection and tensor transfers |
| Saving/loading | Best validation checkpoint and CLI inference |

## What the mechanics mean

- **Embedding:** a trainable lookup table mapping discrete words to dense vectors.
- **Batching:** processing several variable-length reviews together after padding.
- **Forward propagation:** computing logits from inputs with current parameters.
- **Loss:** a differentiable measure of prediction error.
- **Gradients/backpropagation:** derivatives computed by autograd after `backward()`.
- **Optimization:** using gradients to update parameters with Adam or SGD.
- **Train vs evaluation:** modes that control layers such as dropout.
- **Overfitting:** training scores improve while validation scores stagnate or fall.

## Development

```bash
pytest -q
```

`notebooks/exploration.ipynb` provides a lightweight interactive entry point. The
source modules remain the canonical implementation so notebook results are
reproducible from the command line.
