# Data

The training script downloads [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)
through Hugging Face `datasets` and caches it
in the normal datasets cache. No dataset files are committed to this repository.
Use `--train-limit` and `--test-limit` for a quick educational run, or pass `0`
for the full split.
