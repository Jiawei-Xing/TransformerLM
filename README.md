# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Run model
Model source codes are updated in `cs336_basics` and has passed most tests except for the time and memory limits of BPE. Function modules are built from scratch instead of importing from PyTorch!

### BPE
First, download datasets in `../data/` as described above. To train a byte pair encoding model on `TinyStoriesV2-GPT4-train.txt` as an example:
```
python -m cs336_basics.train_bpe
```
This will generate two files: `../results/vocab.json` (token vocabulary with IDs) and `../results/merges.txt` (merges of token pairs in order).

### Tokenization
To run a tokenizer on `TinyStoriesV2-GPT4-train.txt`:
```
python -m cs336_basics.tokenizer
```
This will encode the text file into token IDs.

### Model training
To train the transformer-based language model:
```
python -m cs336_basics.training_together --input "../results/tokens.npy" --checkpoint "../results/checkpoint.pt"
```
This will generate a trained model in the checkpoint file.

Here we use an example model including 4 transformer blocks with 16 heads multi-head attention, as well as classic pre-Norm, SwiGLU activation, and rotary position encoding. For training, we use AdamW optimizer with cosine annealing learning rate scheduling and gradient clipping. Users are encouraged to test different parameters. To list all parameters, run `python -m cs336_basics.training_together --help`.

### Text generation
To generate text from an initial sentence using the model:
```
python -m cs336_basics.decoding --init "Once upon a time, " --checkpoint "../results/checkpoint.pt" --vocab "../results/vocab.json" --merges "../results/merges.txt"
```
This will print out a tiny story. Here is one example:
```
Once upon a time, there was a little boy named Tom. Tom liked to play outside with his friends. One day, Tom and his friends decided to have a big party. They invited all their friends and family. The party was very happy to go to the party for the guests.
At the party, they played games, ate yummy treats, and had lots of fun. Tom was not lay anymore. He played all day long and had lots...
```
Not perfect but a good start!

To list all parameters, run `python -m cs336_basics.decoding --help`.
